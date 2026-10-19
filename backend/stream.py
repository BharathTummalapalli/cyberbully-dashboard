"""
Streaming Ingestion and WebSocket Broadcaster Module.
Mirrors an asynchronous Kafka producer/consumer architecture:
- Producer: Pushes raw social media events from tweets.csv into an asyncio.Queue.
- Consumer: Pulls events from the queue, executes Toxic-BERT / Lexicon scoring,
            evaluates the Rule-Based Context Adjustment Layer, applies Additive Clamped Fusion,
            persists records to the 5-table schema, and broadcasts live events over WebSocket (/ws/feed).
- WebSocket Manager: Manages active client connections with origin verification.
"""

import os
import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import WebSocket, WebSocketDisconnect

import db
import explain
import fusion
from utils import resolve_target

logger = logging.getLogger(__name__)

# Allowed WebSocket origins (derived from environment)
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_STR.split(",") if o.strip()]

class WebSocketConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> bool:
        """
        Validates origin header against ALLOWED_ORIGINS during handshake.
        Accepts valid connections; rejects unauthorized cross-origin attempts.
        """
        origin = websocket.headers.get("origin")
        # In dev mode, permit connections from localhost/127.0.0.1 or empty (e.g. tools/scripts)
        if origin:
            is_allowed = any(origin.rstrip("/") == allowed.rstrip("/") for allowed in ALLOWED_ORIGINS)
            if not is_allowed and "localhost" not in origin and "127.0.0.1" not in origin:
                logger.warning(f"Rejected WebSocket connection from unauthorized origin: {origin}")
                await websocket.close(code=1008)  # Policy violation
                return False

        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected from {origin or 'direct client'}. Total active: {len(self.active_connections)}")
        return True

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcasts JSON payload to all active WebSocket clients concurrently."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error sending message to WebSocket client: {e}")
                disconnected.append(connection)

        for dead_conn in disconnected:
            self.disconnect(dead_conn)

ws_manager = WebSocketConnectionManager()

class StreamIngestionPipeline:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.is_streaming: bool = False
        self.interval: float = 3.0  # seconds between simulated events
        self.producer_task: Optional[asyncio.Task] = None
        self.consumer_task: Optional[asyncio.Task] = None
        self.tweets_df: Optional[pd.DataFrame] = None
        self.current_index: int = 0

    def load_dataset(self):
        """Loads tweets.csv for streaming simulation."""
        csv_path = os.path.join(os.path.dirname(__file__), "tweets.csv")
        if os.path.exists(csv_path):
            self.tweets_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.tweets_df)} simulation tweets from {csv_path}")
        else:
            logger.warning(f"Simulation dataset not found at {csv_path}")
            self.tweets_df = pd.DataFrame()

    async def producer_loop(self):
        """Asynchronously feeds tweets from the dataset into the queue (Producer)."""
        logger.info("Starting Streaming Ingestion Producer...")
        while self.is_streaming:
            try:
                if self.tweets_df is not None and len(self.tweets_df) > 0:
                    row = self.tweets_df.iloc[self.current_index % len(self.tweets_df)]
                    self.current_index += 1
                    
                    item = {
                        "text": str(row["text"]),
                        "author": str(row["username"]) if "username" in row else "@anonymous",
                        "user_id": str(row["user_id"]) if "user_id" in row else "user_sim",
                        "parent_username": str(row["parent_username"]) if "parent_username" in row and not pd.isna(row["parent_username"]) else None
                    }
                    await self.queue.put(item)
                    
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in producer loop: {e}", exc_info=True)
                await asyncio.sleep(2.0)

    async def consumer_loop(self):
        """Pulls events from the queue, runs classification & fusion, logs to DB, and broadcasts (Consumer)."""
        logger.info("Starting Streaming Ingestion Consumer...")
        while self.is_streaming:
            try:
                item = await self.queue.get()
                processed_event = await self.process_event(item)
                await ws_manager.broadcast(processed_event)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def process_event(self, item: Dict[str, Any], use_real_model: Optional[bool] = None) -> Dict[str, Any]:
        """
        Processes a single event end-to-end through the Learned Context-Aware Pipeline:
        1. Resolves target
        2. Context Collector (Conversation, Behavioral, Temporal, Graph)
        3. Toxic-BERT (P_toxic) + Contextual DistilBERT (P_context + multi-task heads)
        4. Learned Fusion MLP (P_final) + Decision Engine
        5. Logs to database & broadcasts to WebSocket subscribers
        """
        from inference.pipeline import pipeline

        text = item["text"]
        author = item.get("author", "@anonymous")
        parent_user = item.get("parent_username")

        if use_real_model is None:
            use_real_model = explain.ML_AVAILABLE

        res = pipeline.process(
            text=text,
            author=author,
            target_user=parent_user,
            use_real_model=use_real_model,
            skip_lime=False,
            persist=True
        )

        return {
            **res,
            "user_id": item.get("user_id", "user_guest"),
            # Backward-compatible fields
            "delta_neg": 0.0,
            "delta_slang": 0.0,
            "delta_sarcasm": 0.0,
            "delta_target": 0.0,
            "total_context_delta": round(res["final_score"] - res["p_toxic"], 4),
            "formula": f"MLP_Fusion(P_tox={res['p_toxic']:.3f}, P_ctx={res['p_context']:.3f}, V_1h={res['context_features']['v_1h']}) = {res['final_score']:.3f}"
        }

    def start(self, interval: Optional[float] = None):
        """Starts background streaming worker tasks."""
        if self.is_streaming:
            return
        if interval:
            self.interval = interval
            
        if self.tweets_df is None:
            self.load_dataset()

        self.is_streaming = True
        loop = asyncio.get_event_loop()
        self.producer_task = loop.create_task(self.producer_loop())
        self.consumer_task = loop.create_task(self.consumer_loop())
        logger.info("Streaming ingestion pipeline activated.")

    def stop(self):
        """Stops background streaming worker tasks."""
        self.is_streaming = False
        if self.producer_task and not self.producer_task.done():
            self.producer_task.cancel()
        if self.consumer_task and not self.consumer_task.done():
            self.consumer_task.cancel()
        logger.info("Streaming ingestion pipeline halted.")

    def set_speed(self, interval: float):
        self.interval = max(0.5, min(15.0, interval))

# Global streaming pipeline singleton
stream_pipeline = StreamIngestionPipeline()
