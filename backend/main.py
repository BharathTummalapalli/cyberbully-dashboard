"""
FastAPI Main Application Server.
Provides:
- JWT Authentication & RBAC (Admin / Moderator)
- CORS & WebSocket Handshake with Origin Verification (/ws/feed)
- Streaming Ingestion Queue (Producer/Consumer)
- Human-in-the-Loop Review Queue (/queue)
- Fairness & Bias Audit Reporting (/fairness)
- Model Explainability (LIME) & NetworkX Graph Centrality
- Dual-path routing (accessible via both /api/* and /* for proxy resilience)
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd

import db
import auth
import explain
import fusion
import graph
import fairness
import evaluate
from stream import ws_manager, stream_pipeline
from utils import resolve_target

# Configure standard logging format and levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="CyberShield: Context-Aware Cyberbullying Detection System API",
    description="Multi-tier API supporting JWT auth, 5-table schema, WebSocket streaming, score fusion, and fairness audits.",
    version="2.0.0"
)

# CORS Configuration
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000,http://localhost,http://localhost:80")
allowed_origins = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",") if origin.strip()]
logger.info(f"Configured CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
USE_REAL_MODEL = explain.ML_AVAILABLE

# Pydantic Schemas
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "moderator"

class PredictionRequest(BaseModel):
    text: str
    user_id: str = "user_guest"
    username: str = "@anonymous"
    target_user: str = "@anonymous"

class ActionRequest(BaseModel):
    action: str
    notes: Optional[str] = None

class StreamSpeedRequest(BaseModel):
    interval: float

# Router for Dual-Path Mounting (/api/* and /*)
api_router = APIRouter()

# Startup Event
@app.on_event("startup")
def startup_event():
    global USE_REAL_MODEL
    logger.info("Initializing database schema and migrating legacy records...")
    db.init_db()
    stream_pipeline.load_dataset()
    
    eager_load = os.getenv("USE_REAL_MODEL_ON_BOOT", "false").lower() == "true"
    if eager_load and explain.ML_AVAILABLE:
        logger.info("Eager loading HuggingFace BERT model...")
        success = explain.init_ml_model()
        USE_REAL_MODEL = success
    else:
        logger.info("Running with lazy-loading model configuration.")

# ---------------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# ---------------------------------------------------------------------

@api_router.post("/auth/login")
def login(payload: LoginRequest, request: Request, session: Session = Depends(db.get_db)):
    """Authenticates user, checks rate limits, and issues JWT token."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    auth.check_rate_limit(client_ip)
    
    user = session.query(db.User).filter(db.User.email == payload.email).first()
    if not user or not auth.verify_password(payload.password, user.hashed_password):
        auth.record_failed_attempt(client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password."
        )
        
    auth.reset_rate_limit(client_ip)
    access_token = auth.create_access_token(
        data={"sub": user.email, "role": user.role, "user_id": user.id}
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user.to_dict()
    }

@api_router.get("/auth/me")
def get_current_user_profile(current_user: dict = Depends(auth.get_current_user)):
    """Returns the profile and role of the currently authenticated token owner."""
    return current_user

@api_router.post("/auth/register")
def register_user(payload: RegisterRequest, session: Session = Depends(db.get_db)):
    """Registers a new moderator or admin account."""
    existing = session.query(db.User).filter(db.User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
        
    new_user = db.User(
        email=payload.email,
        hashed_password=auth.get_password_hash(payload.password),
        role=payload.role if payload.role in ["admin", "moderator"] else "moderator"
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return {"status": "success", "user": new_user.to_dict()}

# ---------------------------------------------------------------------
# STREAMING MANAGEMENT ENDPOINTS
# ---------------------------------------------------------------------

@api_router.post("/stream/start")
def start_stream(payload: Optional[StreamSpeedRequest] = None):
    """Starts backend streaming ingestion worker."""
    interval = payload.interval if payload else 3.0
    stream_pipeline.start(interval)
    return {"status": "streaming_started", "interval": stream_pipeline.interval}

@api_router.post("/stream/stop")
def stop_stream():
    """Halts backend streaming ingestion worker."""
    stream_pipeline.stop()
    return {"status": "streaming_stopped"}

@api_router.get("/stream/status")
def get_stream_status():
    """Returns streaming state and active WebSocket connection count."""
    return {
        "is_streaming": stream_pipeline.is_streaming,
        "interval": stream_pipeline.interval,
        "active_ws_connections": len(ws_manager.active_connections)
    }

# ---------------------------------------------------------------------
# HUMAN-IN-THE-LOOP REVIEW QUEUE (MODERATOR / ADMIN)
# ---------------------------------------------------------------------

@api_router.get("/queue")
def get_review_queue(
    status: str = "pending", 
    session: Session = Depends(db.get_db)
):
    """Fetches escalated items in the uncertain band [40%, 70%]. Accessible by moderators and admins."""
    return db.get_review_queue_items(session, status=status)

@api_router.post("/queue/{prediction_id}/resolve")
def resolve_queue_item(
    prediction_id: str, 
    payload: ActionRequest, 
    session: Session = Depends(db.get_db),
    current_user: dict = Depends(auth.require_role(["admin", "moderator"]))
):
    """Submits a human moderator decision (Approved / Hidden) for an escalated case."""
    valid_actions = ["Approved", "Hidden", "Review"]
    if payload.action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Choose from {valid_actions}")

    success = db.resolve_review_item(
        prediction_id=prediction_id,
        action=payload.action,
        notes=payload.notes,
        moderator_id=current_user["id"],
        session=session
    )
    if not success:
        raise HTTPException(status_code=404, detail="Prediction item not found in queue.")

    return {
        "status": "success",
        "prediction_id": prediction_id,
        "action": payload.action,
        "moderator": current_user["email"]
    }

# ---------------------------------------------------------------------
# FAIRNESS & BIAS AUDIT ENDPOINT
# ---------------------------------------------------------------------

@api_router.get("/fairness")
def get_fairness_audit_metrics():
    """Returns demographic & dialect parity audit metrics (AAVE vs SAE proxy FPR gap)."""
    try:
        return fairness.run_fairness_audit()
    except Exception as e:
        logger.error(f"Error running fairness audit: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------
# CORE PREDICTION & SIMULATOR FALLBACK ENDPOINTS
# ---------------------------------------------------------------------

@api_router.post("/predict")
def predict_post(payload: PredictionRequest, session: Session = Depends(db.get_db)):
    """Classifies custom input text via the Learned Context-Aware Inference Pipeline."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty or whitespace only.")

    try:
        from inference.pipeline import pipeline

        resolved_target = resolve_target(
            payload.text,
            parent_username=payload.target_user,
            author_username=payload.username
        )

        res = pipeline.process(
            text=payload.text,
            author=payload.username,
            target_user=resolved_target,
            use_real_model=USE_REAL_MODEL,
            skip_lime=False,
            persist=True,
            session=session
        )

        return {
            **res,
            "user_id": payload.user_id,
            "username": payload.username,
            # Backward-compatible fields
            "delta_neg": 0.0,
            "delta_slang": 0.0,
            "delta_sarcasm": 0.0,
            "delta_target": 0.0,
            "total_context_delta": round(res["final_score"] - res["p_toxic"], 4),
            "formula": f"MLP_Fusion(P_tox={res['p_toxic']:.3f}, P_ctx={res['p_context']:.3f}, V_1h={res['context_features']['v_1h']}) = {res['final_score']:.3f}"
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in predict_post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/simulator/next")
async def simulator_next(session: Session = Depends(db.get_db)):
    """Pulls the next tweet from the simulation dataset (HTTP fallback for WebSocket)."""
    if stream_pipeline.tweets_df is None or len(stream_pipeline.tweets_df) == 0:
        stream_pipeline.load_dataset()
        
    df = stream_pipeline.tweets_df
    if df is None or len(df) == 0:
        raise HTTPException(status_code=500, detail="Simulation dataset tweets.csv is empty or missing.")

    row = df.iloc[stream_pipeline.current_index % len(df)]
    stream_pipeline.current_index += 1

    item = {
        "text": str(row["text"]),
        "author": str(row["username"]) if "username" in row else "@anonymous",
        "user_id": str(row["user_id"]) if "user_id" in row else "user_sim",
        "parent_username": str(row["parent_username"]) if "parent_username" in row and not pd.isna(row["parent_username"]) else None
    }

    event = await stream_pipeline.process_event(item, use_real_model=USE_REAL_MODEL)
    await ws_manager.broadcast(event)
    return event

@api_router.get("/posts")
def get_posts(session: Session = Depends(db.get_db)):
    """Fetches all classified posts logged in the system."""
    return db.get_all_posts(session)

@api_router.post("/posts/{post_id}/action")
def update_action(post_id: str, payload: ActionRequest, session: Session = Depends(db.get_db)):
    """Updates moderator action for a specific post."""
    valid_actions = ["Approved", "Hidden", "Review"]
    if payload.action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Choose from {valid_actions}")
        
    success = db.update_moderator_action(post_id, payload.action)
    if not success:
        raise HTTPException(status_code=404, detail="Post not found or failed to update.")
        
    return {"status": "success", "post_id": post_id, "action": payload.action}

@api_router.get("/graph")
def get_graph_data(session: Session = Depends(db.get_db)):
    """Generates social network directed graph analysis from database logs."""
    try:
        all_logs = db.get_all_posts(session)
        analysis = graph.analyze_graph(all_logs)
        return analysis
    except Exception as e:
        logger.error(f"Error compiling graph data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/analytics")
def get_dashboard_analytics(session: Session = Depends(db.get_db)):
    """Compiles dashboard analytics including daily trends, counts, and moderator accuracy."""
    analytics = db.get_analytics(session)
    graph_data = get_graph_data(session)
    analytics["top_abusers"] = graph_data.get("top_abusers", [])
    return analytics

@api_router.post("/toggle-mode")
def toggle_model_mode():
    """Toggles AI backend mode between Keyword Fallback and Real BERT."""
    global USE_REAL_MODEL
    if not explain.ML_AVAILABLE:
        return {
            "status": "error",
            "message": "PyTorch/Transformers missing. BERT unavailable.",
            "use_real_model": False
        }
        
    USE_REAL_MODEL = not USE_REAL_MODEL
    if USE_REAL_MODEL and explain._nlp_pipeline is None:
        explain.init_ml_model()
        
    return {
        "status": "success",
        "use_real_model": USE_REAL_MODEL,
        "message": f"Switched to {'Toxic-BERT Transformer' if USE_REAL_MODEL else 'Keyword Fallback Mode'}"
    }

@api_router.get("/feedback/dataset")
def export_feedback_dataset(session: Session = Depends(db.get_db)):
    """Exports human moderator decisions and contextual features for active learning retraining loops."""
    feedback = db.get_feedback_dataset(session)
    return {
        "total_records": len(feedback),
        "dataset": feedback
    }

@api_router.get("/status")
def get_system_status():
    """Returns model status and hardware details."""
    return {
        "ml_packages_installed": explain.ML_AVAILABLE,
        "use_real_model": USE_REAL_MODEL,
        "model_loaded": explain._nlp_pipeline is not None,
        "active_model": "unitary/toxic-bert" if USE_REAL_MODEL else "Keyword Fallback Lexicon",
        "context_model": "DistilBERT Multi-Task (context-distilbert-v1)",
        "fusion_engine": "Learned Context-Aware Fusion MLP (fusion-mlp-v1)",
        "context_layer": "Multi-Source Context Collector (Conversation, Behavioral, Temporal, Graph)"
    }

@api_router.get("/evaluation")
def get_model_evaluation():
    """Runs model evaluation across the 4 Tiers, CheckList, and Fairness audits."""
    try:
        return evaluate.run_evaluation(include_checklist=True, include_fairness=True)
    except Exception as e:
        logger.error(f"Error running evaluation API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Mount Router on both '/' and '/api' for complete proxy and direct compatibility
app.include_router(api_router)
app.include_router(api_router, prefix="/api")

# ---------------------------------------------------------------------
# WEBSOCKET ENDPOINT (Origin validated)
# ---------------------------------------------------------------------

@app.websocket("/ws/feed")
@app.websocket("/ws")
async def websocket_feed_endpoint(websocket: WebSocket):
    """
    Real-time streaming WebSocket endpoint.
    Performs handshake origin validation and pushes classified events to subscribers.
    """
    connected = await ws_manager.connect(websocket)
    if not connected:
        return
        
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            if action == "start":
                interval = float(data.get("interval", 3.0))
                stream_pipeline.start(interval)
                await websocket.send_json({"type": "stream_status", "is_streaming": True, "interval": stream_pipeline.interval})
            elif action == "stop":
                stream_pipeline.stop()
                await websocket.send_json({"type": "stream_status", "is_streaming": False})
            elif action == "speed":
                stream_pipeline.set_speed(float(data.get("interval", 3.0)))
                await websocket.send_json({"type": "stream_speed", "interval": stream_pipeline.interval})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket session error: {e}")
        ws_manager.disconnect(websocket)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "CyberShield Context-Aware Cyberbullying Detection API",
        "version": "2.0.0",
        "ml_dependencies": explain.ML_AVAILABLE,
        "use_real_model": USE_REAL_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server on localhost:8000...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
