import os
import uuid
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import db
import explain
import graph

# Configure standard logging format and levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Context-Aware Cyberbullying Detection System API",
    description="Backend API for B.Tech final-year project including BERT, LIME, NetworkX, and SQLite logs.",
    version="1.0.0"
)

# CORS configuration
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
logger.info(f"Allowed CORS origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
USE_REAL_MODEL = False
TWEETS_INDEX = 0

# Database Initialization and Conditionally Eager Model Preloading on Startup
@app.on_event("startup")
def startup_event():
    global USE_REAL_MODEL
    db.init_db()
    
    # Conditionally eager load model if USE_REAL_MODEL_ON_BOOT=true
    eager_load = os.getenv("USE_REAL_MODEL_ON_BOOT", "false").lower() == "true"
    if eager_load:
        if explain.ML_AVAILABLE:
            logger.info("Eager loading ML model on boot (USE_REAL_MODEL_ON_BOOT=true)...")
            success = explain.init_ml_model()
            if success:
                USE_REAL_MODEL = True
            else:
                logger.warning("Eager loading failed. Falling back to Fast Demo Mode.")
        else:
            logger.warning("ML packages missing. Cannot eager load. Running in Fast Demo Mode.")
    else:
        logger.info("ML model will be lazy-loaded when model mode is toggled.")

# Pydantic Schemas
class PredictionRequest(BaseModel):
    text: str
    user_id: str
    username: str = "@anonymous"
    target_user: str = "@anonymous"

class ActionRequest(BaseModel):
    action: str

# Endpoints
@app.get("/")
def read_root():
    return {
        "status": "online",
        "ml_dependencies": explain.ML_AVAILABLE,
        "use_real_model": USE_REAL_MODEL
    }

@app.post("/predict")
def predict_post(payload: PredictionRequest):
    """
    Classifies a text block for cyberbullying, generates LIME weights, 
    saves details to the SQLite logs, and returns the metadata.
    """
    # Pre-validate whitespace/empty text to return 400 before calling model
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty or whitespace only.")
        
    try:
        # Run dual-mode explainer (returns dictionary)
        result = explain.get_lime_explanation(
            payload.text, 
            use_real_model=USE_REAL_MODEL
        )
        
        post_id = str(uuid.uuid4())[:8]  # short readable ID
        timestamp = datetime.now().isoformat()
        
        # Log to SQLite
        log_data = {
            "id": post_id,
            "user_id": payload.username,  # map user handle to user_id column
            "target_user": payload.target_user,
            "text": payload.text,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "mode": result["mode"],
            "timestamp": timestamp
        }
        
        success = db.log_post(log_data)
        if not success:
            logger.error("Failed to log post to SQLite database.")
            raise HTTPException(status_code=500, detail="Failed to persist post to the database.")
        
        return {
            "id": post_id,
            "user_id": payload.user_id,
            "username": payload.username,
            "target_user": payload.target_user,
            "text": payload.text,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "explanations": result["explanations"],
            "highlighted_words": result["highlighted"],
            "lime_weights": result["lime_weights"],
            "moderator_action": "Pending",
            "mode": result["mode"],
            "timestamp": timestamp
        }
    except ValueError as ve:
        logger.warning(f"Validation error in get_lime_explanation: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in predict_post endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulator/next")
def simulator_next():
    """
    Simulates a live feed by pulling the next tweet from tweets.csv,
    classifying it, logging it to the DB, and returning it.
    """
    global TWEETS_INDEX
    csv_path = os.path.join(os.path.dirname(__file__), "tweets.csv")
    
    if not os.path.exists(csv_path):
        logger.error(f"Tweets simulation dataset not found at {csv_path}")
        raise HTTPException(status_code=404, detail="tweets.csv simulation dataset not found.")
        
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            logger.error("Simulation dataset tweets.csv is empty.")
            raise HTTPException(status_code=500, detail="tweets.csv is empty")
            
        # Wrap index around
        row = df.iloc[TWEETS_INDEX % len(df)]
        TWEETS_INDEX += 1
        
        # Get details
        text = str(row['text'])
        user_id = str(row['user_id'])
        username = str(row['username'])
        target_user = str(row['target_user']) if 'target_user' in row and not pd.isna(row['target_user']) else "@anonymous"
        
        # Run classification
        result = explain.get_lime_explanation(
            text, 
            use_real_model=USE_REAL_MODEL
        )
        
        post_id = f"sim-{TWEETS_INDEX}-{str(uuid.uuid4())[:4]}"
        
        # Create a spread out timeline to make charts look beautiful.
        # Shift timestamps backwards in time slightly for earlier logs to create daily trends.
        days_back = (TWEETS_INDEX % 5) # Distribute logs over last 5 days
        log_time = datetime.now() - timedelta(days=days_back)
        timestamp = log_time.isoformat()
        
        # Log to SQLite
        log_data = {
            "id": post_id,
            "user_id": username,
            "target_user": target_user,
            "text": text,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "mode": result["mode"],
            "timestamp": timestamp
        }
        
        success = db.log_post(log_data)
        if not success:
            logger.error("Failed to log simulated post to database.")
            raise HTTPException(status_code=500, detail="Failed to persist simulated post to the database.")
        
        return {
            "id": post_id,
            "user_id": user_id,
            "username": username,
            "target_user": target_user,
            "text": text,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "explanations": result["explanations"],
            "highlighted_words": result["highlighted"],
            "lime_weights": result["lime_weights"],
            "moderator_action": "Pending",
            "mode": result["mode"],
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"Error in simulator_next endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/posts")
def get_posts():
    """Fetches all classified posts logged in the system."""
    return db.get_all_posts()

@app.post("/posts/{post_id}/action")
def update_action(post_id: str, payload: ActionRequest):
    """Updates the moderator's action for a specific post (Approve, Hide, Review)."""
    valid_actions = ["Approved", "Hidden", "Review"]
    if payload.action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Choose from {valid_actions}")
        
    success = db.update_moderator_action(post_id, payload.action)
    if not success:
        logger.warning(f"Moderator action update failed: Post ID {post_id} not found.")
        raise HTTPException(status_code=404, detail="Post not found or failed to update.")
        
    return {"status": "success", "post_id": post_id, "action": payload.action}

@app.get("/graph")
def get_graph_data():
    """
    Reads interactions.json and generates social network centrality and group cluster metrics.
    """
    json_path = os.path.join(os.path.dirname(__file__), "interactions.json")
    if not os.path.exists(json_path):
        logger.warning(f"interactions.json not found at {json_path}")
        return {"nodes": [], "edges": [], "top_abusers": [], "clusters": []}
        
    try:
        with open(json_path, "r") as f:
            interactions = json.load(f)
        
        # Dynamically add interactions from database logs (excluding anonymous or self-directed posts)
        all_logs = db.get_all_posts()
        for log in all_logs:
            if log["prediction"] == "Cyberbullying":
                attacker = log["user_id"]
                target = log.get("target_user")
                if attacker and target and attacker != "@anonymous" and target != "@anonymous" and attacker != target:
                    if [attacker, target] not in interactions:
                        interactions.append([attacker, target])
                    
        analysis = graph.analyze_graph(interactions)
        return analysis
    except Exception as e:
        logger.error(f"Error compiling graph data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_dashboard_analytics():
    """Compiles dashboard analytics including daily trends, counts, and moderator accuracy."""
    analytics = db.get_analytics()
    
    # Inject top abusers from NetworkX graph
    graph_data = get_graph_data()
    analytics["top_abusers"] = graph_data.get("top_abusers", [])
    return analytics

@app.post("/toggle-mode")
def toggle_model_mode():
    """Toggles the AI pipeline between Fast Demo and Real BERT modes."""
    global USE_REAL_MODEL
    
    if not explain.ML_AVAILABLE:
        logger.warning("Attempted to switch to BERT, but ML packages are missing.")
        return {
            "status": "error",
            "message": "HuggingFace/PyTorch libraries are not installed locally. Cannot enable BERT.",
            "use_real_model": False
        }
        
    # Toggle model mode
    USE_REAL_MODEL = not USE_REAL_MODEL
    
    # Lazy load model if enabled
    if USE_REAL_MODEL and explain._nlp_pipeline is None:
        logger.info("Lazy loading HuggingFace model on toggle mode...")
        initialized = explain.init_ml_model()
        if not initialized:
            USE_REAL_MODEL = False
            return {
                "status": "error",
                "message": "Error loading transformers model weights. Fallback to Demo Mode.",
                "use_real_model": False
            }
            
    return {
        "status": "success",
        "use_real_model": USE_REAL_MODEL,
        "message": f"Switched system to {'Real BERT Classifier' if USE_REAL_MODEL else 'Fast Demo Mode'}"
    }

@app.get("/status")
def get_system_status():
    """Returns the current model status and hardware settings."""
    return {
        "ml_packages_installed": explain.ML_AVAILABLE,
        "use_real_model": USE_REAL_MODEL,
        "model_loaded": explain._nlp_pipeline is not None,
        "active_model": "unitary/toxic-bert" if USE_REAL_MODEL else "Fast Demo Classifier"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server on localhost:8000...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
