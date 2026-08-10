import os
import uuid
import json
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import db
import explain
import graph

# Initialize FastAPI App
app = FastAPI(
    title="Context-Aware Cyberbullying Detection System API",
    description="Backend API for B.Tech final-year project including BERT, LIME, NetworkX, and SQLite logs.",
    version="1.0.0"
)

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
USE_REAL_MODEL = False
TWEETS_INDEX = 0

# Database Initialization on Startup
@app.on_event("startup")
def startup_event():
    db.init_db()
    # Try pre-loading the ML model in the background if dependencies exist
    if explain.ML_AVAILABLE:
        print("[*] ML packages available. Pre-loading model...")
        # We don't block startup, but let's initialize if possible
        explain.init_ml_model()
    else:
        print("[!] ML packages not available. Running in Fast Demo Mode.")

# Pydantic Schemas
class PredictionRequest(BaseModel):
    text: str
    user_id: str
    username: str = "@anonymous"

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
    try:
        # Run dual-mode explainer
        prediction, confidence, explanations, highlighted, lime_weights = explain.get_lime_explanation(
            payload.text, 
            use_real_model=USE_REAL_MODEL
        )
        
        post_id = str(uuid.uuid4())[:8]  # short readable ID
        timestamp = datetime.now().isoformat()
        
        # Log to SQLite
        log_data = {
            "id": post_id,
            "user_id": payload.username,  # map user handle to user_id column
            "text": payload.text,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": timestamp
        }
        db.log_post(log_data)
        
        return {
            "id": post_id,
            "user_id": payload.user_id,
            "username": payload.username,
            "text": payload.text,
            "prediction": prediction,
            "confidence": confidence,
            "explanations": explanations,
            "highlighted_words": highlighted,
            "lime_weights": lime_weights,
            "moderator_action": "Pending",
            "timestamp": timestamp
        }
    except Exception as e:
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
        raise HTTPException(status_code=404, detail="tweets.csv simulation dataset not found.")
        
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            raise HTTPException(status_code=500, detail="tweets.csv is empty")
            
        # Wrap index around
        row = df.iloc[TWEETS_INDEX % len(df)]
        TWEETS_INDEX += 1
        
        # Get details
        text = str(row['text'])
        user_id = str(row['user_id'])
        username = str(row['username'])
        
        # Run classification
        prediction, confidence, explanations, highlighted, lime_weights = explain.get_lime_explanation(
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
            "text": text,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": timestamp
        }
        db.log_post(log_data)
        
        return {
            "id": post_id,
            "user_id": user_id,
            "username": username,
            "text": text,
            "prediction": prediction,
            "confidence": confidence,
            "explanations": explanations,
            "highlighted_words": highlighted,
            "lime_weights": lime_weights,
            "moderator_action": "Pending",
            "timestamp": timestamp
        }
        
    except Exception as e:
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
        
    db.update_moderator_action(post_id, payload.action)
    return {"status": "success", "post_id": post_id, "action": payload.action}

@app.get("/graph")
def get_graph_data():
    """
    Reads interactions.json and generates social network centrality and group cluster metrics.
    """
    json_path = os.path.join(os.path.dirname(__file__), "interactions.json")
    if not os.path.exists(json_path):
        return {"nodes": [], "edges": [], "top_abusers": [], "clusters": []}
        
    try:
        with open(json_path, "r") as f:
            interactions = json.load(f)
        
        # We can also dynamically add some interactions from the database to expand the graph!
        # Fetch actual cyberbullying logs in DB and add to the graph list
        all_logs = db.get_all_posts()
        for log in all_logs:
            if log["prediction"] == "Cyberbullying":
                # Make up a random victim node if username is known, or map it.
                # E.g. log["user_id"] is the attacker. Let's map to standard victim in database context.
                attacker = log["user_id"]
                # Create a victim representation if not present, just to show dynamic integration
                victim = "@moderator_victim"
                if attacker != "@anonymous" and [attacker, victim] not in interactions:
                    # Limit additions so network stays readable
                    interactions.append([attacker, victim])
                    
        analysis = graph.analyze_graph(interactions)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_dashboard_analytics():
    """Compiles dashboard analytics including daily trends, counts, and moderator accuracy."""
    analytics = db.get_analytics()
    
    # We also inject the top abusers from our NetworkX graph for convenience
    graph_data = get_graph_data()
    analytics["top_abusers"] = graph_data.get("top_abusers", [])
    return analytics

@app.post("/toggle-mode")
def toggle_model_mode():
    """Toggles the AI pipeline between Fast Demo and Real BERT modes."""
    global USE_REAL_MODEL
    
    if not explain.ML_AVAILABLE:
        return {
            "status": "error",
            "message": "HuggingFace/PyTorch libraries are not installed locally. Cannot enable BERT.",
            "use_real_model": False
        }
        
    # Toggle model mode
    USE_REAL_MODEL = not USE_REAL_MODEL
    
    # Preload model if we just enabled it
    if USE_REAL_MODEL and explain._nlp_pipeline is None:
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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
