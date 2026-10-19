"""
Database Management Layer for Cyberbullying Detection & Moderation System.

SCHEMA ARCHITECTURE:
This module manages a 5-table relational schema using SQLAlchemy:
1. users: Authenticated accounts with role-based access (admin, moderator).
2. predictions: Persistent log of text classifications, raw toxicity scores, context deltas, and fused scores.
3. review_queue: Escalated predictions in the uncertain band (0.40 <= S_fused <= 0.70) awaiting human review.
4. moderator_actions: Audit trail of human decisions (Approved, Hidden, Review).
5. fairness_metrics: Demographic & dialect proxy metrics (Blodgett et al. AAVE proxy & Borkan et al. identity terms).

MIGRATION STRATEGY:
When upgrading from legacy versions containing the flat 'moderation_logs' table:
- migrate_legacy_data() runs automatically on startup during init_db().
- Rows from 'moderation_logs' are mapped into the 'predictions' table with sensible defaults.
- The old table is archived as 'moderation_logs_archived'.
- Standard demo accounts (admin@cybershield.org, mod@cybershield.org) and initial seeds are populated.

DUAL-ENGINE SUPPORT:
Supports PostgreSQL via DATABASE_URL environment variable with automatic fallback to local SQLite.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Generator, List, Dict, Any, Optional
import sqlite3

from sqlalchemy import create_engine, func, text, inspect
from sqlalchemy.orm import sessionmaker, Session

from models import Base, User, Prediction, ReviewQueue, ModeratorAction, FairnessMetric
import auth

logger = logging.getLogger(__name__)

# Database URL configuration (PostgreSQL if provided, otherwise SQLite)
DEFAULT_SQLITE_PATH = os.path.join(os.path.dirname(__file__), "moderation.db")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Use SQLite
    db_path = os.getenv("DATABASE_PATH", DEFAULT_SQLITE_PATH)
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    DATABASE_URL = f"sqlite:///{db_path}"

# SQLite specific connect_args
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def migrate_legacy_data():
    """
    Checks for the presence of the legacy 'moderation_logs' flat table.
    Migrates existing records into the new 'predictions' table schema and archives the old table.
    """
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    if "moderation_logs" in tables:
        logger.info("Legacy 'moderation_logs' table detected. Migrating to new 5-table schema...")
        session = SessionLocal()
        try:
            with engine.connect() as conn:
                legacy_rows = conn.execute(text("SELECT * FROM moderation_logs")).fetchall()
                
                count = 0
                for row in legacy_rows:
                    row_dict = dict(row._mapping)
                    row_id = str(row_dict.get("id"))
                    
                    # Check if already migrated
                    existing = session.query(Prediction).filter(Prediction.id == row_id).first()
                    if existing:
                        continue
                        
                    raw_conf = float(row_dict.get("confidence") or 80.0) / 100.0
                    pred_label = str(row_dict.get("prediction") or "Safe")
                    
                    # Parse timestamp
                    ts_str = row_dict.get("timestamp")
                    try:
                        ts = datetime.fromisoformat(ts_str) if ts_str else datetime.utcnow()
                    except Exception:
                        ts = datetime.utcnow()
                        
                    new_pred = Prediction(
                        id=row_id,
                        text=row_dict.get("post_text") or "",
                        author=row_dict.get("user_id") or "@anonymous",
                        target_user=row_dict.get("target_user") or "@anonymous",
                        toxicity_score=raw_conf,
                        context_delta=0.0,
                        final_score=raw_conf,
                        label=pred_label,
                        status="AutoProcessed" if pred_label != "Uncertain" else "EscalatedToHuman",
                        moderator_action=row_dict.get("moderator_action") or "Pending",
                        mode=row_dict.get("mode") or "mock",
                        timestamp=ts
                    )
                    session.add(new_pred)
                    count += 1
                    
                session.commit()
                logger.info(f"Successfully migrated {count} legacy records to 'predictions'.")
                
                # Archive legacy table so it doesn't run again
                conn.execute(text("ALTER TABLE moderation_logs RENAME TO moderation_logs_archived"))
                conn.commit()
                logger.info("Renamed 'moderation_logs' to 'moderation_logs_archived'.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error migrating legacy moderation_logs: {e}", exc_info=True)
        finally:
            session.close()

def seed_demo_users_and_records(session: Session):
    """Seeds demo accounts and sample interactions if tables are empty."""
    # 1. Seed demo accounts (DEMO USE ONLY)
    if session.query(User).count() == 0:
        logger.info("Seeding demo accounts: admin@cybershield.org and mod@cybershield.org...")
        admin = User(
            email="admin@cybershield.org",
            hashed_password=auth.get_password_hash("AdminPass123!"),
            role="admin"
        )
        mod = User(
            email="mod@cybershield.org",
            hashed_password=auth.get_password_hash("ModPass123!"),
            role="moderator"
        )
        session.add(admin)
        session.add(mod)
        session.commit()

    # 2. Seed initial predictions if empty
    if session.query(Prediction).count() == 0:
        logger.info("Database predictions empty. Seeding initial baseline data...")
        now = datetime.utcnow()
        
        # Initial seed items: attacker, target, text, toxicity, delta, final, label, action, delta_time
        seeds = [
            ("@toxic_dan", "@kind_katie", "You are absolute trash, literally the worst person on this site.", 0.95, 0.0, 0.95, "Cyberbullying", "Hidden", timedelta(days=5)),
            ("@toxic_dan", "@kind_katie", "Stupid idiot, stop posting your garbage here, nobody cares.", 0.92, 0.0, 0.92, "Cyberbullying", "Hidden", timedelta(days=4)),
            ("@toxic_dan", "@kind_katie", "You are a pathetic excuse for a human being. Everyone hates you.", 0.88, 0.0, 0.88, "Cyberbullying", "Hidden", timedelta(days=3)),
            ("@toxic_dan", "@kind_katie", "Shut up you brainless troll. You are nothing but a disappointment.", 0.93, 0.0, 0.93, "Cyberbullying", "Pending", timedelta(days=1)),
            
            ("@hater_harry", "@happy_helen", "Nobody likes you. You are a complete loser and will never succeed.", 0.94, 0.0, 0.94, "Cyberbullying", "Hidden", timedelta(days=4)),
            ("@hater_harry", "@happy_helen", "You are a complete disappointment to everyone around you.", 0.89, 0.0, 0.89, "Cyberbullying", "Hidden", timedelta(days=2)),
            ("@hater_harry", "@happy_helen", "Everything you write is trash, just like you.", 0.93, 0.0, 0.93, "Cyberbullying", "Pending", timedelta(hours=12)),
            
            ("@bullier_bob", "@good_vibes", "Wow you look incredibly ugly in that picture, go do something about your face.", 0.96, 0.0, 0.96, "Cyberbullying", "Hidden", timedelta(days=5)),
            ("@bullier_bob", "@good_vibes", "You should be ashamed of yourself. You are completely worthless.", 0.90, 0.0, 0.90, "Cyberbullying", "Pending", timedelta(days=3)),
            
            # Negations & Slang resolved to Safe
            ("@alex_dev", "@kind_katie", "Don't worry, you are not stupid. You just need more practice.", 0.82, -0.55, 0.27, "Safe", "Approved", timedelta(days=3)),
            ("@happy_helen", "@good_vibes", "You are not a loser, you did your absolute best in the game.", 0.85, -0.55, 0.30, "Safe", "Approved", timedelta(days=2)),
            ("@kind_katie", "@good_vibes", "Oh my god, you are killing it on stage! Incredible performance!", 0.75, -0.50, 0.25, "Safe", "Approved", timedelta(hours=4)),
            
            # Sarcasm resolved to Cyberbullying
            ("@toxic_dan", "@alex_dev", "You are so smart and brilliant, said no one ever.", 0.25, +0.50, 0.75, "Cyberbullying", "Pending", timedelta(days=1)),
            
            # Safe positive posts
            ("@kind_katie", "@anonymous", "This is such a lovely picture! Thanks for sharing this view.", 0.05, 0.0, 0.05, "Safe", "Approved", timedelta(days=6)),
            ("@alex_dev", "@anonymous", "I completely agree with your analysis of the tech trends today.", 0.02, 0.0, 0.02, "Safe", "Approved", timedelta(days=5)),
            ("@happy_helen", "@anonymous", "Had a great time with the family at the park this weekend. Truly blessed.", 0.01, 0.0, 0.01, "Safe", "Approved", timedelta(days=4)),
            
            # Uncertain / Escalated cases for review queue demonstration
            ("@troll_tim", "@alex_dev", "I don't think your presentation was completely garbage, but it was borderline.", 0.78, -0.20, 0.58, "Uncertain", "Pending", timedelta(hours=6)),
            ("@mean_mel", "@good_vibes", "You think you're killing it but someone needs to humble you.", 0.60, -0.05, 0.55, "Uncertain", "Pending", timedelta(hours=2))
        ]
        
        for author, target, text_content, p_tox, delta, s_fuse, label, action, td in seeds:
            pred_id = f"seed-{str(abs(hash(text_content)))[:8]}"
            ts = now - td
            is_uncertain = (label == "Uncertain")
            
            pred = Prediction(
                id=pred_id,
                text=text_content,
                author=author,
                target_user=target,
                toxicity_score=p_tox,
                context_delta=delta,
                final_score=s_fuse,
                label=label,
                status="EscalatedToHuman" if is_uncertain else "AutoProcessed",
                moderator_action=action,
                mode="mock",
                timestamp=ts
            )
            session.add(pred)
            
            # If uncertain, auto-add to ReviewQueue
            if is_uncertain:
                rq = ReviewQueue(
                    prediction_id=pred_id,
                    escalation_reason=f"Confidence score {s_fuse*100:.1f}% falls within uncertain band [40%, 70%]. Requires human moderator judgment.",
                    status="pending",
                    escalated_at=ts
                )
                session.add(rq)
                
        session.commit()
        logger.info("Successfully seeded baseline predictions and review queue items.")

def upgrade_predictions_table_schema():
    """
    Checks the predictions table columns.
    If new contextual columns (p_toxic, p_context, etc.) are missing,
    automatically executes ALTER TABLE ADD COLUMN.
    """
    inspector = inspect(engine)
    if "predictions" not in inspector.get_table_names():
        return

    existing_cols = {col["name"] for col in inspector.get_columns("predictions")}
    needed_cols = [
        ("p_toxic", "FLOAT"),
        ("p_context", "FLOAT"),
        ("targeting_score", "FLOAT DEFAULT 0.0"),
        ("repetition_score", "FLOAT DEFAULT 0.0"),
        ("escalation_score", "FLOAT DEFAULT 0.0"),
        ("graph_score", "FLOAT DEFAULT 0.0"),
        ("model_version", "VARCHAR(50) DEFAULT 'toxic-bert-v1'"),
        ("context_model_version", "VARCHAR(50) DEFAULT 'context-distilbert-v1'"),
        ("fusion_model_version", "VARCHAR(50) DEFAULT 'fusion-mlp-v1'"),
        ("context_features_json", "TEXT"),
        ("explanation_json", "TEXT"),
        ("thread_id", "VARCHAR(100)"),
        ("parent_id", "VARCHAR(100)")
    ]

    with engine.connect() as conn:
        for col_name, col_type in needed_cols:
            if col_name not in existing_cols:
                try:
                    conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                    logger.info(f"Added column {col_name} to predictions table.")
                except Exception as e:
                    logger.warning(f"Could not add column {col_name}: {e}")

def init_db():
    """Initializes the database schema, runs legacy migrations, and seeds initial records."""
    logger.info(f"Initializing database with connection: {DATABASE_URL[:25]}...")
    Base.metadata.create_all(bind=engine)
    migrate_legacy_data()
    upgrade_predictions_table_schema()
    
    session = SessionLocal()
    try:
        seed_demo_users_and_records(session)
    finally:
        session.close()

def log_prediction(data: Dict[str, Any], session: Optional[Session] = None) -> Prediction:
    """
    Logs a new prediction to the database with full contextual intelligence.
    If the prediction is in the uncertain band (escalate_to_human is True),
    automatically enqueues an item in ReviewQueue.
    """
    should_close = False
    if session is None:
        session = SessionLocal()
        should_close = True

    try:
        post_id = data["id"]
        pred = Prediction(
            id=post_id,
            text=data["text"],
            author=data.get("user_id") or data.get("author") or "@anonymous",
            target_user=data.get("target_user") or "@anonymous",
            toxicity_score=float(data.get("toxicity_score") or data.get("p_toxic") or 0.0),
            context_delta=float(data.get("context_delta") or data.get("total_context_delta") or 0.0),
            p_toxic=float(data.get("p_toxic") or data.get("toxicity_score") or 0.0),
            p_context=float(data.get("p_context")) if data.get("p_context") is not None else None,
            final_score=float(data.get("final_score") or data.get("s_fused") or 0.0),
            targeting_score=float(data.get("targeting_score") or 0.0),
            repetition_score=float(data.get("repetition_score") or 0.0),
            escalation_score=float(data.get("escalation_score") or 0.0),
            graph_score=float(data.get("graph_score") or 0.0),
            label=data.get("prediction") or data.get("label") or "Safe",
            status=data.get("status", "AutoProcessed"),
            moderator_action=data.get("moderator_action", "Pending"),
            mode=data.get("mode", "mock"),
            model_version=data.get("model_version", "toxic-bert-v1"),
            context_model_version=data.get("context_model_version", "context-distilbert-v1"),
            fusion_model_version=data.get("fusion_model_version", "fusion-mlp-v1"),
            thread_id=data.get("thread_id"),
            parent_id=data.get("parent_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data and isinstance(data["timestamp"], str) else datetime.utcnow()
        )
        session.add(pred)

        # Enqueue to review_queue if uncertain
        if data.get("escalate_to_human") or pred.label == "Uncertain":
            rq = ReviewQueue(
                prediction_id=post_id,
                escalation_reason=data.get("escalation_reason") or "Score in uncertain decision band [40%, 70%].",
                status="pending"
            )
            session.add(rq)

        session.commit()
        session.refresh(pred)
        return pred
    except Exception as e:
        session.rollback()
        logger.error(f"Error logging prediction: {e}", exc_info=True)
        raise e
    finally:
        if should_close:
            session.close()

def get_feedback_dataset(session: Session) -> List[Dict[str, Any]]:
    """
    Exports resolved moderator decisions for active learning and model retraining loops.
    """
    resolved_actions = session.query(ModeratorAction).all()
    feedback_rows = []

    for action in resolved_actions:
        pred = action.prediction
        if not pred:
            continue

        ground_truth = "Cyberbullying" if action.action == "Hidden" else ("Safe" if action.action == "Approved" else None)
        if not ground_truth:
            continue

        feedback_rows.append({
            "prediction_id": pred.id,
            "text": pred.text,
            "author": pred.author,
            "target_user": pred.target_user,
            "model_prediction": pred.label,
            "p_toxic": pred.p_toxic or pred.toxicity_score,
            "p_context": pred.p_context,
            "final_score": pred.final_score,
            "moderator_action": action.action,
            "ground_truth_label": ground_truth,
            "moderator_notes": action.notes,
            "model_version": pred.model_version,
            "context_model_version": pred.context_model_version,
            "fusion_model_version": pred.fusion_model_version,
            "resolved_at": action.timestamp.isoformat() if action.timestamp else None
        })

    return feedback_rows

def get_all_posts(session: Session) -> List[Dict[str, Any]]:
    """Fetches all predictions sorted by timestamp descending."""
    preds = session.query(Prediction).order_by(Prediction.timestamp.desc()).all()
    return [p.to_dict() for p in preds]

def get_review_queue_items(session: Session, status: str = "pending") -> List[Dict[str, Any]]:
    """Fetches review queue items with associated prediction data."""
    query = session.query(ReviewQueue).filter(ReviewQueue.status == status).order_by(ReviewQueue.escalated_at.desc())
    return [rq.to_dict() for rq in query.all()]

def resolve_review_item(
    prediction_id: str, 
    action: str, 
    notes: Optional[str] = None, 
    moderator_id: Optional[str] = None, 
    session: Optional[Session] = None
) -> bool:
    """
    Resolves an escalated review item:
    - Sets ReviewQueue status to 'resolved'.
    - Updates Prediction moderator_action and status.
    - Records an audit row in ModeratorAction.
    """
    should_close = False
    if session is None:
        session = SessionLocal()
        should_close = True

    try:
        pred = session.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not pred:
            return False

        pred.moderator_action = action
        pred.status = "Resolved"

        rq = session.query(ReviewQueue).filter(ReviewQueue.prediction_id == prediction_id).first()
        if rq:
            rq.status = "resolved"

        mod_action = ModeratorAction(
            review_id=rq.id if rq else None,
            prediction_id=pred.id,
            moderator_id=moderator_id,
            action=action,
            notes=notes
        )
        session.add(mod_action)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error resolving review item {prediction_id}: {e}", exc_info=True)
        return False
    finally:
        if should_close:
            session.close()

def update_moderator_action(post_id: str, action: str, moderator_id: Optional[str] = None) -> bool:
    """Quick update for moderator actions from dashboard or review queue."""
    session = SessionLocal()
    try:
        return resolve_review_item(post_id, action, moderator_id=moderator_id, session=session)
    finally:
        session.close()

def get_analytics(session: Session) -> Dict[str, Any]:
    """
    Computes dashboard analytics from predictions and moderator actions:
    - Action counts (Total, Approved, Hidden, Pending)
    - Daily trends (Cyberbullying vs Safe)
    - Human-in-the-loop Moderator Agreement Rate
    """
    total = session.query(Prediction).count()
    approved = session.query(Prediction).filter(Prediction.moderator_action == "Approved").count()
    hidden = session.query(Prediction).filter(Prediction.moderator_action == "Hidden").count()
    pending = session.query(Prediction).filter(Prediction.moderator_action == "Pending").count()
    uncertain_pending = session.query(ReviewQueue).filter(ReviewQueue.status == "pending").count()

    # Daily trends (recent 14 days)
    # Using SQL date function compatible with SQLite and PostgreSQL
    date_col = func.substr(Prediction.timestamp, 1, 10) if DATABASE_URL.startswith("sqlite") else func.to_char(Prediction.timestamp, "YYYY-MM-DD")
    
    daily_query = session.query(
        date_col.label("date"),
        func.sum(case((Prediction.label == "Cyberbullying", 1), else_=0)).label("bullying_count"),
        func.sum(case((Prediction.label == "Safe", 1), else_=0)).label("safe_count")
    ).group_by(date_col).order_by(date_col.desc()).limit(14).all()

    daily_trends = [
        {"date": row.date or "Unknown", "bullying_count": int(row.bullying_count or 0), "safe_count": int(row.safe_count or 0)}
        for row in daily_query
    ]
    daily_trends.reverse()

    # Moderator Agreement Rate:
    # Decided posts: moderator_action in ('Approved', 'Hidden')
    # Agreement: (Label == 'Cyberbullying' and Action == 'Hidden') or (Label == 'Safe' and Action == 'Approved')
    decided = session.query(Prediction).filter(Prediction.moderator_action.in_(["Approved", "Hidden"])).all()
    decided_count = len(decided)
    
    if decided_count > 0:
        agreed = sum(
            1 for p in decided 
            if (p.label == "Cyberbullying" and p.moderator_action == "Hidden") or 
               (p.label == "Safe" and p.moderator_action == "Approved")
        )
        accuracy = round((agreed / decided_count) * 100, 1)
    else:
        accuracy = 100.0

    return {
        "total_posts": total,
        "approved_count": approved,
        "hidden_count": hidden,
        "pending_count": pending,
        "uncertain_pending_count": uncertain_pending,
        "daily_trends": daily_trends,
        "moderator_accuracy": accuracy,
        "moderator_accuracy_sample_size": decided_count
    }

# Safe import of case for SQLAlchemy
from sqlalchemy import case
