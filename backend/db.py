import sqlite3
import os
import logging
from datetime import datetime

DB_PATH = os.getenv("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "moderation.db"))
logger = logging.getLogger(__name__)

def get_db_connection():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database with the moderation_logs table and handles migrations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moderation_logs (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            target_user TEXT,
            post_text TEXT,
            prediction TEXT,
            confidence REAL,
            moderator_action TEXT DEFAULT 'Pending',
            mode TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()

    # Check if target_user or mode columns are missing (auto-migration)
    cursor.execute("PRAGMA table_info(moderation_logs)")
    columns = [row["name"] for row in cursor.fetchall()]

    if "target_user" not in columns:
        try:
            logger.info("Migrating database: adding 'target_user' column to 'moderation_logs' table.")
            cursor.execute("ALTER TABLE moderation_logs ADD COLUMN target_user TEXT")
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to add target_user column: {e}", exc_info=True)

    if "mode" not in columns:
        try:
            logger.info("Migrating database: adding 'mode' column to 'moderation_logs' table.")
            cursor.execute("ALTER TABLE moderation_logs ADD COLUMN mode TEXT")
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to add mode column: {e}", exc_info=True)

    conn.close()

def log_post(data):
    """
    Logs an AI prediction output to the database.
    data format: {id, user_id, target_user, text, prediction, confidence, mode, timestamp}
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO moderation_logs 
            (id, user_id, target_user, post_text, prediction, confidence, mode, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["id"],
            data.get("user_id", "anonymous"),
            data.get("target_user", "@anonymous"),
            data["text"],
            data["prediction"],
            data["confidence"],
            data.get("mode", "mock"),
            data.get("timestamp", datetime.now().isoformat())
        ))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error logging post to DB: {e}", exc_info=True)
        return False
    finally:
        conn.close()

def update_moderator_action(post_id, action):
    """Updates the action taken by a moderator (e.g. 'Approved', 'Hidden', 'Review')."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE moderation_logs
            SET moderator_action = ?
            WHERE id = ?
        """, (action, post_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error updating moderator action: {e}", exc_info=True)
        return False
    finally:
        conn.close()

def get_all_posts():
    """Fetches all logs in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM moderation_logs ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_analytics():
    """
    Computes analytics from the moderation logs:
    - Total, Approved, Hidden counts
    - Daily cyberbullying vs safe post trends
    - Moderator agreement rate (Accuracy of AI matching human moderator actions)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Action Counts
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN moderator_action = 'Approved' THEN 1 ELSE 0 END) as approved,
            SUM(CASE WHEN moderator_action = 'Hidden' THEN 1 ELSE 0 END) as hidden,
            SUM(CASE WHEN moderator_action = 'Pending' THEN 1 ELSE 0 END) as pending
        FROM moderation_logs
    """)
    counts = dict(cursor.fetchone())
    
    # 2. Daily bullying count
    # Extract date part of ISO timestamp (YYYY-MM-DD)
    # Grabs the most recent 14 days (DESC LIMIT 14) then re-sorts ascending for the chart.
    cursor.execute("""
        SELECT 
            SUBSTR(timestamp, 1, 10) as date,
            SUM(CASE WHEN prediction = 'Cyberbullying' THEN 1 ELSE 0 END) as bullying_count,
            SUM(CASE WHEN prediction = 'Safe' THEN 1 ELSE 0 END) as safe_count
        FROM moderation_logs
        GROUP BY date
        ORDER BY date DESC
        LIMIT 14
    """)
    daily_trends = [dict(row) for row in cursor.fetchall()]
    daily_trends.reverse() # Sort ascending chronological for charts

    # 3. Moderator Accuracy (Agreement rate)
    # Definition of agreement:
    # - AI predicted Cyberbullying AND Human hid the post ('Hidden')
    # - OR AI predicted Safe AND Human approved the post ('Approved')
    # We ignore 'Pending' posts and 'Review' posts in accuracy calculation
    cursor.execute("""
        SELECT COUNT(*) FROM moderation_logs 
        WHERE moderator_action IN ('Approved', 'Hidden')
    """)
    decided_count = cursor.fetchone()[0]
    
    if decided_count > 0:
        cursor.execute("""
            SELECT COUNT(*) FROM moderation_logs
            WHERE (prediction = 'Cyberbullying' AND moderator_action = 'Hidden')
               OR (prediction = 'Safe' AND moderator_action = 'Approved')
        """)
        agreed_count = cursor.fetchone()[0]
        accuracy = round((agreed_count / decided_count) * 100, 1)
    else:
        accuracy = None  # Default to None if no posts have been processed yet

    conn.close()

    return {
        "total_posts": counts.get("total", 0) or 0,
        "approved_count": counts.get("approved", 0) or 0,
        "hidden_count": counts.get("hidden", 0) or 0,
        "pending_count": counts.get("pending", 0) or 0,
        "daily_trends": daily_trends,
        "moderator_accuracy": accuracy,
        "moderator_accuracy_sample_size": decided_count
    }
