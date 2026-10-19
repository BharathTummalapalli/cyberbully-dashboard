"""
SQLAlchemy ORM Models for Cyberbullying Detection & Moderation System.
Defines the relational schema:
1. User: Authenticated actors with role-based access control (Admin / Moderator).
2. Prediction: Log of all scanned posts, storing raw BERT score, context delta, fused score, and context features.
3. ReviewQueue: Escalated uncertain predictions (0.40 <= S_fused <= 0.70) requiring human resolution.
4. ModeratorAction: Audit log of human-in-the-loop decisions (Approved / Hidden / Notes).
5. FairnessMetric: Demographic and dialect parity metrics.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Text, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def generate_uuid() -> str:
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="moderator")  # 'admin' or 'moderator'
    created_at = Column(DateTime, default=datetime.utcnow)

    actions = relationship("ModeratorAction", back_populates="moderator")

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    text = Column(Text, nullable=False)
    author = Column(String(100), default="@anonymous", index=True)
    target_user = Column(String(100), default="@anonymous", index=True)
    toxicity_score = Column(Float, nullable=False)   # P_toxic (0.0 to 1.0)
    context_delta = Column(Float, default=0.0)      # Legacy context adjustments or delta
    p_toxic = Column(Float, nullable=True)          # Explicit P_toxic
    p_context = Column(Float, nullable=True)        # Explicit P_context
    final_score = Column(Float, nullable=False)      # S_fused / P_final (0.0 to 1.0)
    targeting_score = Column(Float, default=0.0)
    repetition_score = Column(Float, default=0.0)
    escalation_score = Column(Float, default=0.0)
    graph_score = Column(Float, default=0.0)
    label = Column(String(50), nullable=False)       # 'Safe', 'Uncertain', 'Cyberbullying'
    status = Column(String(50), default="AutoProcessed")  # 'AutoProcessed', 'EscalatedToHuman', 'Resolved'
    moderator_action = Column(String(50), default="Pending")  # 'Pending', 'Approved', 'Hidden', 'Review'
    mode = Column(String(50), default="mock")        # 'real' (BERT) or 'mock' (Lexicon)
    model_version = Column(String(50), default="toxic-bert-v1")
    context_model_version = Column(String(50), default="context-distilbert-v1")
    fusion_model_version = Column(String(50), default="fusion-mlp-v1")
    context_features_json = Column(Text, nullable=True)
    explanation_json = Column(Text, nullable=True)
    thread_id = Column(String(100), nullable=True, index=True)
    parent_id = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    review_item = relationship("ReviewQueue", back_populates="prediction", uselist=False, cascade="all, delete-orphan")
    actions = relationship("ModeratorAction", back_populates="prediction", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "author": self.author,
            "target_user": self.target_user,
            "toxicity_score": round(self.toxicity_score, 4),
            "context_delta": round(self.context_delta, 4) if self.context_delta is not None else 0.0,
            "p_toxic": round(self.p_toxic, 4) if self.p_toxic is not None else round(self.toxicity_score, 4),
            "p_context": round(self.p_context, 4) if self.p_context is not None else None,
            "final_score": round(self.final_score, 4),
            "targeting_score": round(self.targeting_score, 4) if self.targeting_score is not None else 0.0,
            "repetition_score": round(self.repetition_score, 4) if self.repetition_score is not None else 0.0,
            "escalation_score": round(self.escalation_score, 4) if self.escalation_score is not None else 0.0,
            "graph_score": round(self.graph_score, 4) if self.graph_score is not None else 0.0,
            "confidence": round(self.final_score * 100, 1),
            "prediction": self.label,
            "status": self.status,
            "moderator_action": self.moderator_action,
            "mode": self.mode,
            "model_version": self.model_version,
            "context_model_version": self.context_model_version,
            "fusion_model_version": self.fusion_model_version,
            "thread_id": self.thread_id,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    prediction_id = Column(String(36), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, unique=True)
    escalation_reason = Column(Text, nullable=False)
    status = Column(String(50), default="pending", index=True)  # 'pending', 'resolved'
    escalated_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="review_item")
    actions = relationship("ModeratorAction", back_populates="review_item", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "prediction_id": self.prediction_id,
            "escalation_reason": self.escalation_reason,
            "status": self.status,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
            "prediction": self.prediction.to_dict() if self.prediction else None
        }

class ModeratorAction(Base):
    __tablename__ = "moderator_actions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    review_id = Column(String(36), ForeignKey("review_queue.id", ondelete="CASCADE"), nullable=True)
    prediction_id = Column(String(36), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=True)
    moderator_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(50), nullable=False)  # 'Approved', 'Hidden', 'Review'
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    review_item = relationship("ReviewQueue", back_populates="actions")
    prediction = relationship("Prediction", back_populates="actions")
    moderator = relationship("User", back_populates="actions")

    def to_dict(self):
        return {
            "id": self.id,
            "review_id": self.review_id,
            "prediction_id": self.prediction_id,
            "moderator_id": self.moderator_id,
            "action": self.action,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

class FairnessMetric(Base):
    __tablename__ = "fairness_metrics"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    proxy_dataset = Column(String(255), nullable=False)
    group_name = Column(String(100), nullable=False)
    false_positive_rate = Column(Float, nullable=False)
    false_negative_rate = Column(Float, nullable=False)
    disparate_impact_ratio = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "proxy_dataset": self.proxy_dataset,
            "group_name": self.group_name,
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "disparate_impact_ratio": round(self.disparate_impact_ratio, 4),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
