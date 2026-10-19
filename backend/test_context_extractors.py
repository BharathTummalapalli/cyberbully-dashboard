"""
Unit and Regression Tests for Context Extractors and Leakage Prevention.
Verifies:
1. Temporal Leakage Prevention (Strict inequality on timestamp < T)
2. Conversation context extraction and parent resolving
3. Behavioral attack counting and repetition metrics
4. Temporal velocity and escalation curves
5. Graph metrics and role assignments
"""

from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from context.collector import ContextCollector
from context.conversation import ConversationContextExtractor
from context.behavioral import BehavioralContextExtractor
from context.temporal import TemporalContextExtractor
from context.graph_features import GraphFeatureExtractor

def test_leakage_prevention():
    print("Testing temporal leakage prevention...")
    t_now = datetime(2026, 3, 15, 12, 0, 0)

    # Simulated history with past events AND a future event
    history = [
        # Past events (< t_now)
        {"id": "1", "author": "@attacker_sam", "target_user": "@victim_vic", "timestamp": t_now - timedelta(hours=2), "prediction": "Cyberbullying", "label": "Cyberbullying", "final_score": 0.9},
        {"id": "2", "author": "@attacker_sam", "target_user": "@victim_vic", "timestamp": t_now - timedelta(minutes=30), "prediction": "Cyberbullying", "label": "Cyberbullying", "final_score": 0.85},
        # Concurrent / Future events (>= t_now) -> MUST NOT BE INCLUDED!
        {"id": "3", "author": "@attacker_sam", "target_user": "@victim_vic", "timestamp": t_now, "prediction": "Cyberbullying", "label": "Cyberbullying", "final_score": 0.95},
        {"id": "4", "author": "@attacker_sam", "target_user": "@victim_vic", "timestamp": t_now + timedelta(minutes=10), "prediction": "Cyberbullying", "label": "Cyberbullying", "final_score": 0.99},
    ]

    temp_extractor = TemporalContextExtractor()
    t_res = temp_extractor.extract("@attacker_sam", t_now, history)

    # At t_now, only events 1 and 2 exist in the past!
    # Event 2 was 30 mins ago (in 1h window). Event 1 was 2 hours ago (in 6h window).
    assert t_res["v_1h"] == 1, f"Expected v_1h=1, got {t_res['v_1h']}"
    assert t_res["v_6h"] == 2, f"Expected v_6h=2, got {t_res['v_6h']}"
    assert t_res["time_since_prev_attack_hours"] == 0.5, f"Expected 0.5 hours, got {t_res['time_since_prev_attack_hours']}"

    behav_extractor = BehavioralContextExtractor()
    b_res = behav_extractor.extract("@attacker_sam", "@victim_vic", t_now, history)
    assert b_res["total_attacks_sent"] == 2, f"Expected 2 past attacks, got {b_res['total_attacks_sent']}"
    assert b_res["target_attacks_count"] == 2, f"Expected 2 target attacks, got {b_res['target_attacks_count']}"

    print("  [PASS] Temporal leakage prevention verified: zero future records leaked.")

def test_conversation_context():
    print("Testing conversation context extraction...")
    t_now = datetime(2026, 3, 15, 12, 0, 0)
    history = [
        {"id": "m1", "text": "What do you think of this design?", "author": "@victim_vic", "target_user": "@attacker_sam", "timestamp": t_now - timedelta(minutes=15), "thread_id": "th1"},
        {"id": "m2", "text": "It has some potential, looks nice.", "author": "@friend_fred", "target_user": "@victim_vic", "timestamp": t_now - timedelta(minutes=10), "thread_id": "th1"},
        {"id": "m3", "text": "Thanks Fred, working hard on it!", "author": "@victim_vic", "target_user": "@friend_fred", "timestamp": t_now - timedelta(minutes=5), "thread_id": "th1"}
    ]

    conv_extractor = ConversationContextExtractor()
    res = conv_extractor.extract(
        text="Actually it's pure garbage, you are stupid.",
        author="@attacker_sam",
        target_user="@victim_vic",
        timestamp=t_now,
        thread_id="th1",
        history_records=history
    )

    assert res["turn_count"] == 3, f"Expected turn_count=3, got {res['turn_count']}"
    assert len(res["previous_messages"]) == 3, f"Expected 3 previous messages, got {len(res['previous_messages'])}"
    assert res["parent_text"] == "Thanks Fred, working hard on it!", f"Parent text mismatch: {res['parent_text']}"
    assert res["time_delta_seconds"] == 300.0, f"Expected 300s, got {res['time_delta_seconds']}"
    print("  [PASS] Conversation context verified.")

def test_graph_features():
    print("Testing graph feature extraction & roles...")
    t_now = datetime(2026, 3, 15, 12, 0, 0)
    # Give @serial_bully 4 attacks in the past 7 days to qualify as Abuser (>= 3)
    history = [
        {"id": f"a{i}", "author": "@serial_bully", "target_user": "@victim_vic", "timestamp": t_now - timedelta(days=i), "prediction": "Cyberbullying", "label": "Cyberbullying", "final_score": 0.9}
        for i in range(1, 5)
    ]

    graph_extractor = GraphFeatureExtractor()
    res = graph_extractor.extract("@serial_bully", "@victim_vic", t_now, history)

    assert res["author_role"] == "Abuser", f"Expected 'Abuser', got {res['author_role']}"
    assert res["target_role"] == "Victim", f"Expected 'Victim', got {res['target_role']}"
    assert res["edge_weight"] == 4.0, f"Expected edge_weight=4, got {res['edge_weight']}"
    assert res["author_role_abuser"] == 1.0, f"Expected author_role_abuser=1.0"
    assert res["target_role_victim"] == 1.0, f"Expected target_role_victim=1.0"
    print("  [PASS] Graph features and role logic verified.")

def test_context_collector():
    print("Testing unified ContextCollector...")
    t_now = datetime(2026, 3, 15, 12, 0, 0)
    history = [
        {"id": "1", "author": "@bully_bob", "target_user": "@victim_vic", "timestamp": t_now - timedelta(hours=1), "prediction": "Cyberbullying", "label": "Cyberbullying", "final_score": 0.88, "text": "Shut up loser."}
    ]
    collector = ContextCollector()
    res = collector.collect(
        text="You are still a loser.",
        author="@bully_bob",
        target_user="@victim_vic",
        timestamp=t_now,
        history_records=history
    )

    assert "conversation" in res
    assert "behavioral" in res
    assert "temporal" in res
    assert "graph" in res
    assert "flat_features" in res
    assert len(res["flat_features"]) >= 15
    print(f"  [PASS] ContextCollector assembled {len(res['flat_features'])} numerical features cleanly.")

if __name__ == "__main__":
    test_leakage_prevention()
    test_conversation_context()
    test_graph_features()
    test_context_collector()
    print("\nALL CONTEXT EXTRACTOR TESTS PASSED!")
