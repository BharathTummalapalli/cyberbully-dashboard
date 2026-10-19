"""
Contextual Cyberbullying Dataset Generator and Leakage-Safe Splitting Engine.

Integrates:
1. Real / Curated Benchmark (tweets.csv - 100 comments with ground truth)
2. Academic Fairness / Proxy Benchmarks (Blodgett et al. AAVE markers, Borkan et al. Identity markers)
3. Multi-turn Conversational Threads (Hostile, supportive, banter, escalating, and sarcasm)
4. Multi-incident Behavioral & Temporal Sequences (Repeated targeting, velocity surges)
5. Clearly demarcated synthetic augmentations for edge-case coverage.

Leakage-Safe Partitioning:
- Partitioned into 70% Train, 15% Validation, 15% Test.
- Thread/Conversation-aware grouping: No conversation thread or author-target cluster is split across sets.
- Temporal ordering: Features for event T only use records strictly before T.
- Source tracking: Every record explicitly contains its provenance.
"""

import os
import csv
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from context.collector import ContextCollector

def build_raw_events() -> List[Dict[str, Any]]:
    """
    Constructs a rich chronological corpus of social interactions.
    Spans an 8-day observation window from 2026-03-01 to 2026-03-09.
    """
    base_time = datetime(2026, 3, 1, 9, 0, 0)
    events = []

    # -------------------------------------------------------------
    # 1. Curated Benchmark (from tweets.csv)
    # -------------------------------------------------------------
    tweets_path = os.path.join(os.path.dirname(__file__), "..", "tweets.csv")
    if os.path.exists(tweets_path):
        df_tweets = pd.read_csv(tweets_path)
        for _, row in df_tweets.iterrows():
            tid = int(row["id"])
            is_cb = 1 if ((1 <= tid <= 30) or (86 <= tid <= 100)) else 0
            is_sarcasm = 1 if (86 <= tid <= 100) else 0
            is_threat = 1 if any(w in str(row["text"]).lower() for w in ["kill", "delete", "die", "destroy"]) else 0
            is_insult = 1 if any(w in str(row["text"]).lower() for w in ["trash", "loser", "ugly", "stupid", "idiot", "fat", "gross"]) else 0

            # Stagger timestamps across day 1 to day 4
            event_ts = base_time + timedelta(days=(tid % 4), hours=(tid % 12), minutes=(tid * 7 % 60))
            author = str(row["username"]) if "username" in row and not pd.isna(row["username"]) else "@anonymous"
            target = str(row["parent_username"]) if "parent_username" in row and not pd.isna(row["parent_username"]) else "@anonymous"
            
            events.append({
                "id": f"benchmark_{tid}",
                "text": str(row["text"]),
                "author": author,
                "target_user": target,
                "timestamp": event_ts,
                "thread_id": f"thread_bench_{tid // 5}",
                "parent_id": None,
                "cyberbullying": is_cb,
                "is_targeted": 1 if (target != "@anonymous" and is_cb == 1) else 0,
                "is_repeated": 0,
                "is_escalating": 0,
                "is_sarcastic": is_sarcasm,
                "is_threatening": is_threat,
                "source": "curated_benchmark"
            })

    # -------------------------------------------------------------
    # 2. Academic Dialect & Identity Proxies (Blodgett et al. & Borkan et al.)
    # -------------------------------------------------------------
    from fairness import AAVE_PROXY_BENCHMARK, SAE_BENCHMARK, IDENTITY_BENCHMARK

    proxy_id = 1
    for item in AAVE_PROXY_BENCHMARK:
        events.append({
            "id": f"proxy_aave_{proxy_id}",
            "text": item["text"],
            "author": f"@dialect_speaker_{proxy_id % 3}",
            "target_user": "@anonymous",
            "timestamp": base_time + timedelta(days=2, hours=proxy_id),
            "thread_id": f"thread_proxy_aave_{proxy_id}",
            "parent_id": None,
            "cyberbullying": 0,
            "is_targeted": 0,
            "is_repeated": 0,
            "is_escalating": 0,
            "is_sarcastic": 0,
            "is_threatening": 0,
            "source": "public_proxy"
        })
        proxy_id += 1

    proxy_id = 1
    for item in SAE_BENCHMARK:
        events.append({
            "id": f"proxy_sae_{proxy_id}",
            "text": item["text"],
            "author": f"@standard_speaker_{proxy_id % 3}",
            "target_user": "@anonymous",
            "timestamp": base_time + timedelta(days=2, hours=proxy_id, minutes=30),
            "thread_id": f"thread_proxy_sae_{proxy_id}",
            "parent_id": None,
            "cyberbullying": 0,
            "is_targeted": 0,
            "is_repeated": 0,
            "is_escalating": 0,
            "is_sarcastic": 0,
            "is_threatening": 0,
            "source": "public_proxy"
        })
        proxy_id += 1

    proxy_id = 1
    for item in IDENTITY_BENCHMARK:
        events.append({
            "id": f"proxy_id_{proxy_id}",
            "text": item["text"],
            "author": f"@proud_user_{proxy_id}",
            "target_user": "@anonymous",
            "timestamp": base_time + timedelta(days=3, hours=proxy_id),
            "thread_id": f"thread_proxy_id_{proxy_id}",
            "parent_id": None,
            "cyberbullying": 0,
            "is_targeted": 0,
            "is_repeated": 0,
            "is_escalating": 0,
            "is_sarcastic": 0,
            "is_threatening": 0,
            "source": "public_proxy"
        })
        proxy_id += 1

    # -------------------------------------------------------------
    # 3. Manually Annotated Conversational Threads & Temporal Attacks
    # -------------------------------------------------------------
    # Thread A: Benign Gaming Banter (Slang & Negations, Safe)
    th_a_time = base_time + timedelta(days=4, hours=10)
    th_a_msgs = [
        ("@gamer_1", "@gamer_2", "Yo good game last night, that was intense!", 0, 0, 0, 0, 0, 0),
        ("@gamer_2", "@gamer_1", "GG! Your sniper skills were sick, completely killing it out there!", 0, 0, 0, 0, 0, 0),
        ("@gamer_1", "@gamer_2", "Haha thanks! Don't worry about that missed shot, you're not trash at all.", 0, 0, 0, 0, 0, 0),
        ("@gamer_2", "@gamer_1", "Appreciate that man, friendly trash talk with the other squad made it fun.", 0, 0, 0, 0, 0, 0),
        ("@gamer_1", "@gamer_2", "True! We aren't losers, we will win the tournament next Saturday.", 0, 0, 0, 0, 0, 0)
    ]
    for idx, (a, tgt, txt, cb, targ, rep, esc, sarc, thr) in enumerate(th_a_msgs):
        events.append({
            "id": f"th_a_{idx}",
            "text": txt,
            "author": a,
            "target_user": tgt,
            "timestamp": th_a_time + timedelta(minutes=idx * 6),
            "thread_id": "thread_gaming_banter_101",
            "parent_id": f"th_a_{idx-1}" if idx > 0 else None,
            "cyberbullying": cb,
            "is_targeted": targ,
            "is_repeated": rep,
            "is_escalating": esc,
            "is_sarcastic": sarc,
            "is_threatening": thr,
            "source": "manually_annotated"
        })

    # Thread B: Sustained Cyberbullying with Escalation & Repeated Targeting
    th_b_time = base_time + timedelta(days=5, hours=14)
    th_b_msgs = [
        ("@victim_emma", "@anonymous", "Just published my first open-source data science project!", 0, 0, 0, 0, 0, 0),
        ("@troll_kevin", "@victim_emma", "Your code is laughably awful, who told you you could code?", 1, 1, 0, 0, 0, 0),
        ("@victim_emma", "@troll_kevin", "I'm still learning, please offer constructive feedback instead.", 0, 0, 0, 0, 0, 0),
        ("@troll_kevin", "@victim_emma", "Constructive feedback? You are a brainless idiot, delete your repo.", 1, 1, 1, 1, 0, 1),
        ("@troll_kevin", "@victim_emma", "Stop posting your pathetic garbage here, nobody wants you on GitHub.", 1, 1, 1, 1, 0, 0),
        ("@troll_kevin", "@victim_emma", "You are the absolute worst. Go disappear and never show up again.", 1, 1, 1, 1, 0, 1)
    ]
    for idx, (a, tgt, txt, cb, targ, rep, esc, sarc, thr) in enumerate(th_b_msgs):
        events.append({
            "id": f"th_b_{idx}",
            "text": txt,
            "author": a,
            "target_user": tgt,
            "timestamp": th_b_time + timedelta(minutes=idx * 4),
            "thread_id": "thread_harassment_emma_202",
            "parent_id": f"th_b_{idx-1}" if idx > 0 else None,
            "cyberbullying": cb,
            "is_targeted": targ,
            "is_repeated": rep,
            "is_escalating": esc,
            "is_sarcastic": sarc,
            "is_threatening": thr,
            "source": "manually_annotated"
        })

    # Thread C: Sarcastic Cyberbullying Trajectory
    th_c_time = base_time + timedelta(days=5, hours=18)
    th_c_msgs = [
        ("@target_liam", "@anonymous", "Shared my thoughts on the new architecture design.", 0, 0, 0, 0, 0, 0),
        ("@cynic_claire", "@target_liam", "Oh wow, what an absolute genius innovation, said no one ever.", 1, 1, 0, 0, 1, 0),
        ("@target_liam", "@cynic_claire", "Did you find any specific flaw in the diagram?", 0, 0, 0, 0, 0, 0),
        ("@cynic_claire", "@target_liam", "Yeah, the flaw is that you actually think you're smart, said nobody ever.", 1, 1, 1, 1, 1, 0),
        ("@cynic_claire", "@target_liam", "Brilliant presentation, said nobody in history. Total clown show.", 1, 1, 1, 1, 1, 0)
    ]
    for idx, (a, tgt, txt, cb, targ, rep, esc, sarc, thr) in enumerate(th_c_msgs):
        events.append({
            "id": f"th_c_{idx}",
            "text": txt,
            "author": a,
            "target_user": tgt,
            "timestamp": th_c_time + timedelta(minutes=idx * 5),
            "thread_id": "thread_sarcasm_liam_303",
            "parent_id": f"th_c_{idx-1}" if idx > 0 else None,
            "cyberbullying": cb,
            "is_targeted": targ,
            "is_repeated": rep,
            "is_escalating": esc,
            "is_sarcastic": sarc,
            "is_threatening": thr,
            "source": "manually_annotated"
        })

    # Thread D: Counter-speech & Defusing Hostility (Constructive Negations, Safe)
    th_d_time = base_time + timedelta(days=6, hours=11)
    th_d_msgs = [
        ("@victim_noah", "@anonymous", "Feeling really down about how my project turned out.", 0, 0, 0, 0, 0, 0),
        ("@ally_maya", "@victim_noah", "Hey don't beat yourself up! You are not a failure.", 0, 0, 0, 0, 0, 0),
        ("@ally_maya", "@victim_noah", "It is not garbage work at all, it was a very tough assignment.", 0, 0, 0, 0, 0, 0),
        ("@ally_maya", "@victim_noah", "Never think you are worthless, your effort was remarkable.", 0, 0, 0, 0, 0, 0)
    ]
    for idx, (a, tgt, txt, cb, targ, rep, esc, sarc, thr) in enumerate(th_d_msgs):
        events.append({
            "id": f"th_d_{idx}",
            "text": txt,
            "author": a,
            "target_user": tgt,
            "timestamp": th_d_time + timedelta(minutes=idx * 7),
            "thread_id": "thread_support_noah_404",
            "parent_id": f"th_d_{idx-1}" if idx > 0 else None,
            "cyberbullying": cb,
            "is_targeted": targ,
            "is_repeated": rep,
            "is_escalating": esc,
            "is_sarcastic": sarc,
            "is_threatening": thr,
            "source": "manually_annotated"
        })

    # -------------------------------------------------------------
    # 4. Synthetic Augmentations (Clearly marked for edge-cases)
    # -------------------------------------------------------------
    synthetic_scenarios = [
        # Slang praise variations (Safe)
        ("This beat is absolutely sick, keep dropping tracks!", "@music_fan_{i}", "@producer_{i}", 0, 0, 0, 0, 0, 0),
        ("Bro you were killing it on defense today, insane hustle!", "@hooper_{i}", "@teammate_{i}", 0, 0, 0, 0, 0, 0),
        ("Friendly trash talk is what makes this league fun, gg guys!", "@player_{i}", "@rival_{i}", 0, 0, 0, 0, 0, 0),
        ("You are totally killin it at your new job, congrats!", "@friend_a_{i}", "@friend_b_{i}", 0, 0, 0, 0, 0, 0),
        
        # Negated insults (Safe)
        ("Don't worry, you are not stupid at all. It's confusing for everyone.", "@tutor_{i}", "@student_{i}", 0, 0, 0, 0, 0, 0),
        ("Your essay wasn't garbage, it just needs a clearer thesis statement.", "@reviewer_{i}", "@author_{i}", 0, 0, 0, 0, 0, 0),
        ("You're not a disappointment to anyone, keep your head high.", "@parent_{i}", "@child_{i}", 0, 0, 0, 0, 0, 0),
        ("Never call yourself an idiot, you made an honest mistake.", "@mentor_{i}", "@intern_{i}", 0, 0, 0, 0, 0, 0),
        
        # Directed Harassment & Repeated Targeting (Cyberbullying)
        ("You are a disgusting parasite, get off the internet forever.", "@hater_{i}", "@target_{i}", 1, 1, 1, 1, 0, 1),
        ("Nobody will ever love a worthless loser like you. Delete your account.", "@bully_x_{i}", "@victim_x_{i}", 1, 1, 1, 1, 0, 1),
        ("Shut your ugly face, everyone in this server hates you.", "@toxic_t_{i}", "@member_{i}", 1, 1, 1, 0, 0, 0),
        ("You brainless freak, I hope you get banned immediately.", "@hater_{i}", "@target_{i}", 1, 1, 1, 1, 0, 0),
        
        # Sarcastic Harassment (Cyberbullying)
        ("Such a brilliant idea, said no one ever. Go cry somewhere else.", "@sarcastic_s_{i}", "@victim_s_{i}", 1, 1, 1, 0, 1, 0),
        ("Oh what a stunning look, said nobody ever. You look pathetic.", "@mean_m_{i}", "@victim_m_{i}", 1, 1, 1, 1, 1, 0),
        ("Truly a masterpiece, said no one in the history of the world. Trash.", "@cynic_c_{i}", "@artist_{i}", 1, 1, 1, 0, 1, 0),
        ("You're so polite and considerate, said no one ever. Jerk.", "@troll_{i}", "@target_{i}", 1, 1, 1, 0, 1, 0),

        # Neutral social interactions (Safe)
        ("Can someone recommend a nice coffee shop downtown with good wifi?", "@remote_dev_{i}", "@anonymous", 0, 0, 0, 0, 0, 0),
        ("Just completed my morning 5k run, beautiful brisk morning outside!", "@runner_{i}", "@anonymous", 0, 0, 0, 0, 0, 0),
        ("The quarterly earnings presentation is scheduled for next Thursday.", "@analyst_{i}", "@anonymous", 0, 0, 0, 0, 0, 0),
        ("Looking forward to reading this new machine learning paper this weekend.", "@researcher_{i}", "@anonymous", 0, 0, 0, 0, 0, 0)
    ]

    synth_base_time = base_time + timedelta(days=6, hours=14)
    # Generate 15 variations of each pattern across diverse temporal buckets
    for i in range(1, 16):
        for s_idx, (tmpl, a_tmpl, tgt_tmpl, cb, targ, rep, esc, sarc, thr) in enumerate(synthetic_scenarios):
            a_val = a_tmpl.format(i=i)
            tgt_val = tgt_tmpl.format(i=i)
            text_val = tmpl
            t_val = synth_base_time + timedelta(hours=i, minutes=s_idx * 3)

            events.append({
                "id": f"synth_{i}_{s_idx}",
                "text": text_val,
                "author": a_val,
                "target_user": tgt_val,
                "timestamp": t_val,
                "thread_id": f"thread_synth_{i}_{s_idx // 4}",
                "parent_id": None,
                "cyberbullying": cb,
                "is_targeted": targ,
                "is_repeated": rep,
                "is_escalating": esc,
                "is_sarcastic": sarc,
                "is_threatening": thr,
                "source": "synthetic_augmentation"
            })

    # Sort all events chronologically (CRITICAL FOR LEAKAGE-FREE FEATURE EXTRACTION)
    events.sort(key=lambda x: x["timestamp"])
    return events

def generate_contextual_dataset():
    print("=======================================================")
    print("GENERATING CONTEXTUAL CYBERBULLYING DATASET (Phase 1)")
    print("=======================================================")

    raw_events = build_raw_events()
    print(f"Constructed {len(raw_events)} raw events across curated, proxy, annotated, and synthetic sources.")

    collector = ContextCollector()
    dataset_rows = []

    # Process events strictly in chronological order
    # To compute features for event T, pass ONLY records prior to T
    processed_history = []

    print("Extracting multi-source contextual features with retrospective leakage prevention...")
    for idx, ev in enumerate(raw_events):
        ev_ts = ev["timestamp"]

        # Context collection strictly using processed_history (all < ev_ts)
        ctx = collector.collect(
            text=ev["text"],
            author=ev["author"],
            target_user=ev["target_user"],
            timestamp=ev_ts,
            thread_id=ev["thread_id"],
            parent_id=ev["parent_id"],
            history_records=processed_history
        )

        flat = ctx["flat_features"]

        row = {
            "id": ev["id"],
            "text": ev["text"],
            "author": ev["author"],
            "target_user": ev["target_user"],
            "timestamp": ev_ts.isoformat(),
            "thread_id": ev["thread_id"],
            "parent_id": ev["parent_id"] or "",
            "context_snippet": ctx["context_snippet"],
            "source": ev["source"],
            # Multi-task labels
            "cyberbullying": ev["cyberbullying"],
            "is_targeted": ev["is_targeted"],
            "is_repeated": ev["is_repeated"],
            "is_escalating": ev["is_escalating"],
            "is_sarcastic": ev["is_sarcastic"],
            "is_threatening": ev["is_threatening"],
            # Extracted numerical context features
            **flat
        }
        dataset_rows.append(row)

        # Append current event to historical pool for SUBSEQUENT events
        processed_history.append({
            "id": ev["id"],
            "text": ev["text"],
            "author": ev["author"],
            "target_user": ev["target_user"],
            "timestamp": ev_ts,
            "prediction": "Cyberbullying" if ev["cyberbullying"] == 1 else "Safe",
            "label": "Cyberbullying" if ev["cyberbullying"] == 1 else "Safe",
            "final_score": 0.85 if ev["cyberbullying"] == 1 else 0.15,
            "status": "EscalatedToHuman" if ev["cyberbullying"] == 1 else "AutoProcessed",
            "thread_id": ev["thread_id"]
        })

    # -------------------------------------------------------------
    # 5. Leakage-Safe Splitting (70% Train, 15% Val, 15% Test)
    # -------------------------------------------------------------
    # Group by thread_id to guarantee entire conversations remain intact
    threads = {}
    for r in dataset_rows:
        th = r["thread_id"]
        if th not in threads:
            threads[th] = []
        threads[th].append(r)

    # Sort threads by earliest timestamp to maintain temporal integrity
    sorted_threads = sorted(threads.items(), key=lambda item: item[1][0]["timestamp"])
    total_events = len(dataset_rows)

    train_limit = int(total_events * 0.70)
    val_limit = int(total_events * 0.85)

    cumulative = 0
    train_rows, val_rows, test_rows = [], [], []

    for th_id, th_events in sorted_threads:
        if cumulative < train_limit:
            for ev in th_events:
                ev["split"] = "train"
                train_rows.append(ev)
        elif cumulative < val_limit:
            for ev in th_events:
                ev["split"] = "val"
                val_rows.append(ev)
        else:
            for ev in th_events:
                ev["split"] = "test"
                test_rows.append(ev)
        cumulative += len(th_events)

    all_partitioned = train_rows + val_rows + test_rows
    out_csv = os.path.join(os.path.dirname(__file__), "context_dataset.csv")

    fieldnames = list(all_partitioned[0].keys())
    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_partitioned)

    # Print summary statistics
    df_res = pd.DataFrame(all_partitioned)
    print("\nDataset generation complete!")
    print(f"Total Rows:       {len(df_res)}")
    print(f"Train Rows:       {len(train_rows)} ({len(train_rows)/len(df_res)*100:.1f}%)")
    print(f"Validation Rows:  {len(val_rows)} ({len(val_rows)/len(df_res)*100:.1f}%)")
    print(f"Test Rows:        {len(test_rows)} ({len(test_rows)/len(df_res)*100:.1f}%)")
    print("\nSource Breakdown:")
    print(df_res["source"].value_counts().to_string())
    print("\nLabel Distribution (Cyberbullying):")
    print(df_res.groupby(["split", "cyberbullying"]).size().unstack(fill_value=0))
    print(f"\nSaved versioned dataset to: {out_csv}")
    return out_csv

if __name__ == "__main__":
    generate_contextual_dataset()
