"""
CheckList Behavioral Test Suite for NLP Cyberbullying Detection.
Based on the CheckList methodology (Ribeiro et al., ACL 2020):
1. Minimum Functionality Tests (MFT): Simple test cases evaluating basic competence.
2. Invariance Tests (INV): Perturbations that should not alter predictions (e.g., name substitution, punctuation).
3. Directional Expectation Tests (DIR): Perturbations with predictable directional output shift:
   - DIR-Negation: Negating an insult must decrease toxicity.
   - DIR-Sarcasm: Adding a sarcastic qualifier must increase toxicity.
   - DIR-Identity: Inserting protected identity words into neutral statements must not increase toxicity.
"""

import sys
import explain
import fusion

def run_checklist_suite() -> dict:
    print("\n=======================================================")
    print("RUNNING CHECKLIST BEHAVIORAL TESTING SUITE (Ribeiro et al.)")
    print("=======================================================\n")
    
    use_real = explain.ML_AVAILABLE
    results = {"mft": {}, "inv": {}, "dir": {}}
    
    # -------------------------------------------------------------
    # 1. MINIMUM FUNCTIONALITY TESTS (MFT)
    # -------------------------------------------------------------
    mft_cases = [
        {"text": "You are the worst human being alive, delete your account", "expected": "Cyberbullying", "desc": "Unambiguous direct harassment"},
        {"text": "Thank you so much for the helpful guidance, really appreciate it", "expected": "Safe", "desc": "Unambiguous polite gratitude"},
        {"text": "I hope you get banned soon, you are a toxic parasite", "expected": "Cyberbullying", "desc": "Hostile demeaning insult"},
        {"text": "The sunset at the beach this evening was so breathtaking", "expected": "Safe", "desc": "Neutral scenic appreciation"},
        {"text": "Nobody likes you, you absolute loser", "expected": "Cyberbullying", "desc": "Direct personal insult"}
    ]
    
    mft_passed = 0
    for case in mft_cases:
        res = explain.get_lime_explanation(case["text"], use_real_model=use_real, bypass_context_rules=True, skip_lime=True)
        p_toxic = (res["confidence"] / 100.0) if res["prediction"] == "Cyberbullying" else (1.0 - res["confidence"] / 100.0)
        fused = fusion.compute_fused_score(case["text"], p_toxic)
        passed = (fused["label"] == case["expected"])
        if passed:
            mft_passed += 1
            print(f"  [PASS] MFT: {case['desc']}")
        else:
            print(f"  [FAIL] MFT: {case['desc']} (Expected: {case['expected']}, Got: {fused['label']})")
            
    mft_rate = round((mft_passed / len(mft_cases)) * 100, 1)
    results["mft"] = {"total": len(mft_cases), "passed": mft_passed, "pass_rate": mft_rate}

    # -------------------------------------------------------------
    # 2. INVARIANCE TESTS (INV) - Name & Punctuation Substitution
    # -------------------------------------------------------------
    inv_pairs = [
        ("Good job on the test today Katie!", "Good job on the test today Helen.", "Safe"),
        ("You look like garbage today Dan.", "You look like garbage today Harry.", "Cyberbullying"),
        ("Have a wonderful week everyone!", "Have a wonderful week everyone.", "Safe"),
        ("Stop posting stupid comments.", "Stop posting stupid comments!", "Cyberbullying")
    ]
    
    inv_passed = 0
    for t1, t2, expected in inv_pairs:
        r1 = explain.get_lime_explanation(t1, use_real_model=use_real, bypass_context_rules=True, skip_lime=True)
        r2 = explain.get_lime_explanation(t2, use_real_model=use_real, bypass_context_rules=True, skip_lime=True)
        p1 = (r1["confidence"] / 100.0) if r1["prediction"] == "Cyberbullying" else (1.0 - r1["confidence"] / 100.0)
        p2 = (r2["confidence"] / 100.0) if r2["prediction"] == "Cyberbullying" else (1.0 - r2["confidence"] / 100.0)
        
        f1 = fusion.compute_fused_score(t1, p1)
        f2 = fusion.compute_fused_score(t2, p2)
        
        passed = (f1["label"] == f2["label"]) and (f1["label"] == expected)
        if passed:
            inv_passed += 1
            print(f"  [PASS] INV: '{t1}' <-> '{t2}' (Preserved '{expected}')")
        else:
            print(f"  [FAIL] INV: Invariant shift detected between '{t1}' and '{t2}'")
            
    inv_rate = round((inv_passed / len(inv_pairs)) * 100, 1)
    results["inv"] = {"total": len(inv_pairs), "passed": inv_passed, "pass_rate": inv_rate}

    # -------------------------------------------------------------
    # 3. DIRECTIONAL EXPECTATION TESTS (DIR)
    # -------------------------------------------------------------
    dir_passed = 0
    dir_tests = [
        # DIR-Negation: Negating an insult should drop the score
        {
            "type": "Negation",
            "base": "You are stupid and trash",
            "perturbed": "You are not stupid and not trash",
            "expectation": "decrease"
        },
        # DIR-Sarcasm: Adding sarcasm qualifier to praise should increase score
        {
            "type": "Sarcasm",
            "base": "You are an absolute genius",
            "perturbed": "You are an absolute genius, said no one ever",
            "expectation": "increase"
        },
        # DIR-Identity: Inserting identity terms into neutral text should NOT cause significant increase (<= +0.10)
        {
            "type": "Identity Invariance",
            "base": "I am a professional software engineer",
            "perturbed": "I am a proud Muslim software engineer",
            "expectation": "stable"
        }
    ]
    
    for dt in dir_tests:
        r_base = explain.get_lime_explanation(dt["base"], use_real_model=use_real, bypass_context_rules=True, skip_lime=True)
        r_pert = explain.get_lime_explanation(dt["perturbed"], use_real_model=use_real, bypass_context_rules=True, skip_lime=True)
        
        p_base = (r_base["confidence"] / 100.0) if r_base["prediction"] == "Cyberbullying" else (1.0 - r_base["confidence"] / 100.0)
        p_pert = (r_pert["confidence"] / 100.0) if r_pert["prediction"] == "Cyberbullying" else (1.0 - r_pert["confidence"] / 100.0)
        
        f_base = fusion.compute_fused_score(dt["base"], p_base)
        f_pert = fusion.compute_fused_score(dt["perturbed"], p_pert)
        
        score_diff = f_pert["s_fused"] - f_base["s_fused"]
        
        passed = False
        if dt["expectation"] == "decrease" and score_diff < -0.10:
            passed = True
        elif dt["expectation"] == "increase" and score_diff > +0.10:
            passed = True
        elif dt["expectation"] == "stable" and abs(score_diff) < 0.15:
            passed = True
            
        if passed:
            dir_passed += 1
            print(f"  [PASS] DIR ({dt['type']}): '{dt['base']}' -> '{dt['perturbed']}' (Delta: {score_diff:+.3f})")
        else:
            print(f"  [FAIL] DIR ({dt['type']}): Directional expectation failed (Delta: {score_diff:+.3f})")
            
    dir_rate = round((dir_passed / len(dir_tests)) * 100, 1)
    results["dir"] = {"total": len(dir_tests), "passed": dir_passed, "pass_rate": dir_rate}

    print("\n-------------------------------------------------------")
    print(f"CHECKLIST SUMMARY: MFT: {mft_rate}% | INV: {inv_rate}% | DIR: {dir_rate}%")
    print("-------------------------------------------------------\n")
    return results

if __name__ == "__main__":
    res = run_checklist_suite()
    all_passed = (res["mft"]["pass_rate"] >= 80 and res["inv"]["pass_rate"] >= 80 and res["dir"]["pass_rate"] >= 80)
    sys.exit(0 if all_passed else 1)
