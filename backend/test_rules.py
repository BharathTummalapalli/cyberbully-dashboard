import sys
import os
import explain

def run_tests():
    print("Executing linguistic context rules regression tests...")
    
    test_cases = [
        {
            "name": "Compound negation (negated + active insult)",
            "text": "You are not stupid but you are the biggest idiot I know",
            "expected_override": False,
            "expected_prediction": "" # Should let baseline decide (which is Cyberbullying)
        },
        {
            "name": "Slang + genuine toxicity mixture",
            "text": "You are killing it on stage but you are a stupid loser",
            "expected_override": False,
            "expected_prediction": "" # Should let baseline decide
        },
        {
            "name": "Sarcasm over negation",
            "text": "You are not stupid, said no one ever",
            "expected_override": True,
            "expected_prediction": "Cyberbullying"
        },
        {
            "name": "Multiple unnegated insults",
            "text": "You are an ugly stupid loser",
            "expected_override": False,
            "expected_prediction": ""
        },
        {
            "name": "Wider negation window (5-token lookback)",
            "text": "Don't think you are stupid",
            "expected_override": True,
            "expected_prediction": "Safe"
        },
        {
            "name": "Proximity-bound slang positive flow",
            "text": "That was a sick beat, nicely played",
            "expected_override": True,
            "expected_prediction": "Safe"
        },
        {
            "name": "Proximity-bound slang isolated from insult",
            "text": "This is a sick beat, but you are a stupid trash poster",
            "expected_override": False,
            "expected_prediction": "" # Insult should win
        }
    ]
    
    failed = 0
    for i, tc in enumerate(test_cases, 1):
        res = explain.check_context_overrides(tc["text"])
        
        override_match = res["overridden"] == tc["expected_override"]
        pred_match = res["prediction"] == tc["expected_prediction"]
        
        if override_match and pred_match:
            print(f"  [PASS] Test {i}: {tc['name']}")
        else:
            print(f"  [FAIL] Test {i}: {tc['name']}")
            print(f"         Text:     '{tc['text']}'")
            print(f"         Expected: Overridden={tc['expected_override']}, Prediction='{tc['expected_prediction']}'")
            print(f"         Got:      Overridden={res['overridden']}, Prediction='{res['prediction']}' (Explanation: '{res.get('explanation')}')")
            failed += 1
            
    if failed == 0:
        print("\nALL REGRESSION TESTS PASSED SUCCESSFULLY!")
        return True
    else:
        print(f"\n{failed} REGRESSION TEST(S) FAILED.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
