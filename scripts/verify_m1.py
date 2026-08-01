"""
verify_m1.py — Verification Runner for Milestone M1 (Dynamic Length Validation).
"""
import sys
import os

generators_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "generators"))
if generators_dir not in sys.path:
    sys.path.append(generators_dir)

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
if data_dir not in sys.path:
    sys.path.append(data_dir)

from validator import validate_conversation_structure, COGITO_SYSTEM_PROMPT


def test_imports():
    print("Testing imports across all refactored modules...")
    import agentic_tools
    import execution_engine
    import identity_core
    import human_conversations
    import personality_quirks
    import retrieval_filter
    import merge_datasets
    print("[PASS] All 7 refactored modules imported successfully!")


def run_unit_tests():
    print("Running validator unit tests...")
    tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
    if tests_dir not in sys.path:
        sys.path.append(tests_dir)

    import test_validator
    
    test_funcs = [
        test_validator.test_valid_3turn_passes,
        test_validator.test_valid_5turn_passes,
        test_validator.test_valid_7turn_passes,
        test_validator.test_invalid_turn0_system_prompt,
        test_validator.test_invalid_turn0_role,
        test_validator.test_missing_required_tags,
        test_validator.test_out_of_range_confidence,
        test_validator.test_sycophancy_detected,
        test_validator.test_invalid_intermediate_loop_action,
        test_validator.test_intermediate_loop_action_missing_feedback_turn,
        test_validator.test_invalid_final_terminal_action,
        test_validator.test_max_turn_ceiling_exceeded,
        test_validator.test_min_turn_floor,
        test_validator.test_uncalibrated_confidence,
    ]

    passed, failed = 0, 0
    for func in test_funcs:
        try:
            func()
            print(f"  [PASS] {func.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {func.__name__}: {e}")
            failed += 1

    print(f"\nUnit Test Results: {passed} PASSED, {failed} FAILED")
    assert failed == 0, f"{failed} unit tests failed!"


def verify_master_dataset():
    master_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cogito_0.9_master_dataset.jsonl"))
    if not os.path.exists(master_dataset_path):
        print(f"[SKIP] Master dataset not found at {master_dataset_path}")
        return

    print(f"Validating existing master dataset: {master_dataset_path}")
    import json
    valid_count, total_count = 0, 0
    with open(master_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            rec = json.loads(line)
            res = validate_conversation_structure(rec.get("messages", []))
            if res:
                valid_count += 1
            else:
                print(f"  [INVALID RECORD {total_count}]: {res[1]}")

    print(f"Master Dataset Verification: {valid_count}/{total_count} records valid.")


if __name__ == "__main__":
    test_imports()
    run_unit_tests()
    verify_master_dataset()
    print("\n[SUCCESS] Milestone M1 Verification Complete!")
