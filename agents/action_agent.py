"""
Action Agent — executes moderation decisions based on Policy verdict.
Applies the decision matrix, generates audit trail, manages review queue.
"""
import json
import time
from datetime import datetime
from config import AUDIT_LOG_PATH, REVIEW_QUEUE_PATH


def decide_action(verdict: dict) -> dict:
    """Apply the decision matrix to a policy verdict."""
    severity = verdict.get("severity", "NONE")
    confidence = verdict.get("confidence", 0.0) or 0.0

    # Decision matrix (from guidelines)
    if severity == "HIGH" and confidence >= 0.75:
        action, queue, needs_recheck = "AUTO_TAKEDOWN", False, False
    elif severity == "HIGH" and confidence < 0.75:
        action, queue, needs_recheck = "URGENT_HUMAN_REVIEW", True, True
    elif severity == "MEDIUM" and confidence >= 0.70:
        action, queue, needs_recheck = "HUMAN_REVIEW", True, False
    elif severity == "MEDIUM" and confidence < 0.70:
        action, queue, needs_recheck = "LOW_PRIORITY_REVIEW", True, True
    elif severity == "LOW":
        action, queue, needs_recheck = "LOG_ONLY", False, False
    else:  # NONE
        action, queue, needs_recheck = "PASS", False, False

    return {
        "action": action,
        "queued_for_review": queue,
        "needs_recheck": needs_recheck,
        "decided_at": datetime.utcnow().isoformat(),
    }


def execute_action(segment_info: dict, verdict: dict) -> dict:
    """Execute the decided action and write to audit trail."""
    decision = decide_action(verdict)

    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "segment": segment_info.get("segment"),
        "video": segment_info.get("video", "unknown"),
        "violation": verdict.get("violation"),
        "policy_id": verdict.get("policy_id"),
        "severity": verdict.get("severity"),
        "confidence": verdict.get("confidence"),
        "action": decision["action"],
        "reasoning": verdict.get("reasoning"),
        "interplay_rule": verdict.get("interplay_rule_applied"),
    }

    # Write to audit log (append)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")

    # If queued for review, add to review queue
    if decision["queued_for_review"]:
        with open(REVIEW_QUEUE_PATH, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

    return {**decision, "audit_entry": audit_entry}


if __name__ == "__main__":
    # Test the decision matrix with sample verdicts
    print("=== Action Agent Decision Matrix Test ===\n")

    test_verdicts = [
        {"severity": "HIGH", "confidence": 0.95, "policy_id": "POLICY_VL_02", "violation": True, "reasoning": "Credible threat"},
        {"severity": "MEDIUM", "confidence": 0.80, "policy_id": "POLICY_HR_01", "violation": True, "reasoning": "Harassment"},
        {"severity": "LOW", "confidence": 0.60, "policy_id": "POLICY_PF_01", "violation": True, "reasoning": "Profanity"},
        {"severity": "NONE", "confidence": 0.0, "policy_id": None, "violation": False, "reasoning": "Safe"},
        {"severity": "HIGH", "confidence": 0.65, "policy_id": "POLICY_VL_01", "violation": True, "reasoning": "Possible violence, uncertain"},
    ]

    for v in test_verdicts:
        result = execute_action({"segment": "test", "video": "test.mp4"}, v)
        print(f"Severity={v['severity']:6} Conf={v['confidence']:.2f} -> {result['action']:22} "
              f"(review={result['queued_for_review']}, recheck={result['needs_recheck']})")

    print(f"\nAudit log written to: {AUDIT_LOG_PATH}")
