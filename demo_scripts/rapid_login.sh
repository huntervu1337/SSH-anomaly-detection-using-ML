#!/bin/bash
# rapid_login.sh — Best-effort UNKNOWN_PATTERN trigger
# Expected result: either NORMAL or UNKNOWN_PATTERN (both valid outcomes)
#
# ⚠️ NOT GUARANTEED: This session has failure_ratio=0 → Layer 1 predicts
# Normal (correct). Whether Layer 2 (IF) flags it as UNKNOWN_PATTERN depends
# on whether the feature vector (duration≈0, high attempts_per_second,
# multi-success) falls outside the decision boundary learned from 123 Normal
# sessions in the original dataset.
#
# Both outcomes are valid for the report:
#   - If flagged: direct evidence Layer 2 works on live traffic
#   - If not flagged: demonstrates the known limitation that IF trained on
#     123 samples may not generalize to demo environment traffic
#
# Usage: ./rapid_login.sh <Ubuntu_VM_IP>

TARGET=$1
if [ -z "$TARGET" ]; then
    echo "Usage: $0 <Ubuntu_VM_IP>"
    exit 1
fi

echo "[*] Rapid concurrent logins to $TARGET as alice..."

# Multiple simultaneous successful logins — simulates session 1783
# (2 successful logins, duration=0s, abnormally high attempts_per_second)
for i in 1 2 3; do
    sshpass -p "CorrectPass123" ssh -o StrictHostKeyChecking=no alice@$TARGET "exit" &
done
wait

echo "[*] Done."
