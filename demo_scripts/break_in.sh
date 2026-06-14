#!/bin/bash
# break_in.sh — Break-in scenario (failed then success with valid user)
# Expected result:
#   EARLY_ALERT[BREAK_IN_RISK] when alice succeeds after failures
#   → FINAL KNOWN_ATTACK (Break-in, class=4) + beep
#
# Usage: ./break_in.sh <Ubuntu_VM_IP>

TARGET=$1
if [ -z "$TARGET" ]; then
    echo "Usage: $0 <Ubuntu_VM_IP>"
    exit 1
fi

echo "[*] Break-in scenario on $TARGET (2 wrong passwords then correct)..."

# 2 failed attempts with valid user
for i in 1 2; do
    sshpass -p "wrongpass$i" ssh -o StrictHostKeyChecking=no alice@$TARGET "exit" 2>/dev/null
    sleep 1
done

# 3rd attempt: correct password → triggers BREAK_IN_RISK
sshpass -p "CorrectPass123" ssh -o StrictHostKeyChecking=no alice@$TARGET "echo Access granted"

echo "[*] Done."
