#!/bin/bash
# normal_login.sh — Baseline NORMAL scenario
# Expected result: FINAL NORMAL, layer1_class=0
#
# Usage: ./normal_login.sh <Ubuntu_VM_IP>

TARGET=$1
if [ -z "$TARGET" ]; then
    echo "Usage: $0 <Ubuntu_VM_IP>"
    exit 1
fi

echo "[*] Normal login to $TARGET as alice..."
sshpass -p "CorrectPass123" ssh -o StrictHostKeyChecking=no alice@$TARGET "echo OK"
echo "[*] Done."
