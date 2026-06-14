#!/bin/bash
# multi_user_scan.sh — Username enumeration / Scan attack
# Expected result:
#   EARLY_ALERT[SCAN] (unique_users>=4 first)
#   EARLY_ALERT[BRUTE_FORCE] (streak>=6 also fires)
#   → FINAL KNOWN_ATTACK (Brute-force, class=3)
#
# This illustrates the "SCAN consistency=0%" finding from offline evaluation:
# sessions with both SCAN and BRUTE_FORCE early alerts end up classified as
# Brute-force (class=3) because the failure streak dominates the label.
#
# Usage: ./multi_user_scan.sh <Ubuntu_VM_IP>

TARGET=$1
if [ -z "$TARGET" ]; then
    echo "Usage: $0 <Ubuntu_VM_IP>"
    exit 1
fi

echo "[*] Multi-user scan on $TARGET (hydra + top usernames)..."
hydra -L /usr/share/seclists/Usernames/top-usernames-shortlist.txt \
      -p "wrongpass" -t 4 ssh://$TARGET
echo "[*] Done."
