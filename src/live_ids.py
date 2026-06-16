"""Live SSH Intrusion Detection System — real-time daemon.

Monitors /var/log/auth.log in real-time using a dual-thread architecture:
  Thread 1 (Tailer):  tail -f → parse → buffer → early alert check
  Thread 2 (Poller):  periodic sweep of expired sessions → final classify

Run with:  sudo .venv/bin/python src/live_ids.py

Based on live_demo_spec.md — demo variant of the offline pipeline.
"""

from __future__ import annotations

import sys
import threading
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from log_processing import SSHLogParser
from data_labeling import aggregate_session
from feature_engineering import prepare_features, CLASS_NAMES
from realtime_simulator import SessionState, EarlyAlertEngine

try:
    from rich.console import Console
except ImportError:
    # Fallback: minimal Console if rich is not installed
    class Console:
        def print(self, *args, **kwargs):
            style = kwargs.pop("style", "")
            print(*args)
        def bell(self):
            print("\a", end="")


# Determine if terminal stdout encoding supports emojis safely
def _can_encode_emojis() -> bool:
    try:
        "🔴".encode(sys.stdout.encoding or "utf-8")
        "⚠️".encode(sys.stdout.encoding or "utf-8")
        return True
    except Exception:
        return False

USE_EMOJI = _can_encode_emojis()


# ---- Demo config (different from offline research config) ----
AUTH_LOG    = "/var/log/auth.log"
IDLE_GAP    = 30          # seconds — SHORTENED for demo (offline research uses 600s)
POLL_EVERY  = 5           # seconds — polling thread sweep interval
DEMO_VALID_USERS = {"alice"}   # hardcoded — DON'T use infer_valid_users_from_file
                                # (file is streaming, not "complete")

OUT_CSV   = "alerts.csv"
OUT_JSONL = "alerts.jsonl"


# ---------------------------------------------------------------------------
# LiveSessionBuffer — thread-safe variant of SessionBuffer
# ---------------------------------------------------------------------------

class LiveSessionBuffer:
    """Thread-safe session buffer with check_expired() for polling thread.

    Same logic as SessionBuffer in realtime_simulator, but uses
    threading.Lock() since two threads access it concurrently.
    """

    def __init__(self, idle_gap: int):
        self.idle_gap = idle_gap
        self.sessions: dict[str, SessionState] = {}
        self.lock = threading.Lock()

    def add_event(self, record: dict):
        """Called from Tailer thread. Same logic as SessionBuffer.add_event."""
        ip = str(record["ip"])
        ts = int(record["ts"])

        with self.lock:
            closed_state = None
            if ip in self.sessions:
                state = self.sessions[ip]
                if ts - state.ts_last > self.idle_gap:
                    closed_state = state
                    del self.sessions[ip]

            if ip not in self.sessions:
                self.sessions[ip] = SessionState(ip=ip, ts_first=ts)

            state = self.sessions[ip]
            state.records.append(record)
            state.ts_last = ts
            state.total_attempts += 1

            if int(record["is_failure"]) == 1:
                state.total_failures += 1
                state.cur_failure_streak += 1
                state.max_failure_streak = max(state.max_failure_streak, state.cur_failure_streak)
            else:
                state.cur_failure_streak = 0

            state.unique_users.add(str(record["user"]))
            state.has_root_attempt = max(state.has_root_attempt, int(record["is_root"]))
            state.has_valid_user_attempt = max(state.has_valid_user_attempt, int(record["is_valid"]))

            return closed_state, state

    def check_expired(self, now_ts: int):
        """Called from Polling thread. Returns list of sessions idle > idle_gap."""
        with self.lock:
            expired_ips = [
                ip for ip, state in self.sessions.items()
                if now_ts - state.ts_last > self.idle_gap
            ]
            return [self.sessions.pop(ip) for ip in expired_ips]

    def flush(self):
        """Close all remaining sessions (shutdown)."""
        with self.lock:
            remaining = list(self.sessions.values())
            self.sessions.clear()
            return remaining


# ---------------------------------------------------------------------------
# Final Classification — is_private override for VM lab
# ---------------------------------------------------------------------------

def final_classify(closed_state: SessionState, rf_model, iso_model, had_early: bool) -> dict:
    """Classify a closed session through Layer 1 (RF) + Layer 2 (IF).

    Includes is_private=0 override: VM lab uses private IPs (192.168.x.x)
    which is OOD vs training data (100% is_private=0). Since this feature
    has importance=0 in both RF and decoupled model, hardcoding to 0
    avoids IF bias without losing information.
    """
    row = aggregate_session(closed_state.records)

    # ⚠️ FIX: is_private is CONSTANT=0 in 100% training data (3916 sessions).
    # VM lab uses private IPs (192.168.x.x) → is_private=1 — OOD value never
    # seen in training. Since this feature is confirmed uninformative
    # (importance=0 in both RF and decoupled model), hardcode to 0 to match
    # training distribution and avoid IsolationForest bias.
    row["is_private"] = 0

    df = pd.DataFrame([row])
    df["class"] = 0  # dummy — prepare_features needs this column
    X_row, _ = prepare_features(df)

    l1_class  = int(rf_model.predict(X_row)[0])
    score     = float(iso_model.decision_function(X_row)[0])
    is_anom   = int(iso_model.predict(X_row)[0] == -1)

    alert_type = (
        "KNOWN_ATTACK"    if l1_class > 0 else
        "UNKNOWN_PATTERN" if is_anom else
        "NORMAL"
    )

    return {
        "alert_kind":       "FINAL_CLASSIFICATION",
        "alert_type":       alert_type,
        "ts_alert":         row["ts_last"],
        "ip":               row["ip"],
        "session_duration": row["session_duration"],
        "total_attempts":   row["total_attempts"],
        "layer1_class":     l1_class,
        "layer1_label":     CLASS_NAMES[l1_class],
        "layer2_score":     round(score, 4),
        "layer2_anomaly":   is_anom,
        "had_early_alert":  int(had_early),
    }


# ---------------------------------------------------------------------------
# Console Output (rich)
# ---------------------------------------------------------------------------

def emit_alert(alert: dict, console: Console, csv_writer, jsonl_fh, fieldnames):
    """Print alert to console and write to CSV/JSONL."""
    ts_str = datetime.fromtimestamp(alert["ts_alert"]).strftime("%H:%M:%S")

    if alert["alert_kind"] == "EARLY_ALERT":
        if USE_EMOJI:
            try:
                console.print(
                    f"[{ts_str}] ⚠️  EARLY_ALERT [{alert['alert_subtype']}] "
                    f"IP={alert['ip']} attempt#{alert['event_index']} — {alert['reason']}",
                    style="bold yellow",
                )
                return
            except Exception:
                pass
        print(
            f"[{ts_str}] [!] EARLY_ALERT [{alert['alert_subtype']}] "
            f"IP={alert['ip']} attempt#{alert['event_index']} - {alert['reason']}"
        )
    else:
        atype = alert["alert_type"]
        if USE_EMOJI:
            try:
                style, icon = {
                    "KNOWN_ATTACK":    ("bold red",     "🔴"),
                    "UNKNOWN_PATTERN": ("bold magenta", "🟣"),
                    "NORMAL":          ("bold green",   "🟢"),
                }[atype]
                console.print(
                    f"[{ts_str}] {icon} FINAL [{atype}] {alert['layer1_label']} "
                    f"IP={alert['ip']} attempts={alert['total_attempts']} "
                    f"dur={alert['session_duration']}s IF_score={alert['layer2_score']}",
                    style=style,
                )
                if alert.get("layer1_class") in (3, 4):  # Brute-force or Break-in
                    console.bell()
                return
            except Exception:
                pass
        
        ascii_icon = {
            "KNOWN_ATTACK":    "[CRIT]",
            "UNKNOWN_PATTERN": "[WARN]",
            "NORMAL":          "[ OK ]",
        }[atype]
        print(
            f"[{ts_str}] {ascii_icon} FINAL [{atype}] {alert['layer1_label']} "
            f"IP={alert['ip']} attempts={alert['total_attempts']} "
            f"dur={alert['session_duration']}s IF_score={alert['layer2_score']}"
        )

    # Structured logging — for post-analysis / downstream tools
    row = {k: alert.get(k, "") for k in fieldnames}
    csv_writer.writerow(row)
    jsonl_fh.write(json.dumps(alert) + "\n")
    jsonl_fh.flush()


# ---------------------------------------------------------------------------
# Thread 1 — Log Tailer (blocking I/O)
# ---------------------------------------------------------------------------

def tail_f(path: str, read_all: bool = False, caught_up_event: Optional[threading.Event] = None):
    """Generator that yields new lines from a file, similar to `tail -f`.
    
    If read_all is True, reads from the beginning of the file.
    Triggers caught_up_event when the end of the file is first reached.
    """
    with open(path, "r", errors="ignore") as f:
        if not read_all:
            f.seek(0, 2)  # seek to END — only process NEW log lines from start
            if caught_up_event:
                caught_up_event.set()
        
        has_set_caught_up = False if read_all else True
        
        while True:
            line = f.readline()
            if not line:
                if not has_set_caught_up:
                    has_set_caught_up = True
                    if caught_up_event:
                        caught_up_event.set()
                time.sleep(0.2)
                continue
            yield line


def tailer_thread(stop_event, caught_up_event, log_file, read_all, verbose, parser, buffer, early_engine, rf_model, iso_model,
                   console, csv_writer, jsonl_fh, fieldnames):
    """Thread 1: reads new log lines, updates buffer, checks early alerts."""
    for line in tail_f(log_file, read_all=read_all, caught_up_event=caught_up_event):
        if stop_event.is_set():
            break
        
        stripped = line.strip()
        if verbose:
            console.print(f"[DEBUG] Raw line read: {stripped}", style="dim yellow")
            
        record = parser.parse_line(stripped)
        if record is None:
            if verbose and "sshd" in stripped:
                console.print(f"[DEBUG] Line matched 'sshd' but failed to parse: {stripped}", style="dim")
            continue

        if verbose:
            console.print(f"[DEBUG] Parsed record: {record}", style="dim green")

        closed_state, current_state = buffer.add_event(record)

        early_alert = early_engine.check(current_state, record)
        if early_alert:
            emit_alert(early_alert, console, csv_writer, jsonl_fh, fieldnames)

        if closed_state is not None:
            had_early = len(closed_state.early_alert_fired) > 0
            final_alert = final_classify(closed_state, rf_model, iso_model, had_early)
            emit_alert(final_alert, console, csv_writer, jsonl_fh, fieldnames)


# ---------------------------------------------------------------------------
# Thread 2 — Timeout Poller
# ---------------------------------------------------------------------------

def poller_thread(stop_event, caught_up_event, buffer, rf_model, iso_model, console, csv_writer, jsonl_fh, fieldnames):
    """Thread 2: periodically sweeps for expired sessions and classifies them."""
    # Wait until the tailer thread has caught up to the live stream
    caught_up_event.wait()
    
    while not stop_event.is_set():
        time.sleep(POLL_EVERY)
        now_ts = int(time.time())
        for state in buffer.check_expired(now_ts):
            had_early = len(state.early_alert_fired) > 0
            final_alert = final_classify(state, rf_model, iso_model, had_early)
            emit_alert(final_alert, console, csv_writer, jsonl_fh, fieldnames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    
    parser_arg = argparse.ArgumentParser(description="Live SSH Intrusion Detection System — real-time daemon.")
    parser_arg.add_argument("--read-all", action="store_true", help="Read auth.log from the beginning instead of tailing new lines.")
    parser_arg.add_argument("--verbose", action="store_true", help="Print debug/verbose log statements.")
    parser_arg.add_argument("--log-file", type=str, default=AUTH_LOG, help=f"Path to authentication log (default: {AUTH_LOG}).")
    parser_arg.add_argument("--idle-gap", type=int, default=IDLE_GAP, help=f"Session idle gap in seconds (default: {IDLE_GAP}).")
    parser_arg.add_argument("--valid-users", type=str, default="alice", help="Comma-separated list of valid SSH users.")
    args = parser_arg.parse_args()

    console = Console()
    
    valid_users = set(args.valid_users.split(","))
    parser  = SSHLogParser(valid_users=valid_users, year=datetime.now().year)
    buffer  = LiveSessionBuffer(idle_gap=args.idle_gap)
    early_engine = EarlyAlertEngine()

    rf_model  = joblib.load("models/best_model.pkl")
    iso_model = joblib.load("models/anomaly_detector.pkl")

    fieldnames = [
        "alert_kind", "alert_subtype", "alert_type", "ts_alert", "ip",
        "total_attempts", "session_duration", "layer1_class", "layer1_label",
        "layer2_score", "layer2_anomaly", "had_early_alert",
        "attempts_so_far", "failures_so_far", "reason", "event_index",
    ]

    csv_fh = open(OUT_CSV, "w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    csv_writer.writeheader()
    jsonl_fh = open(OUT_JSONL, "w")

    stop_event = threading.Event()
    caught_up_event = threading.Event()

    console.print(f"[bold cyan]Live IDS started — watching {args.log_file}[/bold cyan]")
    console.print(f"[bold cyan]args: read_all={args.read_all} | verbose={args.verbose} | idle_gap={args.idle_gap}s | valid_users={valid_users}[/bold cyan]\n")

    t1 = threading.Thread(target=tailer_thread, args=(
        stop_event, caught_up_event, args.log_file, args.read_all, args.verbose,
        parser, buffer, early_engine, rf_model, iso_model,
        console, csv_writer, jsonl_fh, fieldnames), daemon=True)
    t2 = threading.Thread(target=poller_thread, args=(
        stop_event, caught_up_event, buffer, rf_model, iso_model,
        console, csv_writer, jsonl_fh, fieldnames), daemon=True)

    t1.start()
    t2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Shutting down...[/bold cyan]")
        stop_event.set()
        # Wake up the poller if it's waiting
        caught_up_event.set()

        # Flush remaining open sessions
        for state in buffer.flush():
            had_early = len(state.early_alert_fired) > 0
            final_alert = final_classify(state, rf_model, iso_model, had_early)
            emit_alert(final_alert, console, csv_writer, jsonl_fh, fieldnames)

        csv_fh.close()
        jsonl_fh.close()
        console.print(f"[bold cyan]Saved {OUT_CSV} and {OUT_JSONL}[/bold cyan]")


if __name__ == "__main__":
    main()
