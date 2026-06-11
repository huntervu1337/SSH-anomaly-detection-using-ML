"""Real-time SSH anomaly detection — streaming simulation.

Implements the hybrid detection pipeline from ``realtime_detection_spec.md``:

  Log stream  →  SessionBuffer  →  EarlyAlertEngine  (rule-based, per-event)
                      ↓ (session closed)
                 _final_classify  →  Layer 1 (RF)  +  Layer 2 (IF)

Usage::

    from realtime_simulator import run_simulation
    alerts = run_simulation(
        "data/raw/SSH.log",
        rf_model=joblib.load("models/best_model.pkl"),
        iso_model=joblib.load("models/anomaly_detector.pkl"),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_labeling import aggregate_session, IDLE_GAP
from feature_engineering import prepare_features, CLASS_NAMES
from log_processing import SSHLogParser
from anomaly_detector import score_session


# ---------------------------------------------------------------------------
# SessionBuffer — in-memory state tracking
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Mutable state of an active (open) session."""

    ip: str
    records: List[dict] = field(default_factory=list)
    ts_first: int = 0
    ts_last: int = 0
    total_attempts: int = 0
    total_failures: int = 0
    max_failure_streak: int = 0
    cur_failure_streak: int = 0
    unique_users: set = field(default_factory=set)
    has_root_attempt: int = 0
    has_valid_user_attempt: int = 0
    early_alert_fired: set = field(default_factory=set)


class SessionBuffer:
    """In-memory state tracking for active sessions.

    A session closes when:
    - A new event from the same IP has gap > ``idle_gap`` seconds
    - ``flush()`` is called (end of stream)
    """

    def __init__(self, idle_gap: int = IDLE_GAP):
        self.idle_gap = idle_gap
        self.sessions: Dict[str, SessionState] = {}

    def add_event(
        self, record: dict
    ) -> Tuple[Optional[SessionState], SessionState]:
        """Process one parsed event.

        Steps:
            1. State check — does this IP have an open session?
            2. Gap evaluation — is the gap large enough to close?
            3. Incremental update — update session state

        Returns
        -------
        closed_state : SessionState or None
            The state of the session that was just closed (if gap triggered),
            including its records. ``None`` if no session was closed.
        current_state : SessionState
            Updated state of the current (open) session.
        """
        ip = str(record["ip"])
        ts = int(record["ts"])

        # Step 1 & 2: Gap evaluation
        closed_state = None
        if ip in self.sessions:
            state = self.sessions[ip]
            if ts - state.ts_last > self.idle_gap:
                # Gap too large → close old session
                closed_state = state
                del self.sessions[ip]

        # Step 3: Init or update
        if ip not in self.sessions:
            self.sessions[ip] = SessionState(ip=ip, ts_first=ts)

        state = self.sessions[ip]
        state.records.append(record)
        state.ts_last = ts
        state.total_attempts += 1

        if int(record["is_failure"]) == 1:
            state.total_failures += 1
            state.cur_failure_streak += 1
            state.max_failure_streak = max(
                state.max_failure_streak, state.cur_failure_streak
            )
        else:
            state.cur_failure_streak = 0

        state.unique_users.add(str(record["user"]))
        state.has_root_attempt = max(
            state.has_root_attempt, int(record["is_root"])
        )
        state.has_valid_user_attempt = max(
            state.has_valid_user_attempt, int(record["is_valid"])
        )

        return closed_state, state

    def flush(self) -> List[SessionState]:
        """Close all open sessions (end of stream)."""
        remaining = list(self.sessions.values())
        self.sessions.clear()
        return remaining


# ---------------------------------------------------------------------------
# EarlyAlertEngine — rule-based, fires per-event
# ---------------------------------------------------------------------------

class EarlyAlertEngine:
    """Rule-based early alerting — fires once per session per alert type.

    Each alert subtype fires at most once per session (tracked via
    ``state.early_alert_fired``).
    """

    BRUTE_FORCE_STREAK = 6    # consecutive failures to trigger
    SCAN_UNIQUE_USERS = 4     # unique usernames to trigger
    BREAK_IN_RISK = True      # enable break-in detection

    def check(
        self, state: SessionState, record: dict
    ) -> Optional[dict]:
        """Check early alert conditions after an incremental update.

        Returns an alert dict if triggered, else None.
        """

        # Trigger 1: Brute-force in progress
        if (
            "BRUTE_FORCE" not in state.early_alert_fired
            and state.max_failure_streak >= self.BRUTE_FORCE_STREAK
        ):
            state.early_alert_fired.add("BRUTE_FORCE")
            return self._make_alert(
                state, record,
                alert_subtype="BRUTE_FORCE",
                reason=(
                    f"max_failure_streak={state.max_failure_streak} "
                    f">= {self.BRUTE_FORCE_STREAK}"
                ),
            )

        # Trigger 2: Scan in progress
        if (
            "SCAN" not in state.early_alert_fired
            and len(state.unique_users) >= self.SCAN_UNIQUE_USERS
            and state.total_failures > 0
        ):
            state.early_alert_fired.add("SCAN")
            return self._make_alert(
                state, record,
                alert_subtype="SCAN",
                reason=(
                    f"unique_users={len(state.unique_users)} "
                    f">= {self.SCAN_UNIQUE_USERS}"
                ),
            )

        # Trigger 3: Break-in risk
        if (
            "BREAK_IN_RISK" not in state.early_alert_fired
            and int(record["is_failure"]) == 0       # just succeeded
            and int(record["is_valid"]) == 1          # with valid user
            and state.total_failures > 0              # had failures before
        ):
            state.early_alert_fired.add("BREAK_IN_RISK")
            return self._make_alert(
                state, record,
                alert_subtype="BREAK_IN_RISK",
                reason="valid user succeeded after previous failures",
            )

        return None

    @staticmethod
    def _make_alert(
        state: SessionState, record: dict,
        alert_subtype: str, reason: str,
    ) -> dict:
        return {
            "alert_kind":      "EARLY_ALERT",
            "alert_subtype":   alert_subtype,
            "ts_alert":        int(record["ts"]),
            "ip":              state.ip,
            "ts_first":        state.ts_first,
            "attempts_so_far": state.total_attempts,
            "failures_so_far": state.total_failures,
            "reason":          reason,
            "event_index":     state.total_attempts,
        }


# ---------------------------------------------------------------------------
# Final classification — runs when session closes
# ---------------------------------------------------------------------------

def _final_classify(
    closed_state: SessionState,
    rf_model,
    iso_model,
    had_early: bool,
) -> dict:
    """Classify a closed session through Layer 1 + Layer 2.

    Parameters
    ----------
    closed_state : SessionState
        The state (with records) of the session that just closed.
    rf_model : trained RandomForest
    iso_model : trained IsolationForest
    had_early : bool
        Whether this session received an EARLY_ALERT during its lifetime.
    """
    row = aggregate_session(closed_state.records)
    df = pd.DataFrame([row])
    # Assign dummy target for prepare_features (it needs 'class' column)
    df["class"] = 0
    X_row, _ = prepare_features(df)

    l1_class = int(rf_model.predict(X_row)[0])
    l2_result = score_session(iso_model, X_row)

    alert_type = (
        "KNOWN_ATTACK"    if l1_class > 0 else
        "UNKNOWN_PATTERN" if l2_result["is_anomaly"] else
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
        "layer2_score":     round(l2_result["anomaly_score"], 4),
        "layer2_anomaly":   l2_result["is_anomaly"],
        "had_early_alert":  int(had_early),
    }


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    log_path: str,
    rf_model,
    iso_model,
    idle_gap: int = IDLE_GAP,
    verbose: bool = True,
) -> List[dict]:
    """Run streaming detection simulation on a log file.

    Parameters
    ----------
    log_path : str
        Path to raw SSH log file.
    rf_model : trained RandomForest (Layer 1)
    iso_model : trained IsolationForest (Layer 2)
    idle_gap : int
        Session gap in seconds (default: 600).
    verbose : bool
        Print progress and summary.

    Returns
    -------
    list of dict
        All alerts (EARLY_ALERT + FINAL_CLASSIFICATION) in chronological order.
    """
    # --- Setup ---
    parser = SSHLogParser()
    parser.valid_users = parser.infer_valid_users_from_file(log_path)
    parser.ip_state = {}
    parser._current_year = parser.year
    parser._last_month = None
    parser._reverse_mapping_ips = set()

    buffer = SessionBuffer(idle_gap=idle_gap)
    early_engine = EarlyAlertEngine()
    alerts: List[dict] = []
    ips_with_early_alert: set[tuple] = set()

    lines_processed = 0
    events_processed = 0

    # --- Stream ---
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines_processed += 1
            record = parser.parse_line(line.strip())
            if record is None:
                continue

            events_processed += 1

            # Step 1-3: State check + gap eval + incremental update
            closed_state, current_state = buffer.add_event(record)

            # Step 4: Early alert check
            early_alert = early_engine.check(current_state, record)
            if early_alert:
                alerts.append(early_alert)
                ips_with_early_alert.add((current_state.ip, current_state.ts_first))

            # Final classification when session closes
            if closed_state is not None:
                had_early = (closed_state.ip, closed_state.ts_first) in ips_with_early_alert
                final_alert = _final_classify(
                    closed_state, rf_model, iso_model, had_early
                )
                alerts.append(final_alert)

    # --- Flush remaining sessions ---
    for remaining_state in buffer.flush():
        had_early = (remaining_state.ip, remaining_state.ts_first) in ips_with_early_alert
        final_alert = _final_classify(
            remaining_state, rf_model, iso_model, had_early
        )
        alerts.append(final_alert)

    if verbose:
        n_early = sum(1 for a in alerts if a["alert_kind"] == "EARLY_ALERT")
        n_final = sum(1 for a in alerts if a["alert_kind"] == "FINAL_CLASSIFICATION")
        print(f"Simulation complete:")
        print(f"  Lines processed: {lines_processed:,}")
        print(f"  Events parsed:   {events_processed:,}")
        print(f"  Alerts total:    {len(alerts)}")
        print(f"    EARLY_ALERT:   {n_early}")
        print(f"    FINAL:         {n_final}")

    return alerts
