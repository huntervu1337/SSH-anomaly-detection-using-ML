from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDLE_GAP = 600  # seconds (10 min) — gap between events to split sessions

SESSION_OUTPUT_FIELDS = [
	"ip",
	"is_private",
	"total_attempts",
	"session_duration",
	"attempts_per_second",
	"is_single_event",
	"total_failures",
	"total_successes",
	"failure_ratio",
	"unique_users_count",
	"unique_users_ratio",
	"has_root_attempt",
	"has_valid_user_attempt",
	"max_failure_streak",
	"invalid_user_attempts",
	"has_reverse_mapping_failed",
	"ts_first",
	"ts_last",
	"class",
]


# ---------------------------------------------------------------------------
# Session grouping
# ---------------------------------------------------------------------------

def group_into_sessions(
	records: Iterable[Dict[str, object]],
	idle_gap: int = IDLE_GAP,
) -> List[List[Dict[str, object]]]:
	"""Group parsed records into sessions by IP, split by *idle_gap* seconds.

	Records are first grouped by IP, then sorted by timestamp.  A new session
	starts whenever two consecutive events from the same IP are separated by
	more than *idle_gap* seconds.
	"""

	by_ip: Dict[str, List[Dict[str, object]]] = defaultdict(list)
	for record in records:
		by_ip[str(record["ip"])].append(record)

	sessions: List[List[Dict[str, object]]] = []

	for ip, ip_records in by_ip.items():
		ip_records.sort(key=lambda r: int(r["ts"]))

		current_session: List[Dict[str, object]] = [ip_records[0]]
		for prev, cur in zip(ip_records, ip_records[1:]):
			if int(cur["ts"]) - int(prev["ts"]) > idle_gap:
				sessions.append(current_session)
				current_session = [cur]
			else:
				current_session.append(cur)
		sessions.append(current_session)

	return sessions


# ---------------------------------------------------------------------------
# Session-level feature aggregation
# ---------------------------------------------------------------------------

def aggregate_session(session_records: List[Dict[str, object]]) -> Dict[str, object]:
	"""Compute session-level features from a list of per-event records."""

	ip = str(session_records[0]["ip"])
	is_private = max(int(r["is_private"]) for r in session_records)

	# --- Volume & Time ---
	total_attempts = len(session_records)
	ts_first = int(session_records[0]["ts"])
	ts_last = int(session_records[-1]["ts"])
	session_duration = max(0, ts_last - ts_first)
	attempts_per_second = total_attempts / (session_duration + 1)
	is_single_event = 1 if total_attempts == 1 else 0

	# --- Ratios ---
	total_failures = sum(int(r["is_failure"]) for r in session_records)
	total_successes = total_attempts - total_failures
	failure_ratio = total_failures / total_attempts

	users_in_session = set(str(r["user"]) for r in session_records)
	unique_users_count = len(users_in_session)
	unique_users_ratio = unique_users_count / total_attempts

	# --- Boolean aggregation ---
	has_root_attempt = max(int(r["is_root"]) for r in session_records)
	has_valid_user_attempt = max(int(r["is_valid"]) for r in session_records)
	# Per-session count of failure events targeting invalid usernames
	invalid_user_attempts = sum(
		1 for r in session_records
		if int(r["is_valid"]) == 0 and int(r["is_failure"]) == 1
	)
	has_reverse_mapping_failed = max(
		int(r.get("is_reverse_mapping_failed", 0)) for r in session_records
	)

	# Longest consecutive failure streak within this session
	max_failure_streak = 0
	cur_streak = 0
	for r in session_records:
		if int(r["is_failure"]) == 1:
			cur_streak += 1
			max_failure_streak = max(max_failure_streak, cur_streak)
		else:
			cur_streak = 0

	return {
		"ip": ip,
		"is_private": is_private,
		"total_attempts": total_attempts,
		"session_duration": session_duration,
		"attempts_per_second": round(attempts_per_second, 6),
		"is_single_event": is_single_event,
		"total_failures": total_failures,
		"total_successes": total_successes,
		"failure_ratio": round(failure_ratio, 6),
		"unique_users_count": unique_users_count,
		"unique_users_ratio": round(unique_users_ratio, 6),
		"has_root_attempt": has_root_attempt,
		"has_valid_user_attempt": has_valid_user_attempt,
		"max_failure_streak": max_failure_streak,
		"invalid_user_attempts": invalid_user_attempts,
		"has_reverse_mapping_failed": has_reverse_mapping_failed,
		"ts_first": ts_first,
		"ts_last": ts_last,
	}


# ---------------------------------------------------------------------------
# Multi-level labeling
# ---------------------------------------------------------------------------
#   0 = Normal          — all events are successes
#   1 = Single failure  — isolated failure(s), no attack pattern
#   2 = Scan            — probing many different usernames
#   3 = Brute-force     — rapid repeated failures against same/few accounts
#   4 = Break-in        — success AFTER failure(s) in the same session

def _label_record_multilevel(record: Dict[str, object]) -> int:
	"""Assign a per-record severity hint: 0 = success, 1 = failure.

	Brute-force detection is handled at session level to avoid relying on
	the parser's per-IP cumulative ``no_failure`` counter which leaks across
	session boundaries.
	"""
	if int(record["is_failure"]) == 0:
		return 0
	return 1


def label_session(session_records: List[Dict[str, object]]) -> int:
	"""Assign a multi-level label to an entire session.

	Priority (highest wins):
	  4  Break-in   — success with a user that previously failed in session
	  3  Brute-force — >5 consecutive failures within the session
	  2  Scan        — multiple usernames probed (ratio > 50%)
	  1  Single failure — failures present but no strong attack pattern
	  0  Normal      — all events are successes
	"""

	# --- Class 4: Break-in (success with a previously-failed username) ---
	seen_failure_users: set[str] = set()
	for r in session_records:
		user = str(r["user"])
		if int(r["is_failure"]) == 1:
			seen_failure_users.add(user)
		elif user in seen_failure_users:
			return 4  # success with a user that previously failed

	# --- Class 3: Brute-force (long consecutive failure streak) ---
	max_streak = 0
	cur_streak = 0
	for r in session_records:
		if int(r["is_failure"]) == 1:
			cur_streak += 1
			max_streak = max(max_streak, cur_streak)
		else:
			cur_streak = 0
	is_brute_force = max_streak > 5

	# --- Per-record base severity (0 or 1) ---
	has_failures = any(int(r["is_failure"]) == 1 for r in session_records)

	# --- Class 2: Scan (probing multiple usernames) ---
	unique_users = set(str(r["user"]) for r in session_records)
	total = len(session_records)
	unique_ratio = len(unique_users) / total
	is_scan = len(unique_users) >= 2 and unique_ratio > 0.5 and total >= 2 and has_failures

	# --- Resolve final label (highest severity wins) ---
	if is_brute_force and is_scan:
		return 3  # brute-force supersedes scan
	if is_brute_force:
		return 3
	if is_scan:
		return 2
	if has_failures:
		return 1
	return 0


# ---------------------------------------------------------------------------
# Public pipeline API
# ---------------------------------------------------------------------------

def build_session_dataset(
	records: Iterable[Dict[str, object]],
	idle_gap: int = IDLE_GAP,
) -> List[Dict[str, object]]:
	"""Full pipeline: records → sessions → features + labels."""

	sessions = group_into_sessions(records, idle_gap=idle_gap)
	dataset: List[Dict[str, object]] = []

	for session_records in sessions:
		row = aggregate_session(session_records)
		row["class"] = label_session(session_records)
		dataset.append(row)

	return dataset


def write_session_csv(
	dataset: Iterable[Dict[str, object]],
	output_path: str | Path,
) -> Path:
	"""Write session-level dataset to CSV."""

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=SESSION_OUTPUT_FIELDS)
		writer.writeheader()
		for row in dataset:
			writer.writerow(row)

	return output_path


# ---------------------------------------------------------------------------
# Backward-compatible helpers (deprecated — kept for existing notebooks)
# ---------------------------------------------------------------------------

_LEGACY_OUTPUT_FIELDS = [
	"user", "is_private", "is_failure", "is_root", "is_valid",
	"not_valid_count", "ip_failure", "ip_success", "no_failure",
	"first", "td", "ts", "class",
]


def label_record(record: Dict[str, object]) -> Dict[str, object]:
	"""Legacy per-record binary labeling.  Prefer build_session_dataset()."""
	labeled = dict(record)
	labeled["class"] = 0

	is_valid = int(record.get("is_valid", 0))
	is_private = int(record.get("is_private", 0))
	is_failure = int(record.get("is_failure", 0))
	is_root = int(record.get("is_root", 0))
	ip_failure = int(record.get("ip_failure", 0))
	ip_success = int(record.get("ip_success", 0))
	no_failure = int(record.get("no_failure", 0))
	td = int(record.get("td", 0))
	first = int(record.get("first", 0))

	if is_valid == 0 and is_private == 0 and is_failure == 1:
		labeled["class"] = 1
	if is_root == 1 and is_failure == 1 and is_private == 0:
		labeled["class"] = 1
	if ip_failure > 3 and ip_failure >= (ip_success * 2):
		labeled["class"] = 1
	if no_failure > 3 and td < 10 and first == 0:
		labeled["class"] = 1

	return labeled


def label_records(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
	"""Legacy batch labeling.  Prefer build_session_dataset()."""
	return [label_record(record) for record in records]


def write_labeled_csv(records: Iterable[Dict[str, object]], output_path: str | Path) -> Path:
	"""Legacy CSV writer.  Prefer write_session_csv()."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=_LEGACY_OUTPUT_FIELDS)
		writer.writeheader()
		for record in records:
			writer.writerow(record)
	return output_path
