from __future__ import annotations

import ipaddress
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class SSHLogParser:
	"""Parse SSH auth log lines into structured records.

	The parser keeps per-IP state so downstream labeling can use failure/success
	balance, first-seen status, and time delta information.
	"""

	def __init__(self, valid_users: Optional[Iterable[str]] = None, year: int = 2023):
		self.valid_users = set(valid_users or [])
		self.year = year
		self.ip_state: Dict[str, Dict[str, int]] = {}
		self._current_year = year
		self._last_month: Optional[int] = None
		self._reverse_mapping_ips: set[str] = set()

	def infer_valid_users_from_file(self, input_path: str | Path) -> set[str]:
		"""Learn valid usernames from successful login events in a first pass."""

		input_path = Path(input_path)
		learned_users: set[str] = set(self.valid_users)

		with input_path.open("r", encoding="utf-8", errors="ignore") as handle:
			for line in handle:
				if not line or "sshd" not in line:
					continue
				if self._event_type(line.strip()) != "success":
					continue
				user = self._parse_user(line.strip())
				if user != "-1":
					learned_users.add(user)

		return learned_users

	def _parse_timestamp(self, line: str) -> Optional[int]:
		try:
			parsed = datetime.strptime(line[0:15], "%b %d %H:%M:%S")
			# Detect cross-year rollover (e.g., Dec -> Jan)
			if self._last_month is not None and parsed.month < self._last_month:
				self._current_year += 1
			self._last_month = parsed.month
			parsed = parsed.replace(year=self._current_year)
			return int(parsed.timestamp())
		except Exception:
			return None

	def _parse_ip(self, line: str) -> str:
		match = re.search(r"(\bfrom\s)(\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b)", line)
		if match:
			return match.group(2)
		return "-1"

	def _parse_user(self, line: str) -> str:
		if "Accepted password" in line:
			match = re.search(r"(\bfor\s)(\w+)", line)
			return match.group(2) if match else "-1"

		if "sudo:" in line:
			match = re.search(r"(sudo:\s+)(\w+)", line)
			return match.group(2) if match else "-1"

		if "for invalid user" in line:
			match = re.search(r"(\buser\s)(\w+)", line)
			return match.group(2) if match else "-1"

		if "Invalid user" in line:
			start = line.find("Invalid user ") + len("Invalid user ")
			end = line.find(" from")
			if start >= 0 and end > start:
				return line[start:end]
			return "-1"

		if "Failed password for" in line:
			start = line.find("Failed password for ") + len("Failed password for ")
			end = line.find(" from")
			if start >= 0 and end > start:
				return line[start:end]
			return "-1"

		return "-1"

	def _is_private(self, ip: str) -> int:
		try:
			return int(ipaddress.ip_address(ip).is_private)
		except Exception:
			return 0

	def _is_valid_user(self, user: str) -> int:
		return int(user in self.valid_users)

	def _is_root(self, line: str) -> int:
		return int(bool(re.search(r"\broot\b", line)))

	def _parse_repeated(self, line: str) -> tuple[Optional[str], int]:
		"""Extract inner message and repeat count from 'message repeated N times' lines."""
		match = re.search(r"message repeated (\d+) times:\s*\[\s*(.+?)\s*\]$", line)
		if match:
			return match.group(2).strip(), int(match.group(1))
		return None, 1

	def _event_type(self, line: str) -> Optional[str]:
		if "Accepted password for" in line:
			return "success"
		if (
			"Invalid user" in line
			or "Failed password for" in line
		):
			return "failure"
		return None

	def parse_line(self, line: str) -> Optional[Dict[str, object]]:
		if not line or "sshd" not in line:
			return None

		# Track IPs with DNS reverse mapping failures
		if "POSSIBLE BREAK-IN ATTEMPT" in line:
			ip_match = re.search(r"\[(\d{1,3}(?:\.\d{1,3}){3})\]", line)
			if ip_match:
				self._reverse_mapping_ips.add(ip_match.group(1))

		# Handle "message repeated N times: [...]" lines
		inner_line, repeat_count = self._parse_repeated(line)
		content_line = inner_line if inner_line else line

		event_type = self._event_type(content_line)
		if event_type is None:
			return None

		ts = self._parse_timestamp(line)
		if ts is None:
			return None

		user = self._parse_user(content_line)
		ip = self._parse_ip(content_line)
		if user == "-1" or ip == "-1":
			return None

		is_failure = 1 if event_type == "failure" else 0
		is_private = self._is_private(ip)
		is_root = self._is_root(content_line)
		is_valid = self._is_valid_user(user)

		state = self.ip_state.get(ip)
		if state is None:
			not_valid_count = 0 if is_valid else repeat_count
			ip_failure = repeat_count if is_failure else 0
			ip_success = repeat_count if not is_failure else 0
			no_failure = repeat_count if is_failure else 0
			first = 1
			td = 0
		else:
			not_valid_count = state["not_valid_count"] if is_valid else state["not_valid_count"] + repeat_count
			ip_failure = state["ip_failure"] + (repeat_count if is_failure else 0)
			ip_success = state["ip_success"] + (repeat_count if not is_failure else 0)
			no_failure = (state["no_failure"] + repeat_count) if is_failure else 0
			first = 0
			td = max(0, ts - state["ts"])

		record = {
			"ip": ip,
			"user": user,
			"is_private": is_private,
			"is_failure": is_failure,
			"is_root": is_root,
			"is_valid": is_valid,
			"not_valid_count": not_valid_count,
			"ip_failure": ip_failure,
			"ip_success": ip_success,
			"no_failure": no_failure,
			"first": first,
			"td": td,
			"ts": ts,
			"is_reverse_mapping_failed": 1 if ip in self._reverse_mapping_ips else 0,
		}

		self.ip_state[ip] = {
			"not_valid_count": not_valid_count,
			"ip_failure": ip_failure,
			"ip_success": ip_success,
			"no_failure": no_failure,
			"ts": ts,
		}
		return record

	def parse_file(self, input_path: str | Path, learn_valid_users: bool = True) -> List[Dict[str, object]]:
		input_path = Path(input_path)
		records: List[Dict[str, object]] = []

		if learn_valid_users:
			self.valid_users = self.infer_valid_users_from_file(input_path)

		self.ip_state = {}
		self._current_year = self.year
		self._last_month = None
		self._reverse_mapping_ips = set()

		with input_path.open("r", encoding="utf-8", errors="ignore") as handle:
			for line in handle:
				parsed = self.parse_line(line.strip())
				if parsed is not None:
					records.append(parsed)

		return records
