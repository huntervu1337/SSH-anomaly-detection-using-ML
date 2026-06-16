import sys
from pathlib import Path
from datetime import datetime

# Add src/ to import path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from log_processing import SSHLogParser

def diagnose():
    print("=" * 60)
    print("DIAGNOSTIC LOG PARSING TEST")
    print("=" * 60)
    
    auth_log = "/var/log/auth.log"
    if not Path(auth_log).exists():
        print(f"Error: {auth_log} does not exist.")
        return
        
    parser = SSHLogParser(valid_users={"alice"}, year=datetime.now().year)
    
    print(f"Reading last 50 lines of {auth_log}...")
    with open(auth_log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    last_lines = lines[-50:]
    sshd_count = 0
    parsed_count = 0
    
    for idx, raw_line in enumerate(last_lines):
        line = raw_line.strip()
        has_sshd = "sshd" in line
        
        if has_sshd:
            sshd_count += 1
            ts = parser._parse_timestamp(line)
            event_type = parser._event_type(line)
            user = parser._parse_user(line)
            ip = parser._parse_ip(line)
            record = parser.parse_line(line)
            
            print(f"\n[Line {idx+1}] {line}")
            print(f"  -> Has sshd: {has_sshd}")
            print(f"  -> Parsed Timestamp: {ts} (date: {datetime.fromtimestamp(ts) if ts else 'FAILED'})")
            print(f"  -> Event Type: {event_type}")
            print(f"  -> Parsed User: {user}")
            print(f"  -> Parsed IP: {ip}")
            print(f"  -> Final Record: {record}")
            if record:
                parsed_count += 1
                
    print("\n" + "=" * 60)
    print(f"Diagnostic Summary: Checked {len(last_lines)} lines.")
    print(f"  - Lines with 'sshd': {sshd_count}")
    print(f"  - Lines successfully parsed: {parsed_count}")
    print("=" * 60)

if __name__ == "__main__":
    diagnose()
