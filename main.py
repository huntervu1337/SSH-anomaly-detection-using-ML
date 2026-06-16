#!/usr/bin/env python
"""
SSH Anomaly Detection - Complete Model Training Pipeline

Reconstructs the full training pipeline from raw logs to serialized models:
1. Parses SSH logs and extracts events.
2. Groups events into sessions and assigns labels based on heuristic rules.
3. Preprocesses features (log1p transform, edge cases) and exports split sets.
4. Trains the Random Forest Classifier on all sessions.
5. Trains the Isolation Forest Anomaly Detector on Normal sessions.
6. Serializes the models to `models/` directory for live_ids.py usage.

Usage:
  python main.py           # Runs on the full data/raw/SSH.log (10-15 seconds)
  python main.py --quick   # Runs on a subset data/raw/SSH_2k.log (under 1 second)
"""

import sys
import argparse
from pathlib import Path
import time
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Ensure src/ is in the import search path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "src"))

from log_processing import SSHLogParser
from data_labeling import build_session_dataset, write_session_csv
from feature_engineering import load_session_csv, prepare_features
from anomaly_detector import train_anomaly_detector


def run_pipeline(quick: bool = False):
    print("=" * 60)
    print("SSH ANOMALY DETECTION - PIPELINE RUN")
    print("=" * 60)

    # 1. Define and verify paths
    raw_dir = ROOT_DIR / "data" / "raw"
    processed_dir = ROOT_DIR / "data" / "processed"
    models_dir = ROOT_DIR / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    log_filename = "SSH_2k.log" if quick else "SSH.log"
    log_path = raw_dir / log_filename

    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        print("Please ensure you have placed the log file in data/raw/")
        sys.exit(1)

    print(f"Using log file: {log_path} (Size: {log_path.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Parse Raw Logs
    print("\n--- Step 1: Parsing Raw Logs ---")
    start_time = time.time()
    parser = SSHLogParser(year=2023)
    
    print("Inferring valid users and parsing log file...")
    records = parser.parse_file(log_path, learn_valid_users=True)
    parse_duration = time.time() - start_time
    
    print(f"Successfully parsed {len(records):,} event records in {parse_duration:.2f} seconds.")
    print(f"Learned valid users: {parser.valid_users}")

    # 3. Sessionization and Label Heuristics
    print("\n--- Step 2: Session Grouping and Labeling ---")
    start_time = time.time()
    dataset = build_session_dataset(records)
    session_duration = time.time() - start_time
    
    session_csv_path = processed_dir / "ssh_sessions.csv"
    write_session_csv(dataset, session_csv_path)
    
    print(f"Grouped events into {len(dataset):,} sessions in {session_duration:.2f} seconds.")
    print(f"Saved session dataset to: {session_csv_path}")

    # 4. Feature Engineering and Preprocessing
    print("\n--- Step 3: Feature Engineering and Split Generation ---")
    df = load_session_csv(session_csv_path)
    X, y = prepare_features(df)
    
    # Export full training ready sets
    X_train_ready_path = processed_dir / "X_train_ready.csv"
    y_train_ready_path = processed_dir / "y_train_ready.csv"
    X.to_csv(X_train_ready_path, index=False)
    y.to_csv(y_train_ready_path, index=False, header=True)
    
    # Perform time-based split (75% train, 25% test cutoff chronologically)
    ts_first = df["ts_first"]
    ts_min = ts_first.min()
    ts_max = ts_first.max()
    ts_range = ts_max - ts_min
    ts_cutoff = ts_min + ts_range * 0.75
    
    train_mask = ts_first <= ts_cutoff
    test_mask = ts_first > ts_cutoff
    
    X_time_train = X[train_mask]
    y_time_train = y[train_mask]
    X_time_test = X[test_mask]
    y_time_test = y[test_mask]
    
    # Save the splits to CSV
    X_time_train.to_csv(processed_dir / "X_time_train.csv", index=False)
    y_time_train.to_csv(processed_dir / "y_time_train.csv", index=False, header=True)
    X_time_test.to_csv(processed_dir / "X_time_test.csv", index=False)
    y_time_test.to_csv(processed_dir / "y_time_test.csv", index=False, header=True)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Time-based splits generated: Train={X_time_train.shape[0]} sessions, Test={X_time_test.shape[0]} sessions")
    print(f"All processed CSV datasets successfully saved to {processed_dir}")

    # 5. Train Random Forest (Layer 1)
    print("\n--- Step 4: Training Random Forest (Layer 1) ---")
    start_time = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    # Refit on the full dataset for deployment
    rf_model.fit(X, y)
    rf_duration = time.time() - start_time
    
    rf_model_path = models_dir / "best_model.pkl"
    joblib.dump(rf_model, rf_model_path)
    print(f"Random Forest Classifier trained in {rf_duration:.2f} seconds.")
    print(f"Model saved to: {rf_model_path}")

    # 6. Train Isolation Forest Anomaly Detector (Layer 2)
    print("\n--- Step 5: Training Isolation Forest (Layer 2) ---")
    start_time = time.time()
    
    # Train on normal sessions (class == 0)
    best_contamination = 0.02
    iso_model = train_anomaly_detector(
        X, y,
        contamination=best_contamination,
        random_state=42
    )
    iso_duration = time.time() - start_time
    
    iso_model_path = models_dir / "anomaly_detector.pkl"
    joblib.dump(iso_model, iso_model_path)
    
    n_normal = (y == 0).sum()
    print(f"Isolation Forest trained on {n_normal} Normal sessions in {iso_duration:.2f} seconds.")
    print(f"Model saved to: {iso_model_path}")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("You can now run the real-time intrusion detection daemon using:")
    print("  sudo .venv/bin/python src/live_ids.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSH Anomaly Detection Training Pipeline")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run pipeline on the smaller 2,000 line sample log for quick test verification"
    )
    args = parser.parse_args()
    run_pipeline(quick=args.quick)
