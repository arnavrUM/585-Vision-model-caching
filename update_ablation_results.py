#!/usr/bin/env python3
"""
Update ablation_results.csv with regraded accuracy from ablation_logs_graded.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def calculate_regraded_accuracy_by_file(graded_logs_dir):
    """
    Calculate accuracy from graded CSV files.
    Returns list of (filename, timestamp_str, accuracy) tuples sorted by timestamp.
    """
    graded_path = Path(graded_logs_dir)
    results = []
    
    for csv_file in sorted(graded_path.glob("*.csv")):
        print(f"Processing {csv_file.name}...")
        
        # Extract timestamp from filename: YYYYMMDD-HHMMSS
        parts = csv_file.stem.split('-')
        if len(parts) >= 2:
            date_str = parts[0]  # YYYYMMDD
            time_str = parts[1]  # HHMMSS
            # Convert to datetime for sorting
            timestamp_dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        else:
            timestamp_dt = datetime.now()
        
        total = 0
        correct = 0
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                match_score = int(row.get('match_score', 0))
                correct += match_score
        
        if total > 0:
            accuracy = correct / total
            results.append((csv_file.name, timestamp_dt, accuracy))
            print(f"  {csv_file.name}: {correct}/{total} = {accuracy:.6f}")
    
    # Sort by timestamp
    results.sort(key=lambda x: x[1])
    return results

def map_results_to_experiments(ablation_results_path, sorted_accuracies):
    """
    Map accuracies to experiment rows by matching in order.
    Assumes both ablation_results.csv and graded logs are sorted by timestamp.
    """
    # Read ablation results (should already be sorted by timestamp)
    with open(ablation_results_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    # Simple 1-to-1 mapping by order
    if len(rows) != len(sorted_accuracies):
        print(f"WARNING: Mismatch in counts - {len(rows)} ablation results vs {len(sorted_accuracies)} graded logs")
        print("Will map as many as possible...")
    
    # Update each row
    updated_rows = []
    for i, row in enumerate(rows):
        if i < len(sorted_accuracies):
            filename, timestamp_dt, new_accuracy = sorted_accuracies[i]
            old_accuracy = float(row['answer_accuracy'])
            row['answer_accuracy'] = f"{new_accuracy:.6f}"
            experiment_name = row['experiment']
            print(f"Row {i+1}: {experiment_name} - {filename}: {old_accuracy:.6f} -> {new_accuracy:.6f}")
        else:
            print(f"Row {i+1}: {row['experiment']} - No matching graded log")
        
        updated_rows.append(row)
    
    return fieldnames, updated_rows

def main():
    graded_logs_dir = "experiment_logs/ablation_logs_graded"
    ablation_results_path = "experiment_logs/ablation_results.csv"
    output_path = "experiment_logs/ablation_results_regraded.csv"
    
    print("Calculating regraded accuracies from graded logs...")
    sorted_accuracies = calculate_regraded_accuracy_by_file(graded_logs_dir)
    
    print(f"\nFound {len(sorted_accuracies)} graded log files")
    
    print("\nMapping accuracies to ablation results (by timestamp order)...")
    fieldnames, updated_rows = map_results_to_experiments(
        ablation_results_path, 
        sorted_accuracies
    )
    
    # Write output
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    
    print(f"\nWrote regraded results to {output_path}")

if __name__ == "__main__":
    main()
