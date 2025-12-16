#!/usr/bin/env python3
"""
Analyze caching experiment results from ablation_results.csv
"""
import pandas as pd
import json
from pathlib import Path

def analyze_results(csv_path):
    """Analyze experiment results and print summary statistics"""
    
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("No results found in the CSV file.")
        return
    
    print(f"\n{'='*80}")
    print(f"CACHING EXPERIMENT ANALYSIS")
    print(f"{'='*80}")
    print(f"Total experiments completed: {len(df)}")
    print(f"Total prompts processed: {df['total_prompts'].sum()}")
    print(f"\n{'='*80}")
    print(f"CACHE HIT RATE ANALYSIS")
    print(f"{'='*80}\n")
    
    # Group by experiment type
    for experiment in df['experiment'].unique():
        exp_df = df[df['experiment'] == experiment]
        
        print(f"\n{experiment.upper()}")
        print(f"{'-'*60}")
        
        for idx, row in exp_df.iterrows():
            model_name = row['model'].split('/')[-1]
            hit_rate = row['cache_hit_rate'] * 100
            hits = row['cache_hits']
            total = row['total_prompts']
            accuracy = row['answer_accuracy'] * 100
            
            # Parse hit breakdown
            hit_breakdown = {}
            if pd.notna(row['hit_breakdown']) and row['hit_breakdown'] != '{}':
                try:
                    hit_breakdown = json.loads(row['hit_breakdown'])
                except:
                    pass
            
            print(f"  Model: {model_name}")
            print(f"  Cache Hits: {hits}/{total} ({hit_rate:.2f}%)")
            
            if hit_breakdown:
                print(f"  Hit Breakdown:")
                for cache_type, count in hit_breakdown.items():
                    print(f"    - {cache_type}: {count} ({count/total*100:.2f}%)")
            
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Avg Latency: {row['avg_latency']*1000:.2f}ms")
            
            if row['cache_hit_rate'] > 0:
                print(f"  Avg Latency (hit): {row['avg_latency_hit']*1000:.2f}ms")
                print(f"  Avg Latency (miss): {row['avg_latency_miss']*1000:.2f}ms")
                speedup = row['avg_latency_miss'] / row['avg_latency_hit'] if row['avg_latency_hit'] > 0 else 0
                print(f"  Cache Speedup: {speedup:.2f}x")
            
            print(f"  Wall Time: {row['wall_time']:.2f}s")
            print()
    
    print(f"\n{'='*80}")
    print(f"SUMMARY BY CACHING TECHNIQUE")
    print(f"{'='*80}\n")
    
    # Categorize experiments
    exact_only = df[df['experiment'].str.contains('exact-only', na=False)]
    fusion_only = df[df['experiment'].str.contains('fusion-only', na=False)]
    semantic = df[df['experiment'].str.contains('semantic-0', na=False)]
    embeddings = df[df['experiment'].str.contains('embed', na=False)]
    
    def print_technique_summary(name, subset):
        if len(subset) == 0:
            return
        
        print(f"\n{name}")
        print(f"{'-'*60}")
        print(f"  Experiments: {len(subset)}")
        print(f"  Avg Hit Rate: {subset['cache_hit_rate'].mean()*100:.2f}%")
        print(f"  Avg Accuracy: {subset['answer_accuracy'].mean()*100:.2f}%")
        print(f"  Avg Latency: {subset['avg_latency'].mean()*1000:.2f}ms")
        
        # Calculate average speedup for experiments with hits
        has_hits = subset[subset['cache_hit_rate'] > 0]
        if len(has_hits) > 0:
            avg_speedup = (has_hits['avg_latency_miss'] / has_hits['avg_latency_hit']).mean()
            print(f"  Avg Cache Speedup: {avg_speedup:.2f}x")
    
    print_technique_summary("EXACT TEXT CACHE ONLY", exact_only)
    print_technique_summary("FUSION CACHE ONLY", fusion_only)
    print_technique_summary("SEMANTIC TEXT CACHE", semantic)
    print_technique_summary("EMBEDDING CACHE", embeddings)
    
    # Model comparison
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON")
    print(f"{'='*80}\n")
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_name = model.split('/')[-1]
        
        print(f"\n{model_name}")
        print(f"{'-'*60}")
        print(f"  Experiments: {len(model_df)}")
        print(f"  Avg Hit Rate: {model_df['cache_hit_rate'].mean()*100:.2f}%")
        print(f"  Avg Accuracy: {model_df['answer_accuracy'].mean()*100:.2f}%")
        print(f"  Avg Latency: {model_df['avg_latency'].mean()*1000:.2f}ms")
        print(f"  Total Wall Time: {model_df['wall_time'].sum():.2f}s")
    
    # Threshold analysis for semantic cache
    semantic_exp = df[df['experiment'].str.contains('semantic-0', na=False)]
    if len(semantic_exp) > 0:
        print(f"\n{'='*80}")
        print(f"SEMANTIC CACHE THRESHOLD ANALYSIS")
        print(f"{'='*80}\n")
        
        for threshold in sorted(semantic_exp['similarity_threshold'].unique()):
            thresh_df = semantic_exp[semantic_exp['similarity_threshold'] == threshold]
            
            print(f"\nThreshold: {threshold}")
            print(f"{'-'*60}")
            print(f"  Experiments: {len(thresh_df)}")
            print(f"  Avg Hit Rate: {thresh_df['cache_hit_rate'].mean()*100:.2f}%")
            print(f"  Avg Accuracy: {thresh_df['answer_accuracy'].mean()*100:.2f}%")
            print(f"  Avg Latency: {thresh_df['avg_latency'].mean()*1000:.2f}ms")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    csv_path = Path(__file__).parent / "experiment_logs" / "ablation_results.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        exit(1)
    
    analyze_results(csv_path)
