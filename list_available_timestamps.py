#!/usr/bin/env python3
"""
List Available Timestamps

This script lists all available timestamps from parallel benchmark runs
to help users identify which results to aggregate.
"""

import os
import sys
import glob
from collections import defaultdict

def find_available_timestamps():
    """Find all available timestamps from GPU result directories"""
    
    # Look for base directories matching the pattern results_timestamp
    pattern = "results_*"
    matching_dirs = glob.glob(pattern)
    
    if not matching_dirs:
        print("❌ No GPU result directories found!")
        return {}
    
    # Group by timestamp
    timestamp_groups = defaultdict(list)
    
    for base_dir in matching_dirs:
        if not os.path.isdir(base_dir):
            continue
            
        # Extract timestamp from base directory name
        timestamp = base_dir.replace('results_', '')
        
        # Look for GPU subdirectories
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith('gpu_'):
                gpu_id = item.split('_')[-1]  # Extract GPU ID
                
                timestamp_groups[timestamp].append({
                    'gpu_id': gpu_id,
                    'directory': item_path,
                    'base_directory': base_dir
                })
    
    return timestamp_groups

def print_timestamp_summary(timestamp_groups):
    """Print a summary of available timestamps"""
    
    if not timestamp_groups:
        print("No timestamps found!")
        return
    
    print("📅 Available Parallel Benchmark Timestamps")
    print("=" * 60)
    
    for timestamp in sorted(timestamp_groups.keys(), reverse=True):
        gpu_dirs = timestamp_groups[timestamp]
        gpu_count = len(gpu_dirs)
        base_dir = gpu_dirs[0]['base_directory'] if gpu_dirs else f"results_{timestamp}"
        
        # Check if all expected files exist
        complete_count = 0
        for gpu_dir in gpu_dirs:
            json_file = os.path.join(gpu_dir['directory'], 'engine_conn_results.json')
            if os.path.exists(json_file):
                complete_count += 1
        
        status = "✅" if complete_count == gpu_count else "⚠️"
        print(f"\n{status} {timestamp}")
        print(f"   Base Directory: {base_dir}")
        print(f"   GPUs: {gpu_count} (Complete: {complete_count})")
        
        # List GPU directories
        for gpu_dir in sorted(gpu_dirs, key=lambda x: int(x['gpu_id'])):
            json_file = os.path.join(gpu_dir['directory'], 'engine_conn_results.json')
            file_status = "✅" if os.path.exists(json_file) else "❌"
            print(f"   {file_status} GPU {gpu_dir['gpu_id']}: {gpu_dir['directory']}")
        
        # Check for aggregated results
        aggregated_file = os.path.join(base_dir, "aggregated_results.json")
        if os.path.exists(aggregated_file):
            print(f"   📊 Aggregated results: {aggregated_file}")
        else:
            print(f"   📊 Aggregated results: Not found")

def main():
    """Main function to list available timestamps"""
    
    print("🔍 Scanning for parallel benchmark results...")
    
    timestamp_groups = find_available_timestamps()
    
    if not timestamp_groups:
        print("❌ No parallel benchmark results found!")
        print("\nMake sure you have run parallel benchmarks first:")
        print("  make qwen30-parallel")
        print("  make qwen30-fp8-parallel")
        return 1
    
    print_timestamp_summary(timestamp_groups)
    
    print(f"\n💡 To aggregate results for a specific timestamp, use:")
    print(f"  python aggregate_parallel_results.py TIMESTAMP")
    print(f"  make aggregate-results TIMESTAMP=TIMESTAMP")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
