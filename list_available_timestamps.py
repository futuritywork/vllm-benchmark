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
    
    # Look for directories matching the pattern results_gpu_*_timestamp
    pattern = "results_gpu_*_*"
    matching_dirs = glob.glob(pattern)
    
    if not matching_dirs:
        print("❌ No GPU result directories found!")
        return {}
    
    # Group by timestamp
    timestamp_groups = defaultdict(list)
    
    for dir_path in matching_dirs:
        parts = dir_path.split('_')
        if len(parts) >= 4:
            gpu_id = parts[2]
            timestamp = '_'.join(parts[3:])  # Handle timestamps with underscores
            
            timestamp_groups[timestamp].append({
                'gpu_id': gpu_id,
                'directory': dir_path
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
        
        # Check if all expected files exist
        complete_count = 0
        for gpu_dir in gpu_dirs:
            json_file = os.path.join(gpu_dir['directory'], 'engine_conn_results.json')
            if os.path.exists(json_file):
                complete_count += 1
        
        status = "✅" if complete_count == gpu_count else "⚠️"
        print(f"\n{status} {timestamp}")
        print(f"   GPUs: {gpu_count} (Complete: {complete_count})")
        
        # List GPU directories
        for gpu_dir in sorted(gpu_dirs, key=lambda x: int(x['gpu_id'])):
            json_file = os.path.join(gpu_dir['directory'], 'engine_conn_results.json')
            file_status = "✅" if os.path.exists(json_file) else "❌"
            print(f"   {file_status} GPU {gpu_dir['gpu_id']}: {gpu_dir['directory']}")

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
