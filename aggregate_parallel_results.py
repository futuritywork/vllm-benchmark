#!/usr/bin/env python3
"""
Parallel Benchmark Results Aggregator

This script reads all results from a parallel benchmark run (gpu_0, gpu_1, ...)
for a specific timestamp and aggregates all the engine_conn_results together.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import glob

def find_gpu_result_dirs(timestamp):
    """Find all GPU result directories for a specific timestamp"""
    
    # Look for the base directory: results_timestamp/
    base_dir = f"results_{timestamp}"
    
    if not os.path.exists(base_dir):
        print(f"❌ Base directory not found: {base_dir}")
        return []
    
    # Look for GPU subdirectories: results_timestamp/gpu_0/, gpu_1/, etc.
    gpu_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('gpu_'):
            gpu_dirs.append(item_path)
    
    if not gpu_dirs:
        print(f"❌ No GPU result directories found in {base_dir}")
        return []
    
    # Sort by GPU ID for consistent ordering
    gpu_dirs.sort(key=lambda x: int(x.split('_')[-1]))
    
    print(f"✅ Found {len(gpu_dirs)} GPU result directories in {base_dir}:")
    for dir_path in gpu_dirs:
        print(f"   {dir_path}")
    
    return gpu_dirs

def load_gpu_results(gpu_dirs):
    """Load results from all GPU directories"""
    
    all_results = []
    
    for dir_path in gpu_dirs:
        # Extract GPU ID from directory name (e.g., "gpu_0" -> "0")
        gpu_id = dir_path.split('_')[-1]  # Get the last part after splitting by '_'
        
        # Look for the engine_conn_results.json file
        json_file = os.path.join(dir_path, 'engine_conn_results.json')
        
        if not os.path.exists(json_file):
            print(f"⚠️  Warning: No engine_conn_results.json found in {dir_path}")
            all_results.append({
                'gpu_id': gpu_id,
                'success': False,
                'error': 'Missing engine_conn_results.json',
                'directory': dir_path
            })
            continue
        
        try:
            with open(json_file, 'r') as f:
                benchmark_data = json.load(f)
            
            all_results.append({
                'gpu_id': gpu_id,
                'success': True,
                'directory': dir_path,
                'benchmark_data': benchmark_data
            })
            
            print(f"✅ Loaded results from GPU {gpu_id}")
            
        except Exception as e:
            print(f"❌ Error loading results from GPU {gpu_id}: {e}")
            all_results.append({
                'gpu_id': gpu_id,
                'success': False,
                'error': str(e),
                'directory': dir_path
            })
    
    return all_results

def aggregate_results(gpu_results, timestamp):
    """Aggregate results from all GPU benchmarks"""
    
    successful = [r for r in gpu_results if r['success']]
    failed = [r for r in gpu_results if not r['success']]
    
    # Initialize aggregated data
    aggregated_data = {
        'timestamp': timestamp,
        'aggregation_time': datetime.now().isoformat(),
        'total_gpus': len(gpu_results),
        'successful_gpus': len(successful),
        'failed_gpus': len(failed),
        'gpu_results': gpu_results,
        'summary': {},
        'detailed_metrics': {}
    }
    
    # Extract model name from first successful result
    if successful:
        first_result = successful[0]['benchmark_data']
        if 'model' in first_result:
            aggregated_data['model'] = first_result['model']
    
    # Collect metrics from successful runs
    max_concurrency_values = []
    avg_tokens_per_second_values = []
    p50_tokens_per_second_values = []
    p95_tokens_per_second_values = []
    success_rates = []
    ttft_values = []
    
    # Detailed metrics for each concurrency level
    concurrency_metrics = {}
    
    for result in successful:
        gpu_id = result['gpu_id']
        data = result['benchmark_data']
        
        # Extract max sustainable concurrency
        if 'max_sustainable' in data:
            max_concurrency_values.append(data['max_sustainable'])
        
        # Extract metrics from history
        if 'history' in data and data['history']:
            for run in data['history']:
                concurrency = run.get('concurrency', 0)
                
                if concurrency not in concurrency_metrics:
                    concurrency_metrics[concurrency] = {
                        'gpu_results': {},
                        'success_rates': [],
                        'avg_tokens_per_second': [],
                        'p50_tokens_per_second': [],
                        'p95_tokens_per_second': []
                    }
                
                # Store GPU-specific results for this concurrency level
                concurrency_metrics[concurrency]['gpu_results'][gpu_id] = {
                    'ok_rate': run.get('ok_rate', 0),
                    'avg_tokens_per_second': run.get('avg_tokens_per_second', 0),
                    'tokens_per_second_p50': run.get('tokens_per_second_p50', 0),
                    'tokens_per_second_p95': run.get('tokens_per_second_p95', 0)
                }
                
                # Collect metrics for aggregation
                if 'ok_rate' in run:
                    concurrency_metrics[concurrency]['success_rates'].append(run['ok_rate'])
                if 'avg_tokens_per_second' in run:
                    concurrency_metrics[concurrency]['avg_tokens_per_second'].append(run['avg_tokens_per_second'])
                if 'tokens_per_second_p50' in run:
                    concurrency_metrics[concurrency]['p50_tokens_per_second'].append(run['tokens_per_second_p50'])
                if 'tokens_per_second_p95' in run:
                    concurrency_metrics[concurrency]['p95_tokens_per_second'].append(run['tokens_per_second_p95'])
            
            # Get metrics from the last successful run (max sustainable concurrency)
            last_run = data['history'][-1]
            if 'avg_tokens_per_second' in last_run:
                avg_tokens_per_second_values.append(last_run['avg_tokens_per_second'])
            if 'tokens_per_second_p50' in last_run:
                p50_tokens_per_second_values.append(last_run['tokens_per_second_p50'])
            if 'tokens_per_second_p95' in last_run:
                p95_tokens_per_second_values.append(last_run['tokens_per_second_p95'])
            if 'ok_rate' in last_run:
                success_rates.append(last_run['ok_rate'])
    
    # Calculate summary statistics
    if max_concurrency_values:
        aggregated_data['summary']['max_concurrency'] = {
            'mean': sum(max_concurrency_values) / len(max_concurrency_values),
            'min': min(max_concurrency_values),
            'max': max(max_concurrency_values),
            'std': calculate_std(max_concurrency_values),
            'values': max_concurrency_values
        }
    
    if avg_tokens_per_second_values:
        aggregated_data['summary']['avg_tokens_per_second'] = {
            'mean': sum(avg_tokens_per_second_values) / len(avg_tokens_per_second_values),
            'min': min(avg_tokens_per_second_values),
            'max': max(avg_tokens_per_second_values),
            'std': calculate_std(avg_tokens_per_second_values),
            'values': avg_tokens_per_second_values
        }
    
    if p50_tokens_per_second_values:
        aggregated_data['summary']['p50_tokens_per_second'] = {
            'mean': sum(p50_tokens_per_second_values) / len(p50_tokens_per_second_values),
            'min': min(p50_tokens_per_second_values),
            'max': max(p50_tokens_per_second_values),
            'std': calculate_std(p50_tokens_per_second_values),
            'values': p50_tokens_per_second_values
        }
    
    if p95_tokens_per_second_values:
        aggregated_data['summary']['p95_tokens_per_second'] = {
            'mean': sum(p95_tokens_per_second_values) / len(p95_tokens_per_second_values),
            'min': min(p95_tokens_per_second_values),
            'max': max(p95_tokens_per_second_values),
            'std': calculate_std(p95_tokens_per_second_values),
            'values': p95_tokens_per_second_values
        }
    
    if success_rates:
        aggregated_data['summary']['success_rate'] = {
            'mean': sum(success_rates) / len(success_rates),
            'min': min(success_rates),
            'max': max(success_rates),
            'std': calculate_std(success_rates),
            'values': success_rates
        }
    
    # Add detailed concurrency metrics
    aggregated_data['detailed_metrics'] = concurrency_metrics
    
    return aggregated_data

def calculate_std(values):
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5

def print_summary(aggregated_data):
    """Print a formatted summary of the aggregated results"""
    
    print("\n" + "=" * 80)
    print("📊 PARALLEL BENCHMARK RESULTS AGGREGATION")
    print("=" * 80)
    
    print(f"🕐 Timestamp: {aggregated_data['timestamp']}")
    if 'model' in aggregated_data:
        print(f"🤖 Model: {aggregated_data['model']}")
    print(f"🎯 Total GPUs: {aggregated_data['total_gpus']}")
    print(f"✅ Successful: {aggregated_data['successful_gpus']}")
    print(f"❌ Failed: {aggregated_data['failed_gpus']}")
    
    # Print summary statistics
    summary = aggregated_data['summary']
    
    if 'max_concurrency' in summary:
        mc = summary['max_concurrency']
        print(f"\n📈 Max Sustainable Concurrency:")
        print(f"   Mean ± Std: {mc['mean']:.1f} ± {mc['std']:.1f}")
        print(f"   Range: {mc['min']:.1f} - {mc['max']:.1f}")
        print(f"   Values: {[f'{v:.1f}' for v in mc['values']]}")
    
    if 'avg_tokens_per_second' in summary:
        tps = summary['avg_tokens_per_second']
        print(f"\n⚡ Average Tokens per Second:")
        print(f"   Mean ± Std: {tps['mean']:.1f} ± {tps['std']:.1f}")
        print(f"   Range: {tps['min']:.1f} - {tps['max']:.1f}")
        print(f"   Values: {[f'{v:.1f}' for v in tps['values']]}")
    
    if 'p50_tokens_per_second' in summary:
        p50 = summary['p50_tokens_per_second']
        print(f"\n📊 P50 Tokens per Second:")
        print(f"   Mean ± Std: {p50['mean']:.1f} ± {p50['std']:.1f}")
        print(f"   Range: {p50['min']:.1f} - {p50['max']:.1f}")
    
    if 'p95_tokens_per_second' in summary:
        p95 = summary['p95_tokens_per_second']
        print(f"\n📊 P95 Tokens per Second:")
        print(f"   Mean ± Std: {p95['mean']:.1f} ± {p95['std']:.1f}")
        print(f"   Range: {p95['min']:.1f} - {p95['max']:.1f}")
    
    if 'success_rate' in summary:
        sr = summary['success_rate']
        print(f"\n🎯 Success Rate:")
        print(f"   Mean ± Std: {sr['mean']:.3f} ± {sr['std']:.3f}")
        print(f"   Range: {sr['min']:.3f} - {sr['max']:.3f}")
    
    # Print GPU results
    print(f"\n📁 GPU Results:")
    for result in aggregated_data['gpu_results']:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} GPU {result['gpu_id']}: {result['directory']}")
        if not result['success'] and 'error' in result:
            print(f"      Error: {result['error']}")

def main():
    """Main function to aggregate parallel benchmark results"""
    
    parser = argparse.ArgumentParser(description="Aggregate results from parallel benchmark run")
    parser.add_argument(
        "timestamp",
        help="Timestamp to aggregate (e.g., 20241201_143022)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for aggregated results (default: aggregated_results_timestamp.json)"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print summary, don't save to file"
    )
    
    args = parser.parse_args()
    
    print(f"🔍 Looking for GPU result directories with timestamp: {args.timestamp}")
    
    # Find GPU result directories
    gpu_dirs = find_gpu_result_dirs(args.timestamp)
    
    if not gpu_dirs:
        print("❌ No GPU result directories found!")
        return 1
    
    # Load results from all GPUs
    print(f"\n📂 Loading results from {len(gpu_dirs)} GPU directories...")
    gpu_results = load_gpu_results(gpu_dirs)
    
    # Aggregate results
    print(f"\n🔧 Aggregating results...")
    aggregated_data = aggregate_results(gpu_results, args.timestamp)
    
    # Print summary
    print_summary(aggregated_data)
    
    # Save aggregated results
    if not args.print_only:
        if args.output:
            output_file = args.output
        else:
            # Save in the base directory
            base_dir = f"results_{args.timestamp}"
            output_file = os.path.join(base_dir, "aggregated_results.json")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(aggregated_data, f, indent=2)
            print(f"\n💾 Aggregated results saved to: {output_file}")
        except Exception as e:
            print(f"\n⚠️  Failed to save aggregated results: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
