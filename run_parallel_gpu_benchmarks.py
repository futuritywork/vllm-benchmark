#!/usr/bin/env python3
"""
Parallel GPU Benchmark Runner

This script runs benchmark instances on all available GPUs simultaneously
and aggregates the results at the end.
"""

import asyncio
import json
import os
import sys
import subprocess
import time
from datetime import datetime
from detect_gpus import get_available_gpus
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def run_benchmark_on_gpu(gpu_id, model="Qwen/Qwen3-30B-A3B"):
    """Run benchmark on a specific GPU"""
    
    # Create output directory for this GPU
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_gpu_{gpu_id}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--tensor-parallel-size", "1",
        "--max-concurrency-cap", "1024",
        "--start-concurrency", "2",
        "--max-new-tokens", "500",
        "--trust-remote-code",
        "--log-output",
        "--log-file", f"{output_dir}/llm_outputs.log",
        "--json-out", f"{output_dir}/engine_conn_results.json"
    ]
    
    print(f"🚀 Starting benchmark on GPU {gpu_id} (PID: {os.getpid()})")
    print(f"📁 Output directory: {output_dir}")
    
    # Run the benchmark with CUDA_VISIBLE_DEVICES set
    start_time = time.time()
    try:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ GPU {gpu_id} benchmark completed successfully!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        return {
            'gpu_id': gpu_id,
            'success': True,
            'output_dir': output_dir,
            'duration': duration,
            'error': None
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ GPU {gpu_id} benchmark failed with exit code {e.returncode}")
        return {
            'gpu_id': gpu_id,
            'success': False,
            'output_dir': output_dir,
            'duration': 0,
            'error': f"Exit code {e.returncode}"
        }
    except Exception as e:
        print(f"❌ GPU {gpu_id} benchmark failed with error: {e}")
        return {
            'gpu_id': gpu_id,
            'success': False,
            'output_dir': output_dir,
            'duration': 0,
            'error': str(e)
        }

def aggregate_results(results, model_name):
    """Aggregate results from all GPU benchmarks"""
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    # Load JSON results from successful runs
    aggregated_data = {
        'model': model_name,
        'total_gpus': len(results),
        'successful_gpus': len(successful),
        'failed_gpus': len(failed),
        'start_time': datetime.now().isoformat(),
        'gpu_results': [],
        'summary': {}
    }
    
    max_concurrency_values = []
    avg_tokens_per_second_values = []
    
    for result in results:
        gpu_result = {
            'gpu_id': result['gpu_id'],
            'success': result['success'],
            'output_dir': result['output_dir'],
            'duration': result['duration'],
            'error': result['error']
        }
        
        # Try to load the JSON results file
        json_file = os.path.join(result['output_dir'], 'engine_conn_results.json')
        if os.path.exists(json_file) and result['success']:
            try:
                with open(json_file, 'r') as f:
                    benchmark_data = json.load(f)
                
                gpu_result['benchmark_data'] = benchmark_data
                
                # Extract key metrics
                if 'max_sustainable' in benchmark_data:
                    max_concurrency_values.append(benchmark_data['max_sustainable'])
                    gpu_result['max_sustainable'] = benchmark_data['max_sustainable']
                
                if 'history' in benchmark_data and benchmark_data['history']:
                    # Get the last successful run's tokens per second
                    last_run = benchmark_data['history'][-1]
                    if 'avg_tokens_per_second' in last_run:
                        avg_tokens_per_second_values.append(last_run['avg_tokens_per_second'])
                        gpu_result['avg_tokens_per_second'] = last_run['avg_tokens_per_second']
                
            except Exception as e:
                gpu_result['json_error'] = str(e)
        
        aggregated_data['gpu_results'].append(gpu_result)
    
    # Calculate summary statistics
    if max_concurrency_values:
        aggregated_data['summary']['max_concurrency'] = {
            'mean': sum(max_concurrency_values) / len(max_concurrency_values),
            'min': min(max_concurrency_values),
            'max': max(max_concurrency_values),
            'values': max_concurrency_values
        }
    
    if avg_tokens_per_second_values:
        aggregated_data['summary']['avg_tokens_per_second'] = {
            'mean': sum(avg_tokens_per_second_values) / len(avg_tokens_per_second_values),
            'min': min(avg_tokens_per_second_values),
            'max': max(avg_tokens_per_second_values),
            'values': avg_tokens_per_second_values
        }
    
    return aggregated_data

def print_summary(aggregated_data):
    """Print a formatted summary of the aggregated results"""
    
    print("\n" + "=" * 80)
    print("📊 PARALLEL BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"🤖 Model: {aggregated_data['model']}")
    print(f"🎯 Total GPUs: {aggregated_data['total_gpus']}")
    print(f"✅ Successful: {aggregated_data['successful_gpus']}")
    print(f"❌ Failed: {aggregated_data['failed_gpus']}")
    
    if aggregated_data['summary'].get('max_concurrency'):
        mc = aggregated_data['summary']['max_concurrency']
        print(f"\n📈 Max Sustainable Concurrency:")
        print(f"   Mean: {mc['mean']:.1f}")
        print(f"   Range: {mc['min']:.1f} - {mc['max']:.1f}")
        print(f"   Values: {[f'{v:.1f}' for v in mc['values']]}")
    
    if aggregated_data['summary'].get('avg_tokens_per_second'):
        tps = aggregated_data['summary']['avg_tokens_per_second']
        print(f"\n⚡ Average Tokens per Second:")
        print(f"   Mean: {tps['mean']:.1f}")
        print(f"   Range: {tps['min']:.1f} - {tps['max']:.1f}")
        print(f"   Values: {[f'{v:.1f}' for v in tps['values']]}")
    
    print(f"\n📁 Results directories:")
    for result in aggregated_data['gpu_results']:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} GPU {result['gpu_id']}: {result['output_dir']}")
        if not result['success'] and result['error']:
            print(f"      Error: {result['error']}")

def main():
    """Main function to run parallel benchmarks on all GPUs"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run parallel benchmarks on all available GPUs")
    parser.add_argument(
        "--model", 
        default="Qwen/Qwen3-30B-A3B",
        help="Model to benchmark (default: Qwen/Qwen3-30B-A3B)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: number of GPUs)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run without user confirmation"
    )
    
    args = parser.parse_args()
    
    print("🔍 Detecting available GPUs...")
    gpus = get_available_gpus()
    
    if not gpus:
        print("❌ No GPUs detected!")
        return 1
    
    print(f"✅ Found {len(gpus)} GPU(s): {', '.join(map(str, gpus))}")
    print(f"🤖 Model: {args.model}")
    
    if not args.auto:
        # Ask for confirmation
        response = input(f"\nProceed to run {args.model} benchmark on all {len(gpus)} GPU(s) in parallel? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return 0
    
    # Set number of workers
    max_workers = args.max_workers or len(gpus)
    print(f"🚀 Starting parallel benchmarks with {max_workers} workers...")
    
    # Run benchmarks in parallel
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_gpu = {
            executor.submit(run_benchmark_on_gpu, gpu_id, args.model): gpu_id 
            for gpu_id in gpus
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_gpu):
            gpu_id = future_to_gpu[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ GPU {gpu_id} benchmark failed with exception: {e}")
                results.append({
                    'gpu_id': gpu_id,
                    'success': False,
                    'output_dir': f"results_gpu_{gpu_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'duration': 0,
                    'error': str(e)
                })
    
    total_duration = time.time() - start_time
    
    # Aggregate results
    aggregated_data = aggregate_results(results, args.model)
    aggregated_data['total_duration'] = total_duration
    
    # Print summary
    print_summary(aggregated_data)
    
    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregated_file = f"aggregated_results_{timestamp}.json"
    
    try:
        with open(aggregated_file, 'w') as f:
            json.dump(aggregated_data, f, indent=2)
        print(f"\n💾 Aggregated results saved to: {aggregated_file}")
    except Exception as e:
        print(f"\n⚠️  Failed to save aggregated results: {e}")
    
    # Return success if at least one GPU succeeded
    successful_count = sum(1 for r in results if r['success'])
    return 0 if successful_count > 0 else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
