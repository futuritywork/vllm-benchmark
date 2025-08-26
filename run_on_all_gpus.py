#!/usr/bin/env python3
"""
Script to run Qwen 3.0 benchmark on all available GPUs with tensor parallel size of 1.
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

def run_benchmark_on_gpu(gpu_id, model="Qwen/Qwen3-30B-A3B"):
    """Run benchmark on a specific GPU"""
    
    # Create output directory for this GPU
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_gpu_{gpu_id}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command (without --gpu-device, will use CUDA_VISIBLE_DEVICES)
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
    
    print(f"\n🚀 Starting benchmark on GPU {gpu_id}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 Command: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the benchmark with CUDA_VISIBLE_DEVICES set
    start_time = time.time()
    try:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ GPU {gpu_id} benchmark completed successfully!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📁 Results saved to: {output_dir}/")
        
        return True, output_dir, duration
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ GPU {gpu_id} benchmark failed with exit code {e.returncode}")
        return False, output_dir, 0
    except KeyboardInterrupt:
        print(f"\n⚠️  GPU {gpu_id} benchmark interrupted by user")
        return False, output_dir, 0

def main():
    """Main function to run benchmarks on all GPUs"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run benchmark on all available GPUs")
    parser.add_argument(
        "--model", 
        default="Qwen/Qwen3-30B-A3B",
        help="Model to benchmark (default: Qwen/Qwen3-30B-A3B)"
    )
    
    args = parser.parse_args()
    
    print("🔍 Detecting available GPUs...")
    gpus = get_available_gpus()
    
    if not gpus:
        print("❌ No GPUs detected!")
        return 1
    
    print(f"✅ Found {len(gpus)} GPU(s): {', '.join(map(str, gpus))}")
    print(f"🤖 Model: {args.model}")
    
    # Ask for confirmation
    response = input(f"\nProceed to run {args.model} benchmark on all {len(gpus)} GPU(s)? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return 0
    
    # Run benchmarks on each GPU
    results = []
    total_start_time = time.time()
    
    for gpu_id in gpus:
        success, output_dir, duration = run_benchmark_on_gpu(gpu_id, args.model)
        results.append({
            'gpu_id': gpu_id,
            'success': success,
            'output_dir': output_dir,
            'duration': duration
        })
        
        # Small delay between runs
        if gpu_id != gpus[-1]:  # Don't delay after the last GPU
            print("\n⏳ Waiting 5 seconds before next GPU...")
            time.sleep(5)
    
    total_duration = time.time() - total_start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"🎯 Total GPUs: {len(gpus)}")
    print(f"🤖 Model: {args.model}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_duration:.2f} seconds")
    
    print(f"\n📁 Results directories:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} GPU {result['gpu_id']}: {result['output_dir']}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
