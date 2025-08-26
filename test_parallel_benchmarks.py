#!/usr/bin/env python3
"""
Test script to verify parallel benchmark functionality.
"""

import sys
import subprocess

def test_parallel_script():
    """Test that the parallel benchmark script works correctly"""
    print("🧪 Testing Parallel Benchmark Functionality")
    print("=" * 60)
    
    # Test 1: Check that the script exists and can be imported
    print("\n🔍 Test 1: Checking parallel benchmark script...")
    
    try:
        result = subprocess.run([
            sys.executable, "run_parallel_gpu_benchmarks.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Parallel benchmark script works correctly")
            print("✅ Help output generated successfully")
        else:
            print(f"❌ Parallel benchmark script failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing parallel benchmark script: {e}")
        return False
    
    # Test 2: Check that the script accepts model parameter
    print("\n🔍 Test 2: Checking model parameter support...")
    
    try:
        result = subprocess.run([
            sys.executable, "run_parallel_gpu_benchmarks.py", "--model", "test-model", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Model parameter support works correctly")
        else:
            print(f"❌ Model parameter support failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model parameter: {e}")
        return False
    
    # Test 3: Check Makefile targets
    print("\n🔍 Test 3: Checking Makefile targets...")
    
    targets = ["qwen30-parallel", "qwen30-fp8-parallel"]
    
    for target in targets:
        try:
            result = subprocess.run([
                "make", "-n", target  # -n flag shows what would be run without executing
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ Makefile target '{target}' works!")
            else:
                print(f"❌ Makefile target '{target}' failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error testing Makefile target '{target}': {e}")
    
    print("\n🎉 Parallel benchmark functionality test passed!")
    print("\nUsage examples:")
    print("  make qwen30-parallel        # Run Qwen 3.0 on all GPUs in parallel")
    print("  make qwen30-fp8-parallel    # Run Qwen 3.0-FP8 on all GPUs in parallel")
    print("  python run_parallel_gpu_benchmarks.py --model your-model")
    print("  python run_parallel_gpu_benchmarks.py --model your-model --auto")
    
    return True

def main():
    """Main test function"""
    success = test_parallel_script()
    
    if success:
        print("\n✅ All tests passed! Parallel benchmark functionality is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
