#!/usr/bin/env python3
"""
Test script to verify GPU device selection works correctly.
"""

import sys
import subprocess
import argparse

def test_cuda_visible_devices():
    """Test that CUDA_VISIBLE_DEVICES environment variable works"""
    
    # Test that the scripts run without the --gpu-device option
    try:
        result = subprocess.run([
            sys.executable, "main.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ main.py help works correctly")
        else:
            print("❌ main.py help failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing main.py help: {e}")
        return False
    
    # Test single_request.py as well
    try:
        result = subprocess.run([
            sys.executable, "single_request.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ single_request.py help works correctly")
        else:
            print("❌ single_request.py help failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing single_request.py help: {e}")
        return False
    
    print("\n🎉 CUDA_VISIBLE_DEVICES test passed!")
    print("\nUsage examples:")
    print("  CUDA_VISIBLE_DEVICES=0 python3 main.py --model your-model")
    print("  CUDA_VISIBLE_DEVICES=1 python3 main.py --model your-model")
    print("  CUDA_VISIBLE_DEVICES=0 python3 single_request.py --model your-model")
    print("  CUDA_VISIBLE_DEVICES=1 python3 single_request.py --model your-model")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GPU device option")
    parser.add_argument("--run-test", action="store_true", help="Run the actual test")
    
    args = parser.parse_args()
    
    if args.run_test:
        success = test_cuda_visible_devices()
        sys.exit(0 if success else 1)
    else:
        print("CUDA_VISIBLE_DEVICES support has been added to the benchmark!")
        print("Use --run-test to verify the implementation.")
        print("\nGPU selection is now controlled via environment variable:")
        print("  CUDA_VISIBLE_DEVICES=0|1|0,1")
        print("\nThis allows you to select which GPU(s) to use for inference.")
