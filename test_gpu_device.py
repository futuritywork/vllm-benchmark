#!/usr/bin/env python3
"""
Test script to verify GPU device selection works correctly.
"""

import sys
import subprocess
import argparse

def test_gpu_device_option():
    """Test that the --gpu-device option is properly recognized"""
    
    # Test with help to see if the option is available
    try:
        result = subprocess.run([
            sys.executable, "main.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "--gpu-device" in result.stdout:
            print("✅ GPU device option is available in main.py")
        else:
            print("❌ GPU device option not found in main.py help")
            return False
            
    except Exception as e:
        print(f"❌ Error testing main.py help: {e}")
        return False
    
    # Test single_request.py as well
    try:
        result = subprocess.run([
            sys.executable, "single_request.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "--gpu-device" in result.stdout:
            print("✅ GPU device option is available in single_request.py")
        else:
            print("❌ GPU device option not found in single_request.py help")
            return False
            
    except Exception as e:
        print(f"❌ Error testing single_request.py help: {e}")
        return False
    
    print("\n🎉 GPU device option test passed!")
    print("\nUsage examples:")
    print("  python3 main.py --model your-model --gpu-device cuda:0")
    print("  python3 main.py --model your-model --gpu-device cuda:1")
    print("  python3 single_request.py --model your-model --gpu-device 0")
    print("  python3 single_request.py --model your-model --gpu-device 1")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GPU device option")
    parser.add_argument("--run-test", action="store_true", help="Run the actual test")
    
    args = parser.parse_args()
    
    if args.run_test:
        success = test_gpu_device_option()
        sys.exit(0 if success else 1)
    else:
        print("GPU device option has been added to the benchmark!")
        print("Use --run-test to verify the implementation.")
        print("\nNew options available:")
        print("  --gpu-device cuda:0|cuda:1|0|1")
        print("\nThis allows you to select which GPU to use for inference.")
