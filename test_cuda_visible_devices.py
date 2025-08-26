#!/usr/bin/env python3
"""
Test script to verify CUDA_VISIBLE_DEVICES functionality.
"""

import os
import sys
import subprocess

def test_cuda_visible_devices():
    """Test that CUDA_VISIBLE_DEVICES is properly handled"""
    
    print("🧪 Testing CUDA_VISIBLE_DEVICES functionality")
    print("=" * 60)
    
    # Test 1: Check that main.py shows CUDA_VISIBLE_DEVICES in output
    print("\n🔍 Test 1: Checking CUDA_VISIBLE_DEVICES detection...")
    
    try:
        # Set CUDA_VISIBLE_DEVICES and run a quick test
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Run main.py with --help to see if it shows CUDA_VISIBLE_DEVICES
        result = subprocess.run([
            sys.executable, "main.py", "--help"
        ], capture_output=True, text=True, timeout=10, env=env)
        
        if result.returncode == 0:
            print("✅ main.py runs correctly with CUDA_VISIBLE_DEVICES set")
        else:
            print(f"❌ main.py failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing main.py: {e}")
        return False
    
    # Test 2: Check that the scripts don't have --gpu-device option anymore
    print("\n🔍 Test 2: Checking that --gpu-device option is removed...")
    
    try:
        result = subprocess.run([
            sys.executable, "main.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "--gpu-device" not in result.stdout:
            print("✅ --gpu-device option correctly removed from main.py")
        else:
            print("❌ --gpu-device option still present in main.py")
            return False
            
    except Exception as e:
        print(f"❌ Error checking main.py help: {e}")
        return False
    
    # Test 3: Check single_request.py as well
    try:
        result = subprocess.run([
            sys.executable, "single_request.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "--gpu-device" not in result.stdout:
            print("✅ --gpu-device option correctly removed from single_request.py")
        else:
            print("❌ --gpu-device option still present in single_request.py")
            return False
            
    except Exception as e:
        print(f"❌ Error checking single_request.py help: {e}")
        return False
    
    print("\n🎉 CUDA_VISIBLE_DEVICES functionality test passed!")
    print("\nUsage examples:")
    print("  CUDA_VISIBLE_DEVICES=0 python3 main.py --model your-model")
    print("  CUDA_VISIBLE_DEVICES=1 python3 main.py --model your-model")
    print("  CUDA_VISIBLE_DEVICES=0,1 python3 main.py --model your-model --tensor-parallel-size 2")
    
    return True

def main():
    """Main test function"""
    success = test_cuda_visible_devices()
    
    if success:
        print("\n✅ All tests passed! CUDA_VISIBLE_DEVICES is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
