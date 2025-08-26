#!/usr/bin/env python3
"""
Test script to verify multi-GPU functionality.
"""

import sys
import subprocess

def test_gpu_detection():
    """Test GPU detection script"""
    print("🔍 Testing GPU detection...")
    
    try:
        result = subprocess.run([
            sys.executable, "detect_gpus.py"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpus = result.stdout.strip().split()
            print(f"✅ GPU detection works! Found {len(gpus)} GPU(s): {', '.join(gpus)}")
            return True
        else:
            print(f"❌ GPU detection failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing GPU detection: {e}")
        return False

def test_makefile_targets():
    """Test Makefile targets"""
    print("\n🔍 Testing Makefile targets...")
    
    targets = ["help", "detect-gpus", "gpu-info"]
    
    for target in targets:
        try:
            result = subprocess.run([
                "make", target
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ Makefile target '{target}' works!")
            else:
                print(f"❌ Makefile target '{target}' failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error testing Makefile target '{target}': {e}")

def main():
    """Main test function"""
    print("🧪 Testing Multi-GPU Functionality")
    print("=" * 50)
    
    # Test GPU detection
    gpu_detection_ok = test_gpu_detection()
    
    # Test Makefile targets
    test_makefile_targets()
    
    print("\n" + "=" * 50)
    if gpu_detection_ok:
        print("✅ Multi-GPU functionality is ready!")
        print("\nUsage:")
        print("  make help                    # Show all available targets")
        print("  make detect-gpus             # Detect available GPUs")
        print("  make qwen30-all-gpus         # Run Qwen 3.0 on all GPUs (interactive)")
        print("  make qwen30-all-gpus-auto    # Run Qwen 3.0 on all GPUs (automated)")
        print("\nNote: GPU selection is now controlled via CUDA_VISIBLE_DEVICES")
    else:
        print("❌ Multi-GPU functionality has issues!")
        print("Please check the GPU detection script.")
    
    return 0 if gpu_detection_ok else 1

if __name__ == "__main__":
    sys.exit(main())
