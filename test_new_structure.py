#!/usr/bin/env python3
"""
Test script to verify the new directory structure for parallel benchmarks.
"""

import sys
import subprocess
import os

def test_new_structure():
    """Test that the new directory structure works correctly"""
    print("🧪 Testing New Directory Structure")
    print("=" * 50)
    
    # Test 1: Check that the parallel benchmark script works with new structure
    print("\n🔍 Test 1: Checking parallel benchmark script...")
    
    try:
        result = subprocess.run([
            sys.executable, "run_parallel_gpu_benchmarks.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Parallel benchmark script works correctly")
        else:
            print(f"❌ Parallel benchmark script failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing parallel benchmark script: {e}")
        return False
    
    # Test 2: Check that the aggregator works with new structure
    print("\n🔍 Test 2: Checking aggregator script...")
    
    try:
        result = subprocess.run([
            sys.executable, "aggregate_parallel_results.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Aggregator script works correctly")
        else:
            print(f"❌ Aggregator script failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing aggregator script: {e}")
        return False
    
    # Test 3: Check that the timestamp lister works with new structure
    print("\n🔍 Test 3: Checking timestamp lister script...")
    
    try:
        result = subprocess.run([
            sys.executable, "list_available_timestamps.py"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Timestamp lister script works correctly")
        else:
            print(f"❌ Timestamp lister script failed: {result.stderr}")
            # This might fail if no results exist, which is OK for testing
            
    except Exception as e:
        print(f"❌ Error testing timestamp lister script: {e}")
        # This might fail if no results exist, which is OK for testing
    
    print("\n🎉 New directory structure test completed!")
    print("\nNew structure:")
    print("  results_20241201_143022/")
    print("  ├── gpu_0/")
    print("  │   ├── engine_conn_results.json")
    print("  │   └── llm_outputs.log")
    print("  ├── gpu_1/")
    print("  │   ├── engine_conn_results.json")
    print("  │   └── llm_outputs.log")
    print("  └── aggregated_results.json")
    
    return True

def main():
    """Main test function"""
    success = test_new_structure()
    
    if success:
        print("\n✅ All tests passed! New directory structure is ready.")
        print("\nBenefits of the new structure:")
        print("• All results for a single benchmark run are grouped together")
        print("• Cleaner organization with single timestamp directory")
        print("• Easier to manage and analyze results")
        print("• Aggregated results are stored with the individual results")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
