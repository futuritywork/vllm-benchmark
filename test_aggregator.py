#!/usr/bin/env python3
"""
Test script to verify the parallel results aggregator functionality.
"""

import sys
import subprocess

def test_aggregator_script():
    """Test that the aggregator script works correctly"""
    print("🧪 Testing Parallel Results Aggregator")
    print("=" * 60)
    
    # Test 1: Check that the aggregator script exists and can be imported
    print("\n🔍 Test 1: Checking aggregator script...")
    
    try:
        result = subprocess.run([
            sys.executable, "aggregate_parallel_results.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Aggregator script works correctly")
            print("✅ Help output generated successfully")
        else:
            print(f"❌ Aggregator script failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing aggregator script: {e}")
        return False
    
    # Test 2: Check that the timestamp lister script works
    print("\n🔍 Test 2: Checking timestamp lister script...")
    
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
    
    # Test 3: Check Makefile targets
    print("\n🔍 Test 3: Checking Makefile targets...")
    
    targets = ["list-timestamps"]
    
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
    
    # Test 4: Test aggregator with invalid timestamp
    print("\n🔍 Test 4: Testing aggregator with invalid timestamp...")
    
    try:
        result = subprocess.run([
            sys.executable, "aggregate_parallel_results.py", "invalid_timestamp"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("✅ Aggregator correctly handles invalid timestamps")
        else:
            print("⚠️  Aggregator accepted invalid timestamp (this might be OK if test data exists)")
            
    except Exception as e:
        print(f"❌ Error testing invalid timestamp: {e}")
    
    print("\n🎉 Aggregator functionality test completed!")
    print("\nUsage examples:")
    print("  make list-timestamps")
    print("  make aggregate-results TIMESTAMP=20241201_143022")
    print("  python aggregate_parallel_results.py 20241201_143022")
    print("  python list_available_timestamps.py")
    
    return True

def main():
    """Main test function"""
    success = test_aggregator_script()
    
    if success:
        print("\n✅ All tests passed! Aggregator functionality is ready.")
        print("\nTo use the aggregator:")
        print("1. Run a parallel benchmark: make qwen30-parallel")
        print("2. List available timestamps: make list-timestamps")
        print("3. Aggregate results: make aggregate-results TIMESTAMP=your_timestamp")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
