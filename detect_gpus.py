#!/usr/bin/env python3
"""
Script to detect available GPUs and return their device IDs.
"""

import subprocess
import sys
import json

def get_available_gpus():
    """Get list of available GPU device IDs"""
    try:
        # Try using nvidia-smi to get GPU count
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # Parse nvidia-smi output to count GPUs
            gpu_lines = result.stdout.strip().split('\n')
            gpu_count = len(gpu_lines)
            return list(range(gpu_count))
        else:
            print("Warning: nvidia-smi failed, trying alternative method", file=sys.stderr)
            
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        print("Warning: nvidia-smi not available, trying alternative method", file=sys.stderr)
    
    try:
        # Alternative: try using torch to detect GPUs
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            return list(range(gpu_count))
    except ImportError:
        pass
    
    try:
        # Alternative: try using pynvml
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        return list(range(gpu_count))
    except ImportError:
        pass
    
    # Fallback: assume at least one GPU
    print("Warning: Could not detect GPUs, assuming 1 GPU available", file=sys.stderr)
    return [0]

def main():
    """Main function to output GPU list"""
    gpus = get_available_gpus()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        # Output as JSON for programmatic use
        print(json.dumps(gpus))
    else:
        # Output as space-separated list for shell use
        print(' '.join(map(str, gpus)))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
