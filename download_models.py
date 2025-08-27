#!/usr/bin/env python3
"""
Model download utility for vLLM Engine-Direct Connection Ceiling Benchmark

Downloads models from Hugging Face to local disk for offline use.
"""

import argparse
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM


def download_model_and_tokenizer(model_name: str, trust_remote_code: bool = True):
    """
    Download a model and its tokenizer from Hugging Face to the default cache location.
    
    Args:
        model_name: Hugging Face model identifier (e.g., "Qwen/Qwen3-30B-A3B")
        trust_remote_code: Whether to trust remote code from the model
    """
    print(f"🤖 Downloading model: {model_name}")
    print("📁 Using default Hugging Face cache location")
    
    try:
        # Download tokenizer to default cache
        print("📥 Downloading tokenizer...")
        AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        print("✅ Tokenizer downloaded successfully")
        
        # Download model to default cache
        print("📥 Downloading model (this may take a while)...")
        AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        print("✅ Model downloaded successfully")
        
        # Get the cache directory path
        from transformers import file_utils
        cache_dir = file_utils.default_cache_path
        print(f"📂 Models cached in: {cache_dir}")
        
        print(f"🎉 Model '{model_name}' downloaded successfully to Hugging Face cache")
        
        return model_name  # Return the original model name since it will be found in cache
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None


def main():
    """Main function to handle command line arguments and download models"""
    parser = argparse.ArgumentParser(
        description="Download models from Hugging Face to default cache location"
    )
    parser.add_argument(
        "model_name",
        help="Hugging Face model identifier (e.g., 'Qwen/Qwen3-30B-A3B')"
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Don't trust remote code from the model"
    )
    
    args = parser.parse_args()
    
    # Download the model
    result = download_model_and_tokenizer(
        model_name=args.model_name,
        trust_remote_code=not args.no_trust_remote_code
    )
    
    if result:
        print(f"\n🚀 To use this model in benchmarks, use:")
        print(f"   --model {result}")
        sys.exit(0)
    else:
        print("❌ Model download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
