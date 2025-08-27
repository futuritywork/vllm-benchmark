import argparse

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

    print(args.model_name)

if __name__ == "__main__":
    main()