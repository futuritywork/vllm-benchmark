#!/usr/bin/env python3
"""
Interactive prompt selection module for vLLM benchmark
Scans the prompts/ folder and allows interactive selection if no prompt is specified
"""

from pathlib import Path
from typing import Optional


def scan_prompts_folder(prompts_dir: str = "prompts") -> list[tuple[str, str]]:
    """
    Scan the prompts folder and return a list of (filename, full_path) tuples
    
    Args:
        prompts_dir: Directory containing prompt files
        
    Returns:
        List of tuples (filename, full_path) sorted by filename
    """
    prompts_path = Path(prompts_dir)
    if not prompts_path.exists():
        return []
    
    prompts = []
    for file_path in sorted(prompts_path.glob("*.txt")):
        prompts.append((file_path.name, str(file_path)))
    
    return prompts


def select_prompt_interactively(prompts_dir: str = "prompts") -> str:
    """
    Interactively prompt the user to select a prompt file
    
    Args:
        prompts_dir: Directory containing prompt files
        
    Returns:
        Path to the selected prompt file
    """
    prompts = scan_prompts_folder(prompts_dir)
    
    if not prompts:
        raise FileNotFoundError(
            f"No .txt files found in {prompts_dir}/ directory"
        )
    
    print(f"\n📁 Found {len(prompts)} prompt file(s) in {prompts_dir}/:")
    print("-" * 60)
    
    for idx, (filename, _) in enumerate(prompts, start=1):
        print(f"  {idx}. {filename}")
    
    print("-" * 60)
    
    while True:
        try:
            choice = input(f"\nSelect a prompt file (1-{len(prompts)}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(prompts):
                selected_filename, selected_path = prompts[choice_num - 1]
                print(f"✅ Selected: {selected_filename}")
                return selected_path
            else:
                print(f"❌ Please enter a number between 1 and {len(prompts)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print("\n\n❌ Selection cancelled")
            raise


def get_prompt_path(
    prompt_arg: Optional[str] = None,
    prompts_dir: str = "prompts"
) -> str:
    """
    Get the prompt file path, either from argument or interactive selection
    
    Args:
        prompt_arg: Optional prompt file path provided via command line
        prompts_dir: Directory containing prompt files
        
    Returns:
        Path to the prompt file to use
    """
    # If prompt is provided via argument, use it
    if prompt_arg:
        prompt_path = Path(prompt_arg)
        if prompt_path.exists():
            return str(prompt_path)
        else:
            raise FileNotFoundError(f"Prompt file not found: {prompt_arg}")
    
    # Otherwise, interactively select from prompts folder
    return select_prompt_interactively(prompts_dir)

