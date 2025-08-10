#!/usr/bin/env python3
"""
Ruff linting and formatting helper script for VBC AI backend.
Run this script to check and fix code quality issues.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔍 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Main linting workflow."""
    backend_dir = Path(__file__).parent
    
    print("🚀 Running Ruff linting and formatting for VBC AI backend...")
    print(f"📁 Working directory: {backend_dir}")
    
    # Change to backend directory
    import os
    os.chdir(backend_dir)
    
    success = True
    
    # 1. Check linting issues
    success &= run_command(
        ["ruff", "check", "app/", "--diff"],
        "Checking linting issues"
    )
    
    # 2. Check formatting issues  
    success &= run_command(
        ["ruff", "format", "app/", "--diff"],
        "Checking formatting issues"
    )
    
    # 3. Ask user if they want to fix
    if not success:
        response = input("\n🔧 Would you like to auto-fix the issues? (y/N): ")
        if response.lower() in ['y', 'yes']:
            # Fix linting issues
            run_command(
                ["ruff", "check", "app/", "--fix"],
                "Fixing linting issues"
            )
            
            # Fix formatting issues
            run_command(
                ["ruff", "format", "app/"],
                "Fixing formatting issues"
            )
            
            print("\n✨ Code has been automatically fixed!")
        else:
            print("\n📝 Run with --fix to automatically fix issues.")
    else:
        print("\n🎉 All checks passed! Code is clean.")


if __name__ == "__main__":
    main()
