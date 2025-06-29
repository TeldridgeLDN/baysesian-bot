#!/usr/bin/env python3
"""
Install missing dependencies for the Bayesian Crypto Trading Bot.
"""

import subprocess
import sys
import platform

def install_package(package):
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("📦 Installing Dependencies for Bayesian Crypto Trading Bot")
    print("=" * 60)
    
    # Required packages
    packages = [
        "python-dotenv",  # Environment variables
        "pydantic",       # Data validation
        "PyYAML",         # Configuration files
        "pandas",         # Data manipulation
        "numpy",          # Numerical operations
        "scipy",          # Scientific computing
        "scikit-learn",   # Machine learning utilities
        "ccxt",           # Cryptocurrency exchange APIs
        "python-telegram-bot==20.7",  # Telegram bot
        "psutil",         # System monitoring
    ]
    
    # TensorFlow (platform-specific)
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "darwin" and machine == "arm64":
        # Apple Silicon
        tf_packages = ["tensorflow-macos", "tensorflow-metal"]
        print("📱 Detected Apple Silicon Mac - installing optimized TensorFlow")
    else:
        # Intel/AMD
        tf_packages = ["tensorflow"]
        print("💻 Installing standard TensorFlow")
    
    all_packages = packages + tf_packages
    
    print(f"\n📋 Installing {len(all_packages)} packages...")
    
    failed_packages = []
    
    for i, package in enumerate(all_packages, 1):
        print(f"\n[{i}/{len(all_packages)}] Installing {package}...")
        
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
            failed_packages.append(package)
    
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    
    if failed_packages:
        print(f"❌ Failed to install: {failed_packages}")
        print("\nTry installing manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
        return 1
    else:
        print("✅ All packages installed successfully!")
        print("\n📋 Next steps:")
        print("1. Copy .env.example to .env")
        print("2. Edit .env with your API keys")
        print("3. Run: python test_deployment_readiness.py")
        print("4. Deploy: python deploy_paper_trading.py")
        return 0

if __name__ == "__main__":
    exit(main())