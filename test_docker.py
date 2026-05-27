#!/usr/bin/env python
"""
Docker test script - runs a complete pipeline test
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report results."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Credit Card Fraud Detection - Docker Test Suite")
    print("="*60)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    if not Path('data/creditcard.csv').exists():
        print("❌ Dataset not found at data/creditcard.csv")
        print("📥 Please download from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        return 1
    
    print("✅ Dataset found")
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Run tests
    tests = [
        ("docker --version", "Docker installation check"),
        ("docker-compose --version", "Docker Compose installation check"),
        ("docker-compose build", "Build Docker image"),
        ("docker-compose run fraud-detection python main.py --mode train --data data/creditcard.csv", "Train models"),
        ("docker-compose run fraud-detection python main.py --mode predict --model-name xgb", "Make predictions"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))
    
    # Summary
    print(f"\n\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}\n")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {desc}")
    
    print(f"\n\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        print("\n🌐 Start API with: docker-compose run -p 5000:5000 fraud-api")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
