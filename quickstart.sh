#!/bin/bash

# Quick start script - trains and runs API

set -e

echo "========================================"
echo "Credit Card Fraud Detection - Quick Start"
echo "========================================"

# Check dataset
if [ ! -f "data/creditcard.csv" ]; then
    echo "❌ Error: data/creditcard.csv not found"
    echo "📥 Please download the dataset from:"
    echo "   https://www.kaggle.com/mlg-ulb/creditcardfraud"
    echo "💾 Extract to: data/creditcard.csv"
    exit 1
fi

echo "✅ Dataset found"

# Create directories
mkdir -p models results logs

echo "\n🔨 Building Docker image..."
docker-compose build

echo "\n🚀 Training models..."
docker-compose run fraud-detection python main.py --mode train --data data/creditcard.csv

echo "\n✅ Training complete!"
echo "\n🌐 Starting API server..."
echo "📍 API will be available at: http://localhost:5000"
echo "\n📖 Test endpoints with: bash test_api.sh"

docker-compose up fraud-api
