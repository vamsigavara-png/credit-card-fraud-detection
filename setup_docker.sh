#!/bin/bash

# Quick setup script for Docker

set -e

echo "========================================"
echo "Credit Card Fraud Detection Setup"
echo "========================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop first."
    echo "📥 Download: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker is installed"

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker daemon is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker daemon is running"

# Check if dataset exists
if [ ! -f "data/creditcard.csv" ]; then
    echo "⚠️  Dataset not found at data/creditcard.csv"
    echo "📥 Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud"
    echo "💾 Extract to: data/creditcard.csv"
    read -p "Press Enter once you've downloaded the dataset..."
fi

# Create directories
mkdir -p data models results logs

echo "✅ Directory structure created"

# Build images
echo "\n🔨 Building Docker images..."
docker-compose build

echo "\n========================================"
echo "Setup Complete!"
echo "========================================"
echo "\n🚀 To start training, run:"
echo "   docker-compose up"
echo "\n📊 To start the API, run:"
echo "   docker-compose run -p 5000:5000 fraud-api"
echo "\n📖 For more information, see DOCKER_README.md"
echo "========================================"
