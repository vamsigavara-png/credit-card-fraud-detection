#!/bin/bash

# Script to test the fraud detection API

API_URL="http://localhost:5000"

echo "========================================"
echo "Credit Card Fraud Detection API Tests"
echo "========================================"

# Test 1: Health check
echo -e "\n1. Testing health endpoint..."
curl -s $API_URL/health | jq .

# Test 2: Model info
echo -e "\n2. Testing model info endpoint..."
curl -s $API_URL/model-info | jq .

# Test 3: Single prediction
echo -e "\n3. Testing single prediction..."
curl -s -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -0.1, 0.2, -0.3],
    "threshold": 0.5
  }' | jq .

# Test 4: Batch prediction
echo -e "\n4. Testing batch prediction..."
curl -s -X POST $API_URL/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -0.1, 0.2, -0.3],
      [0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.9, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1, -0.2, 0.3, -0.4]
    ],
    "threshold": 0.5
  }' | jq .

echo -e "\n========================================"
echo "Tests completed!"
echo "========================================"
