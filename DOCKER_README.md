# Fraud Detection Docker Quick Start

## Prerequisites

- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop))
- Docker Compose installed (included with Docker Desktop)

## Quick Start (One Command!)

### 1. Download the dataset

First, download the Credit Card Fraud Detection dataset from Kaggle:

```bash
# Option A: Using Kaggle API
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/

# Option B: Manual download
# Go to: https://www.kaggle.com/mlg-ulb/creditcardfraud
# Download and extract to data/creditcard.csv
```

### 2. Run Everything with Docker Compose

```bash
# Build and start all services
docker-compose up --build
```

This single command will:
- Build the Docker image
- Install all dependencies
- Train the models
- Start the API server
- Everything runs in isolated containers

## Individual Docker Commands

### Train Models Only

```bash
docker-compose run fraud-detection python main.py --mode train --data data/creditcard.csv
```

### Start API Server Only

```bash
docker-compose run -p 5000:5000 fraud-api
```

### Run Predictions

```bash
docker-compose run fraud-detection python main.py --mode predict --model-name xgb
```

### Interactive Shell

```bash
docker-compose run fraud-detection bash
```

## Using the API

Once the API is running at `http://localhost:5000`:

### Health Check

```bash
curl http://localhost:5000/health
```

### Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -0.1, 0.2, -0.3],
    "threshold": 0.5
  }'
```

### Batch Predictions

```bash
curl -X POST http://localhost:5000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      [0.1, -0.2, 0.3, ...],
      [0.2, -0.3, 0.4, ...]
    ],
    "threshold": 0.5
  }'
```

### Model Information

```bash
curl http://localhost:5000/model-info
```

## Docker Commands Reference

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove all data
docker-compose down -v

# Rebuild images
docker-compose build --no-cache
```

## File Structure Inside Container

```
/app/
├── data/              # Dataset (mounted from ./data)
├── models/            # Trained models (mounted from ./models)
├── results/           # Results (mounted from ./results)
├── logs/              # Logs (mounted from ./logs)
├── src/               # Python source code
├── api/               # Flask API
├── main.py            # Training script
└── requirements.txt   # Dependencies
```

## Troubleshooting

### Port 5000 already in use

```bash
# Change port in docker-compose.yml
# ports:
#   - "8000:5000"  # Use 8000 instead
```

### Dataset not found

```bash
# Make sure you have data/creditcard.csv
ls -la data/
```

### Models not found

```bash
# Train models first
docker-compose run fraud-detection python main.py --mode train
```

### Out of memory

```bash
# Increase Docker memory limit in Docker Desktop settings
# Settings > Resources > Memory
```

## Next Steps

1. ✅ Download dataset
2. ✅ Run `docker-compose up --build`
3. ✅ Wait for models to train
4. ✅ Access API at `http://localhost:5000`
5. ✅ Test predictions

## Support

For more information, see the main [README.md](../README.md)
