#!/bin/bash

# Makefile alternative for easy commands

.PHONY: help setup build train predict api test clean logs

help:
	@echo "Credit Card Fraud Detection - Docker Commands"
	@echo ""
	@echo "Usage: make [command]"
	@echo ""
	@echo "Commands:"
	@echo "  setup       - Initial setup and build"
	@echo "  build       - Build Docker images"
	@echo "  train       - Train models"
	@echo "  predict     - Make predictions"
	@echo "  api         - Start API server"
	@echo "  test        - Test API endpoints"
	@echo "  logs        - View container logs"
	@echo "  clean       - Remove all containers and images"
	@echo ""

setup:
	bash setup_docker.sh

build:
	docker-compose build --no-cache

train:
	docker-compose run fraud-detection python main.py --mode train --data data/creditcard.csv

predict:
	docker-compose run fraud-detection python main.py --mode predict --model-name xgb

api:
	docker-compose run -p 5000:5000 fraud-api

test:
	bash test_api.sh

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	@echo "Cleaned up all containers and volumes"
