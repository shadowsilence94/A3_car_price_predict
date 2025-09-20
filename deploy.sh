#!/bin/bash

# Car Price Classifier Deployment Script
# Usage: ./deploy.sh [push|pull|deploy]

set -e

IMAGE_NAME="car-price-classifier"
REGISTRY_URL="https://hub.docker.com/repositories/shadowsilence94"  # Replace with your registry
TAG="latest"

case "$1" in
    "push")
        echo "🚀 Building and pushing Docker image..."
        docker build -t $IMAGE_NAME:$TAG .
        docker tag $IMAGE_NAME:$TAG $REGISTRY_URL/$IMAGE_NAME:$TAG
        docker push $REGISTRY_URL/$IMAGE_NAME:$TAG
        echo "✅ Image pushed successfully!"
        ;;
    
    "pull")
        echo "📥 Pulling Docker image..."
        docker pull $REGISTRY_URL/$IMAGE_NAME:$TAG
        docker tag $REGISTRY_URL/$IMAGE_NAME:$TAG $IMAGE_NAME:$TAG
        echo "✅ Image pulled successfully!"
        ;;
    
    "deploy")
        echo "🚀 Deploying application..."
        docker-compose down || true
        docker-compose up -d
        echo "✅ Application deployed successfully!"
        echo "🌐 Access the app at: http://localhost:8050"
        ;;
    
    "local")
        echo "🏠 Building and running locally..."
        docker build -t $IMAGE_NAME:$TAG .
        docker-compose down || true
        docker-compose up -d
        echo "✅ Application running locally!"
        echo "🌐 Access the app at: http://localhost:8050"
        ;;
    
    *)
        echo "Usage: $0 {push|pull|deploy|local}"
        echo ""
        echo "Commands:"
        echo "  push   - Build and push image to registry"
        echo "  pull   - Pull image from registry"
        echo "  deploy - Deploy using docker-compose"
        echo "  local  - Build and run locally"
        exit 1
        ;;
esac
