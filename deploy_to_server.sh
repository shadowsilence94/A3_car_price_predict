#!/bin/bash
echo "🚀 Deploying to mlbrain server..."

# SSH to server and deploy
ssh mlbrain << 'ENDSSH'
echo "📍 Current location:"
pwd
echo "📂 Navigating to deployment directory..."
cd ~/st126010/car-predictor

echo "📋 Current containers:"
docker ps

echo "🛑 Stopping current containers..."
docker compose down

echo "🗑️ Removing old images..."
docker image rm shadowsilence94/car-price-predit:latest || true

echo "⬇️ Pulling latest image..."
docker pull shadowsilence94/car-price-predit:latest

echo "🔍 Checking docker-compose.yml..."
cat docker-compose.yml

echo "🚀 Starting new containers..."
docker compose up -d

echo "✅ Deployment completed!"
echo "📋 New containers:"
docker ps

echo "🧹 Cleaning up unused images..."
docker system prune -f

echo "🌐 App should be available at: https://st126010.ml.brain.cs.ait.ac.th"
ENDSSH

echo "✅ Deployment script completed!"
