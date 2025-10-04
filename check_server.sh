#!/bin/bash
echo "🔍 Checking server status..."

ssh mlbrain << 'ENDSSH'
echo "📍 Current directory:"
pwd

echo "📂 Checking deployment directory:"
cd ~/st126010/car-predictor
ls -la

echo "📋 Current running containers:"
docker ps

echo "🖼️ Current Docker images:"
docker images | grep car-price

echo "📄 Docker compose file:"
cat docker-compose.yml

echo "🔍 Container logs (last 10 lines):"
docker compose logs --tail=10

ENDSSH

echo "✅ Server check completed!"
