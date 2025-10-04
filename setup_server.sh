#!/bin/bash

# Setup script for mlbrain server deployment
# Run this on your mlbrain server to set up the deployment environment

echo "🚀 Setting up car-price-app deployment environment..."

# Create application directory
mkdir -p ~/car-price-app
cd ~/car-price-app

# Create docker-compose.yml for server
cat > docker-compose.yml << 'EOF'
version: '3.9'

services:
  web:
    command: python app/app.py
    image: shadowsilence94/car-price-predit:latest
    environment:
      - HOST=0.0.0.0
      - PORT=8050
      - PYTHONUNBUFFERED=1
      - PYTHONPATH=/app
    restart: unless-stopped
    labels:
      - traefik.enable=true
      - traefik.http.services.web-st126010.loadbalancer.server.port=8050
      - traefik.http.routers.web-st126010.rule=Host(`st126010.ml.brain.cs.ait.ac.th`)
      - traefik.http.routers.web-st126010.entrypoints=websecure
      - traefik.http.routers.web-st126010.tls.certresolver=myresolver
    networks:
      - default
      - traefik_default

networks:
  default:
  traefik_default:
    external: true
EOF

echo "✅ Server setup completed!"
echo ""
echo "Next steps:"
echo "1. Run: docker compose pull"
echo "2. Run: docker compose up -d"
echo ""
echo "Your app will be available at: https://st126010.ml.brain.cs.ait.ac.th"
