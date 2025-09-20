# Deployment Guide

## Quick Start

### Local Development
```bash
# Run locally with Docker
./deploy.sh local
```

### Production Deployment

#### Option 1: Direct Docker Commands
```bash
# Build image
docker build -t car-price-classifier .

# Run with docker-compose
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs
```

#### Option 2: Using Deployment Script
```bash
# Make script executable
chmod +x deploy.sh

# Deploy locally
./deploy.sh local

# For production with registry
./deploy.sh push    # Push to registry
./deploy.sh pull    # Pull from registry (on server)
./deploy.sh deploy  # Deploy on server
```

## Cross-Platform Deployment

### From macOS to Linux Server

1. **On macOS (Development Machine):**
   ```bash
   # Build multi-platform image
   docker buildx build --platform linux/amd64,linux/arm64 -t car-price-classifier .
   
   # Or build for specific Linux architecture
   docker buildx build --platform linux/amd64 -t car-price-classifier .
   
   # Save image to file
   docker save car-price-classifier > car-price-classifier.tar
   
   # Transfer to Linux server
   scp car-price-classifier.tar user@server:/path/to/deployment/
   ```

2. **On Linux Server:**
   ```bash
   # Load image
   docker load < car-price-classifier.tar
   
   # Run with docker-compose
   docker-compose up -d
   ```

### Alternative: Using Docker Registry

1. **Push from macOS:**
   ```bash
   # Tag for registry
   docker tag car-price-classifier your-registry.com/car-price-classifier
   
   # Push to registry
   docker push your-registry.com/car-price-classifier
   ```

2. **Pull on Linux Server:**
   ```bash
   # Pull from registry
   docker pull your-registry.com/car-price-classifier
   
   # Tag locally
   docker tag your-registry.com/car-price-classifier car-price-classifier
   
   # Deploy
   docker-compose up -d
   ```

## Environment Variables

Create a `.env` file for production:
```bash
PYTHONUNBUFFERED=1
PYTHONPATH=/app
PORT=8050
```

## Health Checks

The application includes health checks:
```bash
# Check container health
docker-compose ps

# View logs
docker-compose logs -f

# Test endpoint
curl http://localhost:8050
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

2. **Permission issues:**
   ```bash
   sudo docker-compose up -d
   ```

3. **Memory issues:**
   ```bash
   # Add memory limits to docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 2G
   ```

### Logs and Debugging

```bash
# View application logs
docker-compose logs car-price-app

# Follow logs in real-time
docker-compose logs -f car-price-app

# Execute commands in container
docker-compose exec car-price-app bash

# Check container stats
docker stats
```

## Production Considerations

1. **Reverse Proxy:** Use nginx or traefik for production
2. **SSL/TLS:** Configure HTTPS certificates
3. **Monitoring:** Add logging and monitoring solutions
4. **Backup:** Regular backup of model artifacts
5. **Updates:** Use rolling deployments for zero downtime

## Performance Optimization

1. **Multi-stage builds** for smaller images
2. **Resource limits** in docker-compose.yml
3. **Caching** for model predictions
4. **Load balancing** for high traffic

## Security

1. **Non-root user** in container
2. **Secrets management** for sensitive data
3. **Network isolation** with Docker networks
4. **Regular updates** of base images
