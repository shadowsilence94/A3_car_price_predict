# GitHub Secrets Setup Guide

To enable automated CI/CD deployment, you need to set up the following secrets in your GitHub repository:

## Required Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions → New repository secret

### Docker Hub Secrets
1. **DOCKERHUB_USERNAME**: Your Docker Hub username
2. **DOCKERHUB_TOKEN**: Your Docker Hub access token
   - Go to Docker Hub → Account Settings → Security → New Access Token

### Server Deployment Secrets
3. **MLBRAIN_HOST**: Your mlbrain server IP address or hostname
4. **MLBRAIN_USERNAME**: SSH username for mlbrain server
5. **MLBRAIN_SSH_KEY**: Private SSH key for mlbrain server access
6. **MLBRAIN_PASSPHRASE**: SSH key passphrase (if your key has one)

## Setting up SSH Key

If you don't have an SSH key set up:

```bash
# Generate SSH key pair
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Copy public key to server
ssh-copy-id username@mlbrain-server-ip

# Copy private key content for GitHub secret
cat ~/.ssh/id_rsa
```

## Environment Setup

Create a production environment in GitHub:
1. Go to Settings → Environments
2. Create new environment named "production"
3. Add protection rules if needed

## Testing the Setup

1. Push code to main branch
2. Check Actions tab for workflow execution
3. Verify Docker image is pushed to Docker Hub
4. Confirm deployment on mlbrain server

## Troubleshooting

- Ensure SSH key has proper permissions (600)
- Verify server can pull from Docker Hub
- Check firewall settings for port 8050
- Monitor GitHub Actions logs for errors
