#!/bin/bash
echo "Setting up GitHub repository variables and secrets..."

# Set repository variables
gh variable set SERVER_HOST --body "ml.brain.cs.ait.ac.th"
gh variable set JUMP_HOST --body "bazooka.cs.ait.ac.th"
gh variable set SERVER_PATH --body "/home/st126010/st126010/car-predictor"

# Set repository secrets
gh secret set SERVER_USER --body "st126010"

echo "✅ Variables set successfully!"
echo "⚠️  You still need to set SERVER_SSH_KEY manually:"
echo "   gh secret set SERVER_SSH_KEY --body-file ~/.ssh/id_rsa"
echo ""
echo "Or go to: https://github.com/shadowsilence94/A3_car_price_predict/settings/secrets/actions"
