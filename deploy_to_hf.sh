#!/bin/bash

# Configuration
# HF_TOKEN should be set in your environment or passed as an argument
HF_SPACE="Nishanth16265/Nishanth"
GITHUB_REPO="https://github.com/Nishanth-1729/Meta_hackathon_01"

echo "🚀 Preparing CRust for Hugging Face Space deployment..."

# 1. Update git remote for HF if not exists
if ! git remote | grep -q hf; then
    echo "Enter your Hugging Face Token to authorize the push:"
    read -s HF_TOKEN
    echo "Adding Hugging Face remote..."
    git remote add hf https://Nishanth16265:$HF_TOKEN@huggingface.co/spaces/$HF_SPACE
fi

# 2. Add and commit all changes (Blog, requirements, app.py)
git add .
git commit -m "Deploying CRust to Hugging Face Space (A10G Optimized)"

# 3. Push to GitHub
echo "Pushing to GitHub..."
git push origin main

# 4. Push to Hugging Face Space
echo "Pushing to Hugging Face Space..."
git push hf main --force

echo "✅ Deployment complete!"
echo "🔗 Watch your Space build here: https://huggingface.co/spaces/$HF_SPACE"
echo "⚠️ IMPORTANT: Make sure to set your HF_TOKEN as a Secret in the Space Settings if you haven't already!"
