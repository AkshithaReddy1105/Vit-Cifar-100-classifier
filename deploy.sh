#!/bin/bash

# Vision Transformer Deployment Script for GCP
# This script automates the entire deployment process

set -e  # Exit on error

echo "======================================"
echo "Vision Transformer GCP Deployment"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install Google Cloud SDK first."
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker first."
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Get project ID
echo ""
read -p "Enter your GCP Project ID: " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    print_error "Project ID cannot be empty!"
    exit 1
fi

print_info "Using project: $PROJECT_ID"

# Set project
gcloud config set project $PROJECT_ID

# Choose region
echo ""
echo "Available regions:"
echo "1) us-central1 (Iowa)"
echo "2) us-east1 (South Carolina)"
echo "3) europe-west1 (Belgium)"
echo "4) asia-east1 (Taiwan)"
read -p "Choose region (1-4) [default: 1]: " region_choice

case $region_choice in
    2) REGION="us-east1" ;;
    3) REGION="europe-west1" ;;
    4) REGION="asia-east1" ;;
    *) REGION="us-central1" ;;
esac

print_info "Using region: $REGION"

# Ask about training
echo ""
read -p "Do you want to train the model locally first? (y/n) [n]: " train_choice

if [[ $train_choice == "y" || $train_choice == "Y" ]]; then
    print_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_info "Starting model training (this will take a while)..."
    python train_vit.py
    
    if [ -f "vit_cifar100.pth" ]; then
        print_info "Model training completed successfully!"
    else
        print_error "Training failed - model file not found"
        exit 1
    fi
else
    print_warning "Skipping training - app will use pretrained ImageNet weights"
fi

# Enable required APIs
echo ""
print_info "Enabling required GCP APIs..."
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Configure Docker for Artifact Registry (newer than Container Registry)
print_info "Configuring Docker authentication..."
gcloud auth configure-docker --quiet

# Build Docker image
echo ""
print_info "Building Docker image (this may take 5-10 minutes)..."
SERVICE_NAME="vit-classifier"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"
docker build -t $IMAGE_NAME .

if [ $? -eq 0 ]; then
    print_info "Docker image built successfully!"
else
    print_error "Docker build failed!"
    exit 1
fi

# Test locally
echo ""
read -p "Do you want to test the application locally first? (y/n) [y]: " test_choice

if [[ $test_choice != "n" && $test_choice != "N" ]]; then
    print_info "Starting local Docker container..."
    print_info "Access at: http://localhost:8080"
    print_warning "Press Ctrl+C to stop and continue with deployment"
    
    docker run -p 8080:8080 $IMAGE_NAME
fi

# Push to Container Registry
echo ""
print_info "Pushing image to Google Container Registry..."
docker push $IMAGE_NAME

if [ $? -eq 0 ]; then
    print_info "Image pushed successfully!"
else
    print_error "Failed to push image!"
    exit 1
fi

# Deploy to Cloud Run
echo ""
print_info "Deploying to Cloud Run..."
print_info "This may take a few minutes..."

gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --allow-unauthenticated \
    --port 8080 \
    --max-instances 10 \
    --quiet

if [ $? -eq 0 ]; then
    echo ""
    print_info "✅ Deployment successful!"
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --platform managed \
        --region $REGION \
        --format 'value(status.url)')
    
    echo ""
    echo "======================================"
    echo "🎉 Your application is live!"
    echo "======================================"
    echo ""
    echo "Service URL: $SERVICE_URL"
    echo ""
    echo "Test the deployment:"
    echo "  Health check: curl $SERVICE_URL/health"
    echo "  Web interface: Open $SERVICE_URL in browser"
    echo ""
    
    # Test health endpoint
    print_info "Testing health endpoint..."
    sleep 5
    curl -s $SERVICE_URL/health | python -m json.tool
    echo ""
    
    # Open in browser
    read -p "Open application in browser? (y/n) [y]: " open_choice
    if [[ $open_choice != "n" && $open_choice != "N" ]]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open $SERVICE_URL
        elif command -v open &> /dev/null; then
            open $SERVICE_URL
        else
            print_info "Please open $SERVICE_URL in your browser"
        fi
    fi
    
    echo ""
    echo "======================================"
    echo "📝 Useful commands:"
    echo "======================================"
    echo ""
    echo "View logs:"
    echo "  gcloud run services logs tail $SERVICE_NAME --region $REGION"
    echo ""
    echo "Update service:"
    echo "  docker build -t $IMAGE_NAME ."
    echo "  docker push $IMAGE_NAME"
    echo "  gcloud run deploy $SERVICE_NAME --image $IMAGE_NAME --region $REGION"
    echo ""
    echo "Delete service:"
    echo "  gcloud run services delete $SERVICE_NAME --region $REGION"
    echo ""
    echo "View in console:"
    echo "  https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME"
    echo ""
    
else
    print_error "Deployment failed! Check the logs above for details."
    exit 1
fi