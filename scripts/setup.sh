#!/bin/bash

# SmartShelf AI - Setup Script
# This script sets up the entire SmartShelf AI platform

set -e

echo "🚀 Setting up SmartShelf AI Platform..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

node_version=$(node --version | cut -d'v' -f2)
echo "✅ Node.js version: $node_version"

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/database
mkdir -p data/models
mkdir -p data/vector_store
mkdir -p data/raw
mkdir -p logs

# Generate initial data
echo "🎲 Generating synthetic data..."
python3 scripts/generate_data.py --months 3 --products 50 --output_dir data/raw

# Initialize database
echo "🗄️ Initializing database..."
python3 scripts/init_database.py

# Train ML models (optional - may take time)
echo "🤖 Training ML models..."
python3 scripts/train_models.py --model demand --data_dir data/raw --output_dir data/models

echo "🎉 SmartShelf AI setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Start the backend: cd backend && python3 -m uvicorn app.main:app --reload"
echo "2. Start the copilot: cd copilot_chatbot && python3 -m uvicorn main:app --reload --port 8001"
echo "3. Start the frontend: cd frontend && npm run dev"
echo ""
echo "🌐 Access points:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo "- Copilot API: http://localhost:8001"
echo ""
echo "📚 For more information, see README.md"
