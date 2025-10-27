#!/bin/bash
# Setup script for the Voice-Controlled AI Assistant

set -e

echo "=================================="
echo "Voice Assistant Setup Script"
echo "=================================="
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: This script is designed for Linux (Raspberry Pi OS)."
    echo "Some steps may not work on other operating systems."
    echo ""
fi

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

required_version="3.9"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "Error: Python 3.9 or higher is required."
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Install system dependencies
echo "Installing system dependencies..."
echo "You may be prompted for your password (sudo)."
echo ""

if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        python3-pip \
        python3-venv \
        portaudio19-dev \
        libopenblas-dev \
        ffmpeg \
        git
    echo "✓ System dependencies installed"
else
    echo "Warning: apt-get not found. Please install dependencies manually:"
    echo "  - Python 3.9+ with pip and venv"
    echo "  - PortAudio development files"
    echo "  - OpenBLAS development files"
    echo "  - FFmpeg"
fi

echo ""

# Create virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

echo "✓ Python dependencies installed"
echo ""

# Create models directory
echo "Creating models directory..."
mkdir -p models/piper
echo "✓ Models directory created"
echo ""

# Create config file if it doesn't exist
if [ ! -f "config.yaml" ]; then
    echo "Creating config.yaml from template..."
    cp config.example.yaml config.yaml
    echo "✓ config.yaml created"
    echo ""
    echo "⚠️  IMPORTANT: You must edit config.yaml with your settings!"
    echo "   - Add your Picovoice Access Key (from https://console.picovoice.ai/)"
    echo "   - Add your HiveMQ Cloud MQTT credentials"
    echo ""
else
    echo "✓ config.yaml already exists"
    echo ""
fi

# Download Piper TTS model (Swedish)
echo "Downloading Piper TTS model (Swedish)..."
if [ ! -f "models/piper/voice-sv-se-nst-medium.onnx" ]; then
    echo "Downloading Swedish voice model..."
    cd models/piper
    
    # Download model and config
    wget -q --show-progress \
        https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-sv-se-nst-medium.onnx \
        https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-sv-se-nst-medium.onnx.json \
        2>/dev/null || echo "Warning: Could not download Swedish voice model. Please download manually."
    
    cd ../..
    echo "✓ TTS model downloaded"
else
    echo "✓ TTS model already exists"
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit config.yaml with your settings:"
echo "   nano config.yaml"
echo ""
echo "2. Get a Picovoice Access Key:"
echo "   https://console.picovoice.ai/"
echo ""
echo "3. Setup HiveMQ Cloud (free tier available):"
echo "   https://www.hivemq.com/mqtt-cloud-broker/"
echo ""
echo "4. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "5. Run the assistant:"
echo "   python main.py"
echo ""
echo "=================================="
