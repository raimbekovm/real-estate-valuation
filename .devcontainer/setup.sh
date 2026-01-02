#!/bin/bash
# Setup script for Codespaces

echo "Installing Chrome..."
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt-get update
sudo apt-get install -y google-chrome-stable

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo "Run parser with: python -m src.scrapers.krisha_kz --city astana --selenium --resume"
