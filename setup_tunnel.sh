#!/bin/bash

echo "🚀 Medical Chatbot - Ollama Tunnel Setup"
echo "======================================="

# Check if Ollama is running
echo "📍 Checking Ollama status..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running on localhost:11434"
else
    echo "❌ Ollama is not running. Please start it first:"
    echo "   brew services start ollama"
    echo "   ollama run llama3.1:8b"
    exit 1
fi

# Check if ngrok is installed
echo "📍 Checking ngrok installation..."
if command -v ngrok &> /dev/null; then
    echo "✅ ngrok is installed"
else
    echo "❌ ngrok is not installed. Installing via brew..."
    brew install ngrok
fi

# Start ngrok tunnel
echo "🌐 Starting ngrok tunnel for Ollama..."
echo "📝 This will create a public URL for your local Ollama instance."
echo "🔒 Remember: This exposes your Ollama API to the internet. Use responsibly!"
echo ""

# Start ngrok in background and capture URL
ngrok http 11434 > /dev/null 2>&1 &
NGROK_PID=$!

# Wait for ngrok to start
sleep 3

# Get the public URL
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tunnel = data['tunnels'][0]
    print(tunnel['public_url'])
except:
    print('ERROR')
")

if [ "$NGROK_URL" == "ERROR" ] || [ -z "$NGROK_URL" ]; then
    echo "❌ Failed to get ngrok URL. Check if ngrok is running."
    kill $NGROK_PID 2>/dev/null
    exit 1
fi

echo "🎉 Tunnel created successfully!"
echo "📍 Ollama Public URL: $NGROK_URL"
echo ""
echo "🔧 For Render deployment, set this environment variable:"
echo "   OLLAMA_BASE_URL=$NGROK_URL"
echo ""
echo "🧪 Test your tunnel:"
echo "   curl $NGROK_URL/api/tags"
echo ""
echo "⚠️  Keep this terminal open to maintain the tunnel."
echo "💡 To stop the tunnel, press Ctrl+C"

# Keep the script running
trap "echo '🛑 Stopping tunnel...'; kill $NGROK_PID 2>/dev/null; exit 0" INT

# Test the tunnel
echo "🧪 Testing tunnel connection..."
if curl -s "$NGROK_URL/api/tags" > /dev/null; then
    echo "✅ Tunnel is working correctly!"
else
    echo "⚠️  Tunnel might need a moment to initialize. Test manually."
fi

echo ""
echo "🔄 Tunnel is active. Press Ctrl+C to stop."

# Wait indefinitely
while true; do
    sleep 60
    # Check if ngrok is still running
    if ! kill -0 $NGROK_PID 2>/dev/null; then
        echo "❌ ngrok process died. Restarting..."
        ngrok http 11434 > /dev/null 2>&1 &
        NGROK_PID=$!
        sleep 3
    fi
done 