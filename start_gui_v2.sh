#!/bin/bash

# Zyrix/Mike Agent Dashboard Launcher

echo "🚀 Initializing Mike Agent Pro Terminal..."

# Check for Python dependencies
echo "📦 Checking Python backend dependencies..."
pip3 install -r requirements.txt > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️ Warning: Failed to install Python dependencies. Please run 'pip3 install -r requirements.txt' manually."
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js to run the frontend."
    exit 1
fi

# Check for npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm."
    exit 1
fi

# Setup Frontend
echo "🎨 Setting up Frontend..."
cd gui_v2/frontend
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies (this may take a minute)..."
    npm install
fi
cd ../..

# Start Backend
echo "🔮 Starting FastAPI Backend..."
python3 -m uvicorn gui_v2.backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to initialize..."
sleep 5

# Start Frontend
echo "💻 Starting Next.js Frontend..."
cd gui_v2/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

echo "✅ System Operational"
echo "------------------------------------------------"
echo "🌐 Frontend: http://localhost:3000"
echo "🔌 Backend:  http://localhost:8000/docs"
echo "------------------------------------------------"
echo "Press CTRL+C to stop all services"

# Handle shutdown
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

wait





