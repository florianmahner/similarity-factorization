#!/bin/bash

# Function to cleanup background processes on exit
cleanup() {
    echo "Shutting down..."
    kill $SERVER_PID 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Start Python Server
echo "Starting Python Backend..."
# We assume this script is run from the dashboard directory
python3 server.py &
SERVER_PID=$!
echo "Backend started with PID $SERVER_PID"

# Wait a moment for backend to start
sleep 2

# Start Vite Frontend
echo "Starting Frontend..."
npm run dev

# Wait for background process
wait $SERVER_PID
