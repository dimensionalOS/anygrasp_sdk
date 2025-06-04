#!/bin/bash

# Start AnyGrasp WebSocket server in tmux session
SESSION_NAME="anygrasp_server"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y tmux
fi

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

# If session doesn't exist, create it
if [ $? != 0 ]; then
    echo "Creating new tmux session: $SESSION_NAME"
    tmux new-session -d -s $SESSION_NAME
    
    # Install requirements
    tmux send-keys -t $SESSION_NAME "cd $(pwd)" C-m
    tmux send-keys -t $SESSION_NAME "pip install -r requirements.txt" C-m
    
    # Start the server
    tmux send-keys -t $SESSION_NAME "python main.py" C-m
    
    echo "AnyGrasp WebSocket server started in tmux session: $SESSION_NAME"
    echo "To attach to the session, run: tmux attach -t $SESSION_NAME"
    echo "To detach from the session, press: Ctrl+B then D"
else
    echo "Session $SESSION_NAME already exists"
    echo "To attach to the session, run: tmux attach -t $SESSION_NAME"
fi

# Print IP information
echo "Server IP information:"
ip addr show | grep -w inet | grep -v "127.0.0.1" | awk '{print $2}' | cut -d/ -f1
