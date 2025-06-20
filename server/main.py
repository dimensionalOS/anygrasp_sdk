import os
import sys
import json
import asyncio
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Any

# Add parent directory to path to import AnyGrasp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="AnyGrasp WebSocket Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure AnyGrasp
class AnyGraspConfig:
    def __init__(self):
        self.checkpoint_path = os.getenv("CHECKPOINT_PATH")
        self.max_gripper_width = float(os.getenv("MAX_GRIPPER_WIDTH", "0.1"))
        self.gripper_height = float(os.getenv("GRIPPER_HEIGHT", "0.03"))
        self.top_down_grasp = os.getenv("TOP_DOWN_GRASP", "false").lower() == "true"
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

# Initialize AnyGrasp
anygrasp = None

@app.on_event("startup")
async def startup_event():
    global anygrasp
    config = AnyGraspConfig()
    print(f"Initializing AnyGrasp with checkpoint: {config.checkpoint_path}")
    anygrasp = AnyGrasp(config)
    anygrasp.load_net()
    print("AnyGrasp initialization complete")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_bytes(self, data: bytes, websocket: WebSocket):
        await websocket.send_bytes(data)

    async def send_json(self, data: Dict[str, Any], websocket: WebSocket):
        await websocket.send_json(data)

manager = ConnectionManager()

def process_grasp(points, colors, lims):
    """Process the grasp detection with AnyGrasp."""
    global anygrasp
    
    # Perform grasp detection
    gg, cloud = anygrasp.get_grasp(
        points, 
        colors, 
        lims=lims, 
        apply_object_mask=True, 
        dense_grasp=False, 
        collision_detection=True
    )

    # Apply NMS and sort by score
    if len(gg) > 0:
        gg = gg.nms().sort_by_score()
    
    return gg, cloud

def prepare_grasp_data(gg, max_grasps=20):
    """Extract only grasp data for transmission."""
    if len(gg) == 0:
        return []
    
    # Get top grasps sorted by score
    gg_pick = gg[0:max_grasps]
    
    # Extract the grasp parameters (similar to how they're displayed in the demo)
    grasp_list = []
    for g in gg_pick:
        grasp_dict = {
            "score": float(g.score),
            "width": float(g.width),
            "height": float(g.height) if hasattr(g, 'height') else float(0.03),
            "depth": float(g.depth),
            "translation": g.translation.tolist(),
            "rotation_matrix": g.rotation_matrix.tolist(),
            "object_id": int(g.object_id) if hasattr(g, 'object_id') else -1
        }
        grasp_list.append(grasp_dict)
    
    return grasp_list

@app.websocket("/ws/grasp")
async def grasp_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive the input data
            data = await websocket.receive_json()
            
            # Extract input parameters
            points = np.array(data["points"], dtype=np.float32)
            colors = np.array(data["colors"], dtype=np.float32)
            
            # Get workspace limits
            lims = data.get("lims", [-0.19, 0.12, 0.02, 0.15, 0.0, 1.0])
            
            # Process grasp detection
            gg, cloud = process_grasp(points, colors, lims)
            
            # We only need the grasp data, not the point cloud
            result = prepare_grasp_data(gg, os.getenv("MAX_GRASPS"))
            
            await manager.send_json(result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"Error processing grasp: {str(e)}")
        await manager.send_json({"error": str(e)}, websocket)
        manager.disconnect(websocket)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "anygrasp_loaded": anygrasp is not None}

# Get IP address endpoint
@app.get("/ip")
async def get_ip():
    import socket
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return {
        "hostname": hostname,
        "ip_address": ip_address,
        "port": os.getenv("PORT", "8000")
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
