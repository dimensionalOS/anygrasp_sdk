#!/usr/bin/env python3
import asyncio
import json
import numpy as np
import websockets
import argparse
from PIL import Image
import open3d as o3d
import sys
import os
import time

# Add parent directory to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def process_depth_image(color_path, depth_path):
    """Process depth and color images to get point cloud data."""
    # Load images
    colors = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    depths = np.array(Image.open(depth_path))
    
    # Camera intrinsics
    fx, fy = 927.17, 927.37
    cx, cy = 651.32, 349.62
    scale = 1000.0
    
    # Generate point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    
    # Filter valid points
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    
    return points, colors

def display_grasps(grasps):
    """Display the received grasp data."""
    print(f"\nReceived {len(grasps)} grasps:")
    
    if len(grasps) == 0:
        print("No grasps detected.")
        return
    
    # Display top 3 grasps or all if less than 3
    num_to_display = min(3, len(grasps))
    for i in range(num_to_display):
        grasp = grasps[i]
        print(f"\nGrasp #{i+1}:")
        print(f"  Score: {grasp['score']:.4f}")
        print(f"  Width: {grasp['width']:.4f}")
        print(f"  Height: {grasp['height']:.4f}")
        print(f"  Depth: {grasp['depth']:.4f}")
        print(f"  Translation: {grasp['translation']}")
        print(f"  Rotation Matrix:")
        for row in grasp['rotation_matrix']:
            print(f"    {row}")
    
    # Print all scores
    print(f"\nAll scores: {[round(grasp['score'], 4) for grasp in grasps]}")

async def send_grasp_request(server_url, points, colors):
    """Send point cloud data to the server and receive grasp results."""
    async with websockets.connect(server_url) as ws:
        # Prepare data to send
        data = {
            "points": points.tolist(),
            "colors": colors.tolist(),
            "lims": [-0.19, 0.12, 0.02, 0.15, 0.0, 1.0]  # Default workspace limits
        }
        
        print(f"Sending point cloud data ({len(points)} points)...")
        send_time = time.time()
        
        # Send the data
        await ws.send(json.dumps(data))
        
        # Receive the response
        response = await ws.recv()
        recv_time = time.time()
        
        # Parse the response
        result = json.loads(response)
        
        print(f"Response received in {(recv_time - send_time):.2f} seconds")
        
        return result

async def main():
    parser = argparse.ArgumentParser(description="AnyGrasp WebSocket Client Example")
    parser.add_argument("--server", type=str, default="ws://localhost:8000/ws/grasp",
                        help="WebSocket server URL")
    parser.add_argument("--color", type=str, default="../grasp_detection/example_data/color.png",
                        help="Path to color image")
    parser.add_argument("--depth", type=str, default="../grasp_detection/example_data/depth.png",
                        help="Path to depth image")
    args = parser.parse_args()
    
    print(f"Connecting to server: {args.server}")
    print(f"Processing images: {args.color} and {args.depth}")
    
    # Process depth image to get point cloud
    try:
        points, colors = process_depth_image(args.color, args.depth)
        print(f"Generated point cloud with {len(points)} points")
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        return
    
    # Send data to server and get grasp results
    try:
        result = await send_grasp_request(args.server, points, colors)
        display_grasps(result)
    except Exception as e:
        print(f"Error communicating with server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
