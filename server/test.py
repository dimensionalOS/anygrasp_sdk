#!/usr/bin/env python3
import os
import sys
import asyncio
import json
import numpy as np
import websockets
import argparse
from PIL import Image
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def process_depth_image(data_dir, downsample_factor=10):
    """Process depth and color images to get point cloud data, same as in demo.py."""
    # Get data
    color_path = os.path.join(data_dir, 'color.png')
    depth_path = os.path.join(data_dir, 'depth.png')
    
    colors = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    depths = np.array(Image.open(depth_path))
    
    # Camera intrinsics - same as in demo.py
    fx, fy = 927.17, 927.37
    cx, cy = 651.32, 349.62
    scale = 1000.0
    
    # Get point cloud - same algorithm as in demo.py
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    
    # Filter points
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    
    original_size = len(points)
    print(f"Original point cloud: {original_size} points")
    print(f"Point cloud bounds: {points.min(axis=0)} to {points.max(axis=0)}")
    
    # Downsample the point cloud to reduce size for WebSocket transmission
    if downsample_factor > 1:
        indices = np.random.choice(len(points), len(points) // downsample_factor, replace=False)
        points = points[indices]
        colors = colors[indices]
        print(f"Downsampled point cloud: {len(points)} points (1/{downsample_factor} of original)")
    
    return points, colors

def display_grasps(grasps):
    """Display the received grasp data."""
    print(f"\nReceived {len(grasps)} grasps:")
    
    if len(grasps) == 0:
        print("No grasps detected.")
        return
    
    # Display top 5 grasps or all if less than 5
    num_to_display = min(5, len(grasps))
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
    print(f"\nAll scores: {[round(grasp['score'], 4) for grasp in grasps[:10]]}")
    if len(grasps) > 10:
        print("... (more scores omitted)")

async def test_grasp_server(server_url, points, colors, workspace_limits=None):
    """Test the grasp server by sending a point cloud and receiving grasp results."""
    print(f"Connecting to server at {server_url}...")
    
    try:
        async with websockets.connect(server_url) as ws:
            # Set default workspace limits if none provided
            if workspace_limits is None:
                workspace_limits = [-0.19, 0.12, 0.02, 0.15, 0.0, 1.0]
            
            # Prepare data to send
            data = {
                "points": points.tolist(),
                "colors": colors.tolist(),
                "lims": workspace_limits
            }
            
            print(f"Sending point cloud with {len(points)} points...")
            start_time = time.time()
            
            # Send the data
            await ws.send(json.dumps(data))
            print("Data sent, waiting for response...")
            
            # Receive the response
            response = await ws.recv()
            end_time = time.time()
            
            # Parse the response
            grasps = json.loads(response)
            
            print(f"Response received in {(end_time - start_time):.2f} seconds")
            display_grasps(grasps)
            
            return grasps
    except Exception as e:
        print(f"Error communicating with server: {str(e)}")
        return []

async def main():
    parser = argparse.ArgumentParser(description="Test AnyGrasp WebSocket Server")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host IP address of the AnyGrasp server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port of the AnyGrasp server")
    parser.add_argument("--data_dir", type=str, default="../grasp_detection/example_data",
                        help="Directory containing example color.png and depth.png")
    parser.add_argument("--downsample", type=int, default=10,
                        help="Downsample factor to reduce point cloud size (higher = smaller data)")
    parser.add_argument("--save_result", action="store_true",
                        help="Save the grasp results to a JSON file")
    args = parser.parse_args()
    
    ws_url = f"ws://{args.host}:{args.port}/ws/grasp"
    
    print(f"Testing AnyGrasp server at {ws_url}")
    print(f"Using example data from: {args.data_dir}")
    
    # Process example depth and color images
    try:
        points, colors = process_depth_image(args.data_dir, args.downsample)
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        return
    
    # Test the server
    grasps = await test_grasp_server(ws_url, points, colors)
    
    # Save results if requested
    if args.save_result and grasps:
        output_file = "grasp_results.json"
        with open(output_file, "w") as f:
            json.dump(grasps, f, indent=2)
        print(f"Saved grasp results to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
