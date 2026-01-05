#!/usr/bin/env python3

"""
Script to read GPS NavSatFix messages from a ROS bag and visualize them.
- 3D plot in one window
- 3 projection planes in another window
Plots raw GPS data (latitude, longitude, altitude).
"""

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def read_gps_from_rosbag(bag_path, topic_name='/gnss'):
    """
    Read GPS NavSatFix messages from a ROS 2 bag.
    
    Args:
        bag_path: Path to the ROS 2 bag directory (containing .db3 file)
        topic_name: Topic name for GPS messages (default: /gnss)
    
    Returns:
        latitudes, longitudes, altitudes as numpy arrays
    """
    latitudes = []
    longitudes = []
    altitudes = []
    
    print(f"Reading ROS 2 bag: {bag_path}")
    print(f"Looking for topic: {topic_name}")
    
    try:
        # Create a SequentialReader
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get topic metadata
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        print(f"Available topics: {list(type_map.keys())}")
        
        if topic_name not in type_map:
            print(f"Topic {topic_name} not found in bag!")
            return None, None, None
        
        # Get message type
        msg_type = get_message(type_map[topic_name])
        
        count = 0
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            if topic == topic_name:
                msg = deserialize_message(data, msg_type)
                latitudes.append(msg.latitude)
                longitudes.append(msg.longitude)
                altitudes.append(msg.altitude)
                count += 1
        
        print(f"Read {count} GPS messages")
        
    except Exception as e:
        print(f"Error reading bag file: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    if count == 0:
        print(f"No messages found on topic {topic_name}")
        return None, None, None
    
    return np.array(latitudes), np.array(longitudes), np.array(altitudes)


def plot_gps_3d(latitudes, longitudes, altitudes):
    """
    Create a 3D plot of GPS data.
    
    Args:
        latitudes: Array of latitude values
        longitudes: Array of longitude values
        altitudes: Array of altitude values
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    ax.scatter(longitudes, latitudes, altitudes, 
               c='blue', 
               marker='o', 
               s=10)
    
    # Set labels
    ax.set_xlabel('Longitude (degrees)', fontsize=10)
    ax.set_ylabel('Latitude (degrees)', fontsize=10)
    ax.set_zlabel('Altitude (meters)', fontsize=10)
    ax.set_title('GPS Trajectory - 3D View', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Points: {len(latitudes)}\n"
    stats_text += f"Lat: [{latitudes.min():.6f}, {latitudes.max():.6f}]\n"
    stats_text += f"Lon: [{longitudes.min():.6f}, {longitudes.max():.6f}]\n"
    stats_text += f"Alt: [{altitudes.min():.2f}, {altitudes.max():.2f}] m"
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()


def plot_gps_projections(latitudes, longitudes, altitudes):
    """
    Create 3 projection planes of GPS data.
    
    Args:
        latitudes: Array of latitude values
        longitudes: Array of longitude values
        altitudes: Array of altitude values
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY plane: Longitude vs Latitude
    axes[0].scatter(longitudes, latitudes, c='blue', s=10)
    axes[0].set_xlabel('Longitude (degrees)', fontsize=10)
    axes[0].set_ylabel('Latitude (degrees)', fontsize=10)
    axes[0].set_title('XY Plane (Lon-Lat)', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')
    
    # XZ plane: Longitude vs Altitude
    axes[1].scatter(longitudes, altitudes, c='blue', s=10)
    axes[1].set_xlabel('Longitude (degrees)', fontsize=10)
    axes[1].set_ylabel('Altitude (meters)', fontsize=10)
    axes[1].set_title('XZ Plane (Lon-Alt)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # YZ plane: Latitude vs Altitude
    axes[2].scatter(latitudes, altitudes, c='blue', s=10)
    axes[2].set_xlabel('Latitude (degrees)', fontsize=10)
    axes[2].set_ylabel('Altitude (meters)', fontsize=10)
    axes[2].set_title('YZ Plane (Lat-Alt)', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GPS NavSatFix messages from ROS 2 bag')
    parser.add_argument('bag_path', type=str, 
                       help='Path to the ROS 2 bag directory or .db3 file')
    parser.add_argument('--topic', type=str, default='/gnss',
                       help='GPS topic name (default: /gnss)')
    
    args = parser.parse_args()
    
    # Read GPS data
    latitudes, longitudes, altitudes = read_gps_from_rosbag(
        args.bag_path, args.topic)
    
    if latitudes is None or len(latitudes) == 0:
        print("No GPS data found. Exiting.")
        return
    
    print(f"\nGPS Data Statistics:")
    print(f"  Total points: {len(latitudes)}")
    print(f"  Latitude range: [{latitudes.min():.6f}, {latitudes.max():.6f}] degrees")
    print(f"  Longitude range: [{longitudes.min():.6f}, {longitudes.max():.6f}] degrees")
    print(f"  Altitude range: [{altitudes.min():.2f}, {altitudes.max():.2f}] meters")
    
    # Create visualizations
    print("\nCreating 3D plot...")
    plot_gps_3d(latitudes, longitudes, altitudes)
    
    print("Creating projection plots...")
    plot_gps_projections(latitudes, longitudes, altitudes)
    
    print("\nDisplaying plots. Close windows to exit.")
    plt.show()


if __name__ == '__main__':
    main()
