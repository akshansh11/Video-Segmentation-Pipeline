# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 11:10:57 2025

@author: Akshansh Mishra
"""

"""
Complete Video Segmentation Pipeline
Processes video with multiple segmentation algorithms and saves results as MP4 files
"""

import cv2
import numpy as np
import os
from datetime import datetime

def setup_output_directory():
    """Create output directory for segmented videos"""
    output_dir = r"C:\Users\pedit\Downloads\segmented_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using existing output directory: {output_dir}")
    return output_dir

def get_video_properties(video_path):
    """Get video properties for creating output videos"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    print(f"Video Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    return fps, width, height, total_frames

def create_video_writer(output_path, fps, width, height):
    """Create video writer with MP4 codec"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise ValueError(f"Cannot create video writer for: {output_path}")
    
    return writer

def color_based_segmentation(input_path, output_dir, fps, width, height):
    """Segment video based on color ranges"""
    print("\n=== Starting Color-Based Segmentation ===")
    
    output_path = os.path.join(output_dir, "01_color_segmentation.mp4")
    cap = cv2.VideoCapture(input_path)
    writer = create_video_writer(output_path, fps, width, height)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define multiple color ranges
        # Blue objects
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Green objects
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Red objects (HSV wraps around for red)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(green_mask, red_mask))
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create colored overlay
        result = frame.copy()
        result[blue_mask > 0] = [255, 0, 0]    # Blue regions in blue
        result[green_mask > 0] = [0, 255, 0]   # Green regions in green
        result[red_mask > 0] = [0, 0, 255]     # Red regions in red
        
        # Blend with original
        final_result = cv2.addWeighted(frame, 0.6, result, 0.4, 0)
        
        writer.write(final_result)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames...")
    
    cap.release()
    writer.release()
    print(f"✓ Color segmentation complete: {output_path}")

def background_subtraction_segmentation(input_path, output_dir, fps, width, height):
    """Background subtraction for moving object detection"""
    print("\n=== Starting Background Subtraction ===")
    
    output_path = os.path.join(output_dir, "02_background_subtraction.mp4")
    cap = cv2.VideoCapture(input_path)
    writer = create_video_writer(output_path, fps, width, height)
    
    # Create background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(
        detectShadows=True,
        varThreshold=50,
        history=500
    )
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply background subtraction
        fg_mask = back_sub.apply(frame)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Remove small contours
                cv2.fillPoly(fg_mask, [contour], 0)
        
        # Create result
        result = frame.copy()
        result[fg_mask == 255] = [0, 255, 255]  # Highlight foreground in yellow
        
        # Blend with original
        final_result = cv2.addWeighted(frame, 0.7, result, 0.3, 0)
        
        writer.write(final_result)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames...")
    
    cap.release()
    writer.release()
    print(f"✓ Background subtraction complete: {output_path}")

def edge_segmentation(input_path, output_dir, fps, width, height):
    """Edge-based segmentation using multiple edge detectors"""
    print("\n=== Starting Edge Segmentation ===")
    
    output_path = os.path.join(output_dir, "03_edge_segmentation.mp4")
    cap = cv2.VideoCapture(input_path)
    writer = create_video_writer(output_path, fps, width, height)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Multiple edge detection methods
        # Canny edges
        canny = cv2.Canny(blurred, 50, 150)
        
        # Sobel edges
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(sobel / sobel.max() * 255)
        
        # Laplacian edges
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Combine edge methods
        combined_edges = cv2.bitwise_or(canny, cv2.bitwise_or(sobel, laplacian))
        
        # Dilate to make edges more visible
        kernel = np.ones((2, 2), np.uint8)
        combined_edges = cv2.dilate(combined_edges, kernel, iterations=1)
        
        # Convert to color and overlay
        edges_colored = cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
        
        writer.write(result)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames...")
    
    cap.release()
    writer.release()
    print(f"✓ Edge segmentation complete: {output_path}")

def kmeans_segmentation(input_path, output_dir, fps, width, height, k=6):
    """K-means clustering for color quantization"""
    print(f"\n=== Starting K-Means Segmentation (k={k}) ===")
    
    output_path = os.path.join(output_dir, f"04_kmeans_k{k}_segmentation.mp4")
    cap = cv2.VideoCapture(input_path)
    writer = create_video_writer(output_path, fps, width, height)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Reshape frame for clustering
        data = frame.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert back to image
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_frame = segmented_data.reshape(frame.shape)
        
        writer.write(segmented_frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames...")
    
    cap.release()
    writer.release()
    print(f"✓ K-means segmentation complete: {output_path}")

def watershed_segmentation(input_path, output_dir, fps, width, height):
    """Watershed algorithm for region segmentation"""
    print("\n=== Starting Watershed Segmentation ===")
    
    output_path = os.path.join(output_dir, "05_watershed_segmentation.mp4")
    cap = cv2.VideoCapture(input_path)
    writer = create_video_writer(output_path, fps, width, height)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(frame, markers)
        
        # Create result
        result = frame.copy()
        result[markers == -1] = [0, 255, 255]  # Mark boundaries in cyan
        
        writer.write(result)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames...")
    
    cap.release()
    writer.release()
    print(f"✓ Watershed segmentation complete: {output_path}")

def optical_flow_segmentation(input_path, output_dir, fps, width, height):
    """Motion segmentation using optical flow"""
    print("\n=== Starting Optical Flow Segmentation ===")
    
    output_path = os.path.join(output_dir, "06_optical_flow_segmentation.mp4")
    cap = cv2.VideoCapture(input_path)
    writer = create_video_writer(output_path, fps, width, height)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create HSV representation
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to BGR
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Create motion mask
        motion_threshold = 3.0
        motion_mask = magnitude > motion_threshold
        
        # Apply segmentation
        result = frame.copy()
        result[motion_mask] = flow_bgr[motion_mask]
        
        # Blend with original
        final_result = cv2.addWeighted(frame, 0.6, result, 0.4, 0)
        
        writer.write(final_result)
        
        # Update previous frame
        prev_gray = curr_gray
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames...")
    
    cap.release()
    writer.release()
    print(f"✓ Optical flow segmentation complete: {output_path}")

def main():
    """Main function to run all segmentation algorithms"""
    print("=" * 60)
    print("VIDEO SEGMENTATION PIPELINE")
    print("=" * 60)
    
    # Input video path
    input_video = r"C:\Users\pedit\Downloads\nerf.mp4"
    
    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"ERROR: Input video not found at: {input_video}")
        print("Please make sure the file exists and the path is correct.")
        return
    
    print(f"Input video: {input_video}")
    
    # Setup output directory
    output_dir = setup_output_directory()
    
    try:
        # Get video properties
        fps, width, height, total_frames = get_video_properties(input_video)
        
        # Run all segmentation algorithms
        print(f"\nStarting segmentation of {total_frames} frames...")
        start_time = datetime.now()
        
        # Algorithm 1: Color-based segmentation
        color_based_segmentation(input_video, output_dir, fps, width, height)
        
        # Algorithm 2: Background subtraction
        background_subtraction_segmentation(input_video, output_dir, fps, width, height)
        
        # Algorithm 3: Edge detection
        edge_segmentation(input_video, output_dir, fps, width, height)
        
        # Algorithm 4: K-means clustering
        kmeans_segmentation(input_video, output_dir, fps, width, height, k=6)
        
        # Algorithm 5: Watershed
        watershed_segmentation(input_video, output_dir, fps, width, height)
        
        # Algorithm 6: Optical flow
        optical_flow_segmentation(input_video, output_dir, fps, width, height)
        
        # Summary
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Total processing time: {processing_time}")
        print(f"Output directory: {output_dir}")
        print("\nGenerated files:")
        
        # List all generated files
        for i, filename in enumerate([
            "01_color_segmentation.mp4",
            "02_background_subtraction.mp4", 
            "03_edge_segmentation.mp4",
            "04_kmeans_k6_segmentation.mp4",
            "05_watershed_segmentation.mp4",
            "06_optical_flow_segmentation.mp4"
        ], 1):
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {i}. {filename} ({size_mb:.1f} MB)")
            else:
                print(f"  {i}. {filename} (FAILED)")
        
        print(f"\n✓ All segmentation algorithms completed successfully!")
        
    except Exception as e:
        print(f"\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
