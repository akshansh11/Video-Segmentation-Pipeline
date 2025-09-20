# Video Segmentation Pipeline

A comprehensive Python pipeline that applies multiple computer vision segmentation algorithms to video files, generating segmented outputs for analysis and research purposes.

![freecompress-segemented (1) (1) (1)](https://github.com/user-attachments/assets/245fcd6d-9600-416c-b31c-285d6d3d7cb1)

## Features

- **6 Different Segmentation Algorithms** implemented from scratch
- **Batch Processing** of entire video files
- **Progress Tracking** with real-time frame count updates
- **Automatic Output Management** with organized file naming
- **Error Handling** and validation for robust processing
- **Performance Metrics** including processing time and file sizes

## Segmentation Algorithms

| Algorithm | Description | Output File |
|-----------|-------------|-------------|
| **Color-Based** | Segments objects by HSV color ranges (blue, green, red) | `01_color_segmentation.mp4` |
| **Background Subtraction** | Detects moving objects using MOG2 algorithm | `02_background_subtraction.mp4` |
| **Edge Detection** | Combines Canny, Sobel, and Laplacian edge detectors | `03_edge_segmentation.mp4` |
| **K-Means Clustering** | Groups similar colors into k clusters (default k=6) | `04_kmeans_k6_segmentation.mp4` |
| **Watershed** | Region-based segmentation with boundary detection | `05_watershed_segmentation.mp4` |
| **Optical Flow** | Motion-based segmentation using Farneback method | `06_optical_flow_segmentation.mp4` |

## Requirements

```bash
pip install opencv-python numpy
```

**System Requirements:**
- Python 3.7+
- OpenCV 4.x
- NumPy
- Sufficient disk space for output videos

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/video-segmentation-pipeline.git
   cd video-segmentation-pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Update input path:**
   Edit the `input_video` variable in `main()` function:
   ```python
   input_video = r"path/to/your/video.mp4"
   ```

4. **Run the pipeline:**
   ```bash
   python video_segmentation.py
   ```

## Usage

### Basic Usage

```python
from video_segmentation import VideoSegmentation

# Initialize with input video path
input_video = "path/to/video.mp4"
output_directory = "segmented_videos"

# Get video properties
fps, width, height, total_frames = get_video_properties(input_video)

# Run individual algorithms
color_based_segmentation(input_video, output_directory, fps, width, height)
background_subtraction_segmentation(input_video, output_directory, fps, width, height)
edge_segmentation(input_video, output_directory, fps, width, height)
kmeans_segmentation(input_video, output_directory, fps, width, height, k=6)
watershed_segmentation(input_video, output_directory, fps, width, height)
optical_flow_segmentation(input_video, output_directory, fps, width, height)
```

### Custom Configuration

**Modify K-Means clusters:**
```python
kmeans_segmentation(input_video, output_directory, fps, width, height, k=8)
```

**Adjust color ranges for color-based segmentation:**
```python
# In color_based_segmentation function
lower_blue = np.array([90, 50, 50])    # Adjust HSV ranges
upper_blue = np.array([140, 255, 255])
```

**Change output directory:**
```python
output_dir = r"C:\custom\output\path"
```

## Output Structure

```
segmented_videos/
├── 01_color_segmentation.mp4
├── 02_background_subtraction.mp4
├── 03_edge_segmentation.mp4
├── 04_kmeans_k6_segmentation.mp4
├── 05_watershed_segmentation.mp4
└── 06_optical_flow_segmentation.mp4
```

## Algorithm Details

### Color-Based Segmentation
Segments objects based on HSV color space ranges. Detects blue, green, and red objects separately and applies morphological operations for noise reduction.

**Parameters:**
- HSV color ranges for each color
- Morphological kernel size (5x5)
- Blend ratio (0.6 original, 0.4 overlay)

### Background Subtraction
Uses MOG2 (Mixture of Gaussians) algorithm to learn background model and detect foreground objects.

**Parameters:**
- `detectShadows=True`
- `varThreshold=50`
- `history=500`
- Minimum contour area: 500 pixels

### Edge Detection
Combines three edge detection methods: Canny, Sobel, and Laplacian for comprehensive edge identification.

**Parameters:**
- Canny thresholds: 50, 150
- Gaussian blur: 5x5 kernel
- Sobel kernel size: 3x3

### K-Means Clustering
Groups pixel colors into k clusters for color quantization and region segmentation.

**Parameters:**
- Default clusters (k): 6
- Termination criteria: 20 iterations or 1.0 epsilon
- Initialization: KMEANS_RANDOM_CENTERS

### Watershed Segmentation
Treats the image as a topographic surface and finds watershed lines to separate regions.

**Process:**
1. OTSU thresholding
2. Morphological operations
3. Distance transform
4. Marker-based watershed

### Optical Flow
Detects motion between consecutive frames using Farneback method for motion-based segmentation.

**Parameters:**
- Pyramid scale: 0.5
- Pyramid levels: 3
- Window size: 15
- Motion threshold: 3.0

## Performance

**Typical processing times** (1920x1080, 30fps video):
- Color-based: ~2-3 seconds per 100 frames
- Background subtraction: ~3-4 seconds per 100 frames
- Edge detection: ~1-2 seconds per 100 frames
- K-means: ~5-6 seconds per 100 frames
- Watershed: ~4-5 seconds per 100 frames
- Optical flow: ~3-4 seconds per 100 frames

## Troubleshooting

### Common Issues

**Video file not found:**
```
ERROR: Input video not found at: path/to/video.mp4
```
- Verify the file path is correct
- Ensure the file exists and is accessible

**Codec issues:**
```
Cannot create video writer for: output.mp4
```
- Install additional codecs: `pip install opencv-contrib-python`
- Try different fourcc codes in `create_video_writer()`

**Memory issues with large videos:**
- Process videos in smaller segments
- Reduce video resolution before processing
- Increase available system RAM

**Slow processing:**
- Reduce video resolution
- Skip frames for faster processing
- Use GPU acceleration (requires opencv-contrib-python)

### Error Handling

The pipeline includes comprehensive error handling:
- Input file validation
- Video codec compatibility checks
- Memory allocation monitoring
- Progress tracking and logging

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Commit your changes (`git commit -am 'Add new segmentation algorithm'`)
4. Push to the branch (`git push origin feature/new-algorithm`)
5. Create a Pull Request

### Adding New Algorithms

To add a new segmentation algorithm:

1. Create a new function following the naming pattern:
   ```python
   def new_algorithm_segmentation(input_path, output_dir, fps, width, height):
       # Implementation here
       pass
   ```

2. Add the function call to `main()`:
   ```python
   new_algorithm_segmentation(input_video, output_dir, fps, width, height)
   ```

3. Update the README with algorithm description

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV community for computer vision algorithms
- NumPy developers for numerical computing support
- Contributors to the open-source computer vision research

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{video_segmentation_pipeline,
  title={Video Segmentation Pipeline},
  author={Akshansh Mishra},
  year={2025},
  url={https://github.com/akshansh11/video-segmentation-pipeline}
}
```

## Contact

- **Author:** Akshansh Mishra
- **Email:** akshansh@aifablab.com
- **GitHub:** [@yourusername](https://github.com/akshansh11)
- **Issues:** [GitHub Issues](https://github.com/akshansh11/video-segmentation-pipeline/issues)
