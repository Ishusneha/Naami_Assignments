# Medical Image Processing Project

This project implements a series of medical image processing tasks focused on knee MRI analysis, specifically targeting tibial bone segmentation, mask manipulation, and landmark detection.

## Project Structure

```
.
├── src/
│   ├── task1.1_segmentation.py    # Initial segmentation implementation
│   ├── task1.2_expansion.py       # Mask expansion functionality
│   ├── task1.3_randomization.py   # Mask randomization implementation
│   ├── task1.4_tibia.py           # Landmark detection
│   └── results/                   # Output results directory
├── visualizations/                 # Images for Visualization 
└── requirements.txt               # Project dependencies
```

## Features

### 1. Segmentation (task1.1_segmentation.py)
- Implements initial segmentation of medical images
- Processes input data to create binary masks
- Saves segmentation results for further processing

### 2. Mask Expansion (task1.2_expansion.py)
- Expands binary masks using morphological operations
- Configurable expansion parameters
- Preserves mask integrity while increasing coverage

### 3. Mask Randomization (task1.3_randomization.py)
- Implements random variations of binary masks
- Maintains anatomical constraints
- Useful for data augmentation and testing

### 4. Landmark Detection (task1.4_tibia.py)
- Detects anatomical landmarks in medical images
- Visualizes landmarks on the original images
- Supports multiple landmark types

## Requirements

The project requires Python 3.x and the following dependencies:
- numpy
- scipy
- scikit-image
- matplotlib
- SimpleITK
- nibabel
- pandas
- tqdm

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

Each script can be run independently. Here's how to use them:

1. Segmentation:
```bash
python src/task1.1_segmentation.py
```

2. Mask Expansion:
```bash
python src/task1.2_expansion.py
```

3. Mask Randomization:
```bash
python src/task1.3_randomization.py
```

4. Landmark Detection:
```bash
python src/task1.4_tibia.py
```


## Data Organization

- Input data should be placed in the root directory
- Results will be saved in the `src/results/` directory
- Each script creates its own subdirectory for outputs

## Output Format

- Segmentation results are saved as binary masks
- Expanded masks maintain the original format with increased coverage
- Randomized masks include variation parameters
- Landmark detection results include coordinates

## Notes
- Ensure sufficient disk space for processing large medical images
- Some operations may require significant memory for 3D volumes
- Results are automatically saved to prevent data loss
- Each script includes progress tracking for long operations