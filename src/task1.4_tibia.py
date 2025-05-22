import numpy as np
import SimpleITK as sitk
import os
import json
from scipy.ndimage import distance_transform_edt

# --- CONFIGURATION ---
RESULT_DIR = "results"
MASK_PATHS = {
    "original": os.path.join(RESULT_DIR, "original_mask.nii.gz"),
    "expanded_2mm": os.path.join(RESULT_DIR, "expanded_mask_2mm.nii.gz"),
    "expanded_4mm": os.path.join(RESULT_DIR, "expanded_mask_4mm.nii.gz"),
    "randomized_1": os.path.join(RESULT_DIR, "randomized_mask_1_seed42.nii.gz"),
    "randomized_2": os.path.join(RESULT_DIR, "randomized_mask_2_seed43.nii.gz")
}

def load_mask(path):
    """Load mask and return array and spacing"""
    mask_image = sitk.ReadImage(path)
    mask_array = sitk.GetArrayFromImage(mask_image)
    spacing = mask_image.GetSpacing()  # (x, y, z)
    return mask_array, spacing, mask_image

def find_tibia(mask_array):
    """Extract tibia (lower bone) from the mask"""
    # Find connected components
    labeled = sitk.GetImageFromArray(mask_array)
    labeled = sitk.ConnectedComponent(labeled)
    labeled_array = sitk.GetArrayFromImage(labeled)
    
    # Get properties of each component
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled)
    
    # Find the component with the lowest centroid (tibia)
    min_z = float('inf')
    tibia_label = None
    
    for label in stats.GetLabels():
        centroid = stats.GetCentroid(label)
        if centroid[2] < min_z:  # z-coordinate
            min_z = centroid[2]
            tibia_label = label
    
    # Create tibia mask
    tibia_mask = (labeled_array == tibia_label).astype(np.uint8)
    return tibia_mask

def find_lowest_points(tibia_mask, spacing):
    """
    Find medial and lateral lowest points on tibial surface
    Returns coordinates in physical space (mm)
    """
    # Get the lowest slice with bone
    z_slices = np.any(tibia_mask, axis=(1, 2))
    lowest_z = np.where(z_slices)[0][-1]
    
    # Get the lowest slice
    lowest_slice = tibia_mask[lowest_z]
    
    # Find the boundary points
    boundary = sitk.GetImageFromArray(lowest_slice)
    boundary = sitk.BinaryContour(boundary)
    boundary_array = sitk.GetArrayFromImage(boundary)
    
    # Get boundary points
    boundary_points = np.where(boundary_array)
    y_coords = boundary_points[0]
    x_coords = boundary_points[1]
    
    # Find leftmost and rightmost points
    left_idx = np.argmin(x_coords)
    right_idx = np.argmax(x_coords)
    
    # Convert to physical coordinates
    medial_point = [
        x_coords[left_idx] * spacing[0],
        y_coords[left_idx] * spacing[1],
        lowest_z * spacing[2]
    ]
    
    lateral_point = [
        x_coords[right_idx] * spacing[0],
        y_coords[right_idx] * spacing[1],
        lowest_z * spacing[2]
    ]
    
    return medial_point, lateral_point

def main():
    results = {}
    
    for mask_name, mask_path in MASK_PATHS.items():
        print(f"\nProcessing {mask_name}...")
        
        # Load mask
        mask_array, spacing, mask_image = load_mask(mask_path)
        
        # Extract tibia
        tibia_mask = find_tibia(mask_array)
        
        # Find landmarks
        medial_point, lateral_point = find_lowest_points(tibia_mask, spacing)
        
        # Store results
        results[mask_name] = {
            "medial_point": medial_point,
            "lateral_point": lateral_point
        }
        
        print(f"Medial point: {medial_point}")
        print(f"Lateral point: {lateral_point}")
    
    # Save results
    output_path = os.path.join(RESULT_DIR, "tibial_landmark.csv")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
