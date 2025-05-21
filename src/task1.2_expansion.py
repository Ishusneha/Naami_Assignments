import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os

# --- CONFIGURATION ---
CT_PATH = "../left_knee.nii.gz"
MASK_PATH = "../src/results/original_mask.nii.gz"
EXPANSION_MM = 2  # Parameter for expanding outward by N mm
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

def expand_mask(mask_array, spacing, expansion_mm):
    """
    Expands the mask uniformly by the specified distance in millimeters.
    
    Args:
        mask_array: The binary mask as a numpy array
        spacing: The voxel spacing (x, y, z) in mm
        expansion_mm: The expansion distance in mm
        
    Returns:
        The expanded binary mask
    """
    # Convert physical distance (mm) to voxel units, rounding up to ensure we meet the required expansion
    expansion_voxels = [
        max(1, int(np.ceil(expansion_mm / s))) for s in spacing[::-1]  # spacing[::-1] = (z, y, x)
    ]
    print(f"Expansion of {expansion_mm}mm converted to voxels: {expansion_voxels}")
    
    # Create a structuring element for binary dilation
    # The size is (2*expansion + 1) to create a symmetric element around the center
    structuring_element = np.ones(
        (2 * expansion_voxels[0] + 1, 2 * expansion_voxels[1] + 1, 2 * expansion_voxels[2] + 1),
        dtype=np.uint8
    )
    
    # Perform binary dilation to expand the mask
    expanded_mask = binary_dilation(mask_array, structure=structuring_element).astype(np.uint8)
    
    return expanded_mask

def main():
    # --- LOAD IMAGES ---
    print("Loading images...")
    ct_image = sitk.ReadImage(CT_PATH)
    mask_image = sitk.ReadImage(MASK_PATH)
    
    image_array = sitk.GetArrayFromImage(ct_image)
    original_mask = sitk.GetArrayFromImage(mask_image).astype(np.uint8)
    
    # --- GET VOXEL SPACING ---
    spacing = ct_image.GetSpacing()  # (x, y, z)
    print("Voxel spacing (mm):", spacing)
    
    # --- PERFORM EXPANSION ---
    print(f"Expanding mask by {EXPANSION_MM}mm...")
    expanded_mask = expand_mask(original_mask, spacing, EXPANSION_MM)
    
    # --- SAVE EXPANDED MASK ---
    print("Saving expanded mask...")
    expanded_mask_sitk = sitk.GetImageFromArray(expanded_mask)
    expanded_mask_sitk.CopyInformation(mask_image)  # Copy metadata from original mask
    expanded_mask_output_path = os.path.join(RESULT_DIR, f"expanded_mask_{EXPANSION_MM}mm.nii.gz")
    sitk.WriteImage(expanded_mask_sitk, expanded_mask_output_path)
    print(f"Expanded mask saved to: {expanded_mask_output_path}")
if __name__ == "__main__":
    main()