import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os

# --- CONFIGURATION ---
CT_PATH = "../left_knee.nii.gz"
MASK_PATH = "../src/results/original_mask.nii.gz"
EXPANSION_MM_2 = 2  # Parameter for expanding outward by N mm
EXPANSION_MM_4 = 4
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
    # Convert physical distance (mm) to voxel units
    # Fix: spacing is (x,y,z) but array is (z,y,x), so we need proper conversion
    expansion_voxels = [
        max(1, int(np.ceil(expansion_mm / spacing[2]))),  # z direction
        max(1, int(np.ceil(expansion_mm / spacing[1]))),  # y direction  
        max(1, int(np.ceil(expansion_mm / spacing[0])))   # x direction
    ]
    print(f"Expansion of {expansion_mm}mm converted to voxels: {expansion_voxels}")
    
    # Create a structuring element for binary dilation
    structuring_element = np.ones(
        (2 * expansion_voxels[0] + 1, 2 * expansion_voxels[1] + 1, 2 * expansion_voxels[2] + 1),
        dtype=np.uint8
    )
    
    # Perform binary dilation to expand the mask
    expanded_mask = binary_dilation(mask_array, structure=structuring_element).astype(np.uint8)
    
    return expanded_mask

def save_mask(mask_array, reference_image, output_path, description):
    """Save mask with proper metadata"""
    mask_sitk = sitk.GetImageFromArray(mask_array)
    mask_sitk.CopyInformation(reference_image)
    sitk.WriteImage(mask_sitk, output_path)
    print(f"{description} saved to: {output_path}")

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
    print("Original mask size:", original_mask.shape)
    print("Original mask voxel count:", np.sum(original_mask))
    
    # --- PERFORM 2MM EXPANSION ---
    print(f"\nExpanding mask by {EXPANSION_MM_2}mm...")
    expanded_mask_2mm = expand_mask(original_mask, spacing, EXPANSION_MM_2)
    print(f"2mm expanded mask voxel count: {np.sum(expanded_mask_2mm)}")
    
    # Save 2mm expansion
    output_path_2mm = os.path.join(RESULT_DIR, f"expanded_mask_{EXPANSION_MM_2}mm.nii.gz")
    save_mask(expanded_mask_2mm, mask_image, output_path_2mm, f"{EXPANSION_MM_2}mm expanded mask")
    
    # --- PERFORM 4MM EXPANSION ---
    print(f"\nExpanding mask by {EXPANSION_MM_4}mm...")
    expanded_mask_4mm = expand_mask(original_mask, spacing, EXPANSION_MM_4)
    print(f"4mm expanded mask voxel count: {np.sum(expanded_mask_4mm)}")
    
    # Save 4mm expansion
    output_path_4mm = os.path.join(RESULT_DIR, f"expanded_mask_{EXPANSION_MM_4}mm.nii.gz")
    save_mask(expanded_mask_4mm, mask_image, output_path_4mm, f"{EXPANSION_MM_4}mm expanded mask")
    
    # --- COMPARISON ---
    print(f"\n=== COMPARISON ===")
    print(f"Original mask: {np.sum(original_mask)} voxels")
    print(f"2mm expansion: {np.sum(expanded_mask_2mm)} voxels (+{np.sum(expanded_mask_2mm) - np.sum(original_mask)})")
    print(f"4mm expansion: {np.sum(expanded_mask_4mm)} voxels (+{np.sum(expanded_mask_4mm) - np.sum(original_mask)})")
    
    # Check if masks are different
    if np.array_equal(expanded_mask_2mm, expanded_mask_4mm):
        print("⚠️  WARNING: 2mm and 4mm expansions are identical!")
        print("   This might indicate an issue with voxel spacing or expansion calculation.")
    else:
        diff_voxels = np.sum(expanded_mask_4mm) - np.sum(expanded_mask_2mm)
        print(f"✅ 4mm expansion has {diff_voxels} more voxels than 2mm expansion")

if __name__ == "__main__":
    main()