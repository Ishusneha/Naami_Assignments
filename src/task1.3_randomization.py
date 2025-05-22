
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, distance_transform_edt
import os
import random
# --- CONFIGURATION ---
CT_PATH = "../left_knee.nii.gz"
MASK_PATH = "../notebooks/masks/original_mask.nii.gz"
MAX_EXPANSION_MM = 2.0  # Maximum expansion parameter (2mm)
RANDOM_SEED = 42        # Random seed parameter for reproducibility
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
    structuring_element = np.ones(
        (2 * expansion_voxels[0] + 1, 2 * expansion_voxels[1] + 1, 2 * expansion_voxels[2] + 1),
        dtype=np.uint8
    )
    
    # Perform binary dilation to expand the mask
    expanded_mask = binary_dilation(mask_array, structure=structuring_element).astype(np.uint8)
    
    return expanded_mask

def create_randomized_contour(original_mask, expanded_mask, spacing, random_seed=42):
    """
    Creates a randomized contour between the original mask and the expanded mask.
    The randomized contour:
    - Will not exceed the expanded mask (max 2mm expansion)
    - Will not shrink below the original mask
    - Will have random variations between the original and expanded boundaries
    
    Args:
        original_mask: The original segmentation mask
        expanded_mask: The uniformly expanded mask (e.g., 2mm expansion)
        spacing: The voxel spacing (x, y, z) in mm
        random_seed: Seed for random number generation
        
    Returns:
        A randomized contour mask
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create a mask of the expansion zone (area between original and expanded masks)
    expansion_zone = np.logical_and(expanded_mask, np.logical_not(original_mask))
    
    # Calculate the distance transform from the original mask boundary
    # This gives us the distance (in voxels) from each point to the nearest original mask boundary
    distance_from_original = distance_transform_edt(np.logical_not(original_mask), sampling=spacing[::-1])
    
    # In the expansion zone, mask will only include voxels based on random threshold
    randomized_mask = original_mask.copy()
    
    # For each voxel in the expansion zone, decide whether to include it based on its distance
    # The closer to the original boundary, the higher probability of inclusion
    for z in range(original_mask.shape[0]):
        for y in range(original_mask.shape[1]):
            for x in range(original_mask.shape[2]):
                if expansion_zone[z, y, x]:
                    # Get the distance from this voxel to the original mask boundary
                    dist = distance_from_original[z, y, x]
                    
                    # Calculate probability based on distance (inverse relationship)
                    # Points closer to original boundary have higher probability of inclusion
                    # We use a non-linear function to create more natural-looking contours
                    max_dist = MAX_EXPANSION_MM  # Maximum expansion in mm
                    
                    # Calculate probability - higher probability for voxels closer to original boundary
                    # This creates a natural falloff as we get further from the original boundary
                    probability = 1 - (dist / max_dist) ** 1.5
                    
                    # Apply randomization
                    if random.random() < probability:
                        randomized_mask[z, y, x] = 1
    
    return randomized_mask

def main():
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
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
    print(f"Expanding mask by {MAX_EXPANSION_MM}mm...")
    expanded_mask = expand_mask(original_mask, spacing, MAX_EXPANSION_MM)
    
    # --- CREATE RANDOMIZED CONTOUR ---
    print("Creating randomized contour...")
    randomized_mask = create_randomized_contour(original_mask, expanded_mask, spacing, RANDOM_SEED)
    
    # --- SAVE MASKS ---
    print("Saving masks...")
    
    # Save expanded mask
    expanded_mask_sitk = sitk.GetImageFromArray(expanded_mask)
    expanded_mask_sitk.CopyInformation(mask_image)
    expanded_path = os.path.join(RESULT_DIR, f"expanded_mask_{MAX_EXPANSION_MM}mm.nii.gz")
    sitk.WriteImage(expanded_mask_sitk, expanded_path)
    
    # Save randomized mask
    randomized_mask_sitk = sitk.GetImageFromArray(randomized_mask)
    randomized_mask_sitk.CopyInformation(mask_image)
    randomized_path = os.path.join(RESULT_DIR, f"randomized_mask_seed{RANDOM_SEED}.nii.gz")
    sitk.WriteImage(randomized_mask_sitk, randomized_path)
    
    print(f"Expanded mask saved to: {expanded_path}")
    print(f"Randomized mask saved to: {randomized_path}")
    
    # --- LOCATE NONZERO SLICES FOR VISUALIZATION ---
    slices_with_mask = [i for i in range(randomized_mask.shape[0]) if np.any(randomized_mask[i])]
    
    if not slices_with_mask:
        raise ValueError("Randomized mask is empty!")
    
    # Use center of relevant slices for visualization
    slice_index = slices_with_mask[len(slices_with_mask) // 2]
    print("Visualizing slice index:", slice_index)
    
    # --- VALIDATE RESULTS ---
    # Check if randomized mask is within bounds
    original_count = np.sum(original_mask)
    expanded_count = np.sum(expanded_mask)
    randomized_count = np.sum(randomized_mask)
    
    # Verify the randomized mask lies between original and expanded
    meets_criteria = (
        randomized_count >= original_count and 
        randomized_count <= expanded_count and
        np.all(np.logical_or(np.logical_not(randomized_mask), expanded_mask)) and
        np.all(np.logical_or(original_mask, np.logical_not(randomized_mask)))
    )
    
    if meets_criteria:
        print("Validation: Randomized mask successfully meets all criteria!")
    else:
        print("WARNING: Randomized mask may not meet all criteria!")
if __name__ == "__main__":
    main()