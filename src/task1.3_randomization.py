# import numpy as np
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# from scipy.ndimage import binary_dilation, distance_transform_edt
# import os
# import random

# # --- CONFIGURATION ---
# CT_PATH = "../left_knee.nii.gz"
# MASK_PATH = "../notebooks/masks/original_mask.nii.gz"
# MAX_EXPANSION_MM = 2.0  # Maximum expansion parameter (2mm)
# RANDOM_SEEDS = [42, 43]  # Generate 2 different random contours
# RESULT_DIR = "results"
# os.makedirs(RESULT_DIR, exist_ok=True)

# def expand_mask(mask_array, spacing, expansion_mm):
#     expansion_voxels = [
#         max(1, int(np.ceil(expansion_mm / s))) for s in spacing[::-1]  # (z, y, x)
#     ]
#     print(f"Expansion of {expansion_mm}mm converted to voxels: {expansion_voxels}")
    
#     structuring_element = np.ones(
#         (2 * expansion_voxels[0] + 1, 2 * expansion_voxels[1] + 1, 2 * expansion_voxels[2] + 1),
#         dtype=np.uint8
#     )
    
#     expanded_mask = binary_dilation(mask_array, structure=structuring_element).astype(np.uint8)
#     return expanded_mask

# def create_randomized_contour(original_mask, expanded_mask, spacing, random_seed):
#     random.seed(random_seed)
#     np.random.seed(random_seed)
    
#     expansion_zone = np.logical_and(expanded_mask, np.logical_not(original_mask))
#     distance_from_original = distance_transform_edt(np.logical_not(original_mask), sampling=spacing[::-1])
    
#     randomized_mask = np.copy(original_mask)
    
#     for z in range(original_mask.shape[0]):
#         for y in range(original_mask.shape[1]):
#             for x in range(original_mask.shape[2]):
#                 if expansion_zone[z, y, x]:
#                     dist = distance_from_original[z, y, x]
#                     max_dist = MAX_EXPANSION_MM
#                     probability = 1 - (dist / max_dist) ** 1.5
#                     probability = max(0.0, min(1.0, probability))  # Clamp to [0, 1]

#                     if random.random() < probability:
#                         randomized_mask[z, y, x] = 1
#     return randomized_mask

# def validate_randomized_mask(original_mask, expanded_mask, randomized_mask, index=1):
#     violations = []

#     if not np.all((original_mask == 1) <= (randomized_mask == 1)):
#         violations.append("Original mask not fully included in randomized mask.")
    
#     if not np.all((randomized_mask == 1) <= (expanded_mask == 1)):
#         violations.append("Randomized mask exceeds expanded mask boundaries.")
    
#     if np.sum(original_mask) > np.sum(randomized_mask):
#         violations.append("Randomized mask smaller than original.")
    
#     if np.sum(randomized_mask) > np.sum(expanded_mask):
#         violations.append("Randomized mask larger than expanded.")

#     if not violations:
#         print(f"Validation passed for randomized mask {index}")
#     else:
#         print(f"WARNING: Validation failed for randomized mask {index}:")
#         for v in violations:
#             print(" -", v)

# def main():
#     print("Loading images...")
#     ct_image = sitk.ReadImage(CT_PATH)
#     mask_image = sitk.ReadImage(MASK_PATH)
    
#     image_array = sitk.GetArrayFromImage(ct_image)
#     original_mask = sitk.GetArrayFromImage(mask_image).astype(np.uint8)
    
#     spacing = ct_image.GetSpacing()  # (x, y, z)
#     print("Voxel spacing (mm):", spacing)
    
#     print(f"Expanding mask by {MAX_EXPANSION_MM}mm...")
#     expanded_mask = expand_mask(original_mask, spacing, MAX_EXPANSION_MM)
    
#     # Save expanded mask
#     expanded_mask_sitk = sitk.GetImageFromArray(expanded_mask)
#     expanded_mask_sitk.CopyInformation(mask_image)
#     expanded_path = os.path.join(RESULT_DIR, f"expanded_mask_{MAX_EXPANSION_MM}mm.nii.gz")
#     sitk.WriteImage(expanded_mask_sitk, expanded_path)
#     print(f"Expanded mask saved to: {expanded_path}")
    
#     for i, seed in enumerate(RANDOM_SEEDS):
#         print(f"Creating randomized contour {i+1} with seed {seed}...")
#         randomized_mask = create_randomized_contour(original_mask, expanded_mask, spacing, seed)

#         randomized_mask_sitk = sitk.GetImageFromArray(randomized_mask)
#         randomized_mask_sitk.CopyInformation(mask_image)
#         randomized_path = os.path.join(RESULT_DIR, f"randomized_mask_{i+1}_seed{seed}.nii.gz")
#         sitk.WriteImage(randomized_mask_sitk, randomized_path)
#         print(f"Randomized mask saved to: {randomized_path}")

#         validate_randomized_mask(original_mask, expanded_mask, randomized_mask, index=i+1)

# if __name__ == "__main__":
#     main()
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
RANDOM_SEEDS = [42, 43]  # Generate 2 different random contours
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

def expand_mask(mask_array, spacing, expansion_mm):
    """
    Expand mask by specified distance in mm
    """
    # Convert mm to voxels - spacing is (x, y, z), array is (z, y, x)
    expansion_voxels = [
        max(1, int(np.ceil(expansion_mm / spacing[2]))),  # z direction
        max(1, int(np.ceil(expansion_mm / spacing[1]))),  # y direction  
        max(1, int(np.ceil(expansion_mm / spacing[0])))   # x direction
    ]
    print(f"Expansion of {expansion_mm}mm converted to voxels: {expansion_voxels}")
    
    structuring_element = np.ones(
        (2 * expansion_voxels[0] + 1, 2 * expansion_voxels[1] + 1, 2 * expansion_voxels[2] + 1),
        dtype=np.uint8
    )
    
    expanded_mask = binary_dilation(mask_array, structure=structuring_element).astype(np.uint8)
    return expanded_mask

def create_randomized_contour(original_mask, expanded_mask, spacing, random_seed, max_expansion_mm):
    """
    Create randomized contour between original and expanded mask
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Find expansion zone (area between original and expanded)
    expansion_zone = np.logical_and(expanded_mask, np.logical_not(original_mask))
    
    # Calculate distance from original mask boundary
    # Note: spacing needs to be in (z, y, x) order for distance_transform_edt
    distance_from_original = distance_transform_edt(
        np.logical_not(original_mask), 
        sampling=[spacing[2], spacing[1], spacing[0]]  # Convert (x,y,z) to (z,y,x)
    )
    
    # Start with original mask (ensures we never shrink below original)
    randomized_mask = np.copy(original_mask)
    
    # Vectorized approach - much faster than triple loop
    expansion_indices = np.where(expansion_zone)
    
    for i in range(len(expansion_indices[0])):
        z, y, x = expansion_indices[0][i], expansion_indices[1][i], expansion_indices[2][i]
        
        dist = distance_from_original[z, y, x]
        
        # Ensure we don't exceed max expansion
        if dist <= max_expansion_mm:
            # Probability decreases with distance from original boundary
            probability = 1 - (dist / max_expansion_mm) ** 1.5
            probability = max(0.0, min(1.0, probability))  # Clamp to [0, 1]

            if random.random() < probability:
                randomized_mask[z, y, x] = 1
    
    return randomized_mask

def validate_randomized_mask(original_mask, expanded_mask, randomized_mask, max_expansion_mm, spacing, index=1):
    """
    Validate that randomized mask meets all requirements
    """
    violations = []

    # Check 1: Original mask fully included in randomized mask
    if not np.all((original_mask == 1) <= (randomized_mask == 1)):
        violations.append("Original mask not fully included in randomized mask.")
    
    # Check 2: Randomized mask doesn't exceed expanded mask
    if not np.all((randomized_mask == 1) <= (expanded_mask == 1)):
        violations.append("Randomized mask exceeds expanded mask boundaries.")
    
    # Check 3: Randomized mask not smaller than original
    if np.sum(original_mask) > np.sum(randomized_mask):
        violations.append("Randomized mask smaller than original.")
    
    # Check 4: Randomized mask not larger than expanded
    if np.sum(randomized_mask) > np.sum(expanded_mask):
        violations.append("Randomized mask larger than expanded.")
    
    # Check 5: Distance constraint - no point should exceed max_expansion_mm from original
    distance_from_original = distance_transform_edt(
        np.logical_not(original_mask), 
        sampling=[spacing[2], spacing[1], spacing[0]]
    )
    randomized_only = np.logical_and(randomized_mask, np.logical_not(original_mask))
    max_distance = np.max(distance_from_original[randomized_only]) if np.any(randomized_only) else 0
    
    if max_distance > max_expansion_mm + 0.01:  # Small tolerance for floating point
        violations.append(f"Randomized mask exceeds {max_expansion_mm}mm limit (max distance: {max_distance:.2f}mm)")

    if not violations:
        print(f"✅ Validation passed for randomized mask {index}")
        print(f"   Original voxels: {np.sum(original_mask)}")
        print(f"   Randomized voxels: {np.sum(randomized_mask)}")
        print(f"   Expanded voxels: {np.sum(expanded_mask)}")
    else:
        print(f"❌ WARNING: Validation failed for randomized mask {index}:")
        for v in violations:
            print(f"   - {v}")

def main():
    print("Loading images...")
    ct_image = sitk.ReadImage(CT_PATH)
    mask_image = sitk.ReadImage(MASK_PATH)
    
    image_array = sitk.GetArrayFromImage(ct_image)
    original_mask = sitk.GetArrayFromImage(mask_image).astype(np.uint8)
    
    spacing = ct_image.GetSpacing()  # (x, y, z)
    print("Voxel spacing (mm):", spacing)
    
    print(f"Expanding mask by {MAX_EXPANSION_MM}mm...")
    expanded_mask = expand_mask(original_mask, spacing, MAX_EXPANSION_MM)
    
    # Save expanded mask
    expanded_mask_sitk = sitk.GetImageFromArray(expanded_mask)
    expanded_mask_sitk.CopyInformation(mask_image)
    expanded_path = os.path.join(RESULT_DIR, f"expanded_mask_{MAX_EXPANSION_MM}mm.nii.gz")
    sitk.WriteImage(expanded_mask_sitk, expanded_path)
    print(f"Expanded mask saved to: {expanded_path}")
    
    # Generate randomized masks
    for i, seed in enumerate(RANDOM_SEEDS):
        print(f"\nCreating randomized contour {i+1} with seed {seed}...")
        randomized_mask = create_randomized_contour(
            original_mask, expanded_mask, spacing, seed, MAX_EXPANSION_MM
        )

        # Save randomized mask
        randomized_mask_sitk = sitk.GetImageFromArray(randomized_mask)
        randomized_mask_sitk.CopyInformation(mask_image)
        randomized_path = os.path.join(RESULT_DIR, f"randomized_mask_{i+1}_seed{seed}.nii.gz")
        sitk.WriteImage(randomized_mask_sitk, randomized_path)
        print(f"Randomized mask saved to: {randomized_path}")

        # Validate the result
        validate_randomized_mask(
            original_mask, expanded_mask, randomized_mask, 
            MAX_EXPANSION_MM, spacing, index=i+1
        )

if __name__ == "__main__":
    main()