import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label
import os

def load_mask(path):
    mask = sitk.ReadImage(path)
    spacing = mask.GetSpacing()
    data = sitk.GetArrayFromImage(mask)  # z, y, x
    return data, spacing

def find_tibia_component(mask_data):
    labeled, num_features = label(mask_data)
    z_sums = [np.mean(np.argwhere(labeled == i + 1)[:, 0]) for i in range(num_features)]
    tibia_label = np.argmax(z_sums) + 1  # component with highest mean z is likely tibia
    tibia_mask = (labeled == tibia_label).astype(np.uint8)
    return tibia_mask

def find_lowest_medial_lateral_points(tibia_mask, spacing):
    coords = np.argwhere(tibia_mask > 0)  # (z, y, x)
    
    if coords.size == 0:
        return None, None

    # Convert to (x, y, z)
    coords_mm = coords[:, [2, 1, 0]] * spacing

    x_mean = np.mean(coords_mm[:, 0])
    medial = coords_mm[coords_mm[:, 0] < x_mean]
    lateral = coords_mm[coords_mm[:, 0] >= x_mean]

    if medial.size == 0 or lateral.size == 0:
        return None, None

    # Find point with lowest z (most inferior)
    medial_lowest = medial[np.argmin(medial[:, 2])]
    lateral_lowest = lateral[np.argmin(lateral[:, 2])]

    return medial_lowest.tolist(), lateral_lowest.tolist()

def process_mask(path):
    data, spacing = load_mask(path)
    tibia_mask = find_tibia_component(data)
    medial, lateral = find_lowest_medial_lateral_points(tibia_mask, spacing)
    return medial, lateral

# File paths (update if needed)
mask_paths = {
    "Original": "results/original_mask.nii.gz",
    "2mm": "results/expanded_mask_2.0mm.nii.gz",
    "4mm": "results/expanded_mask_4.0mm.nii.gz",
    "Random1": "results/randomized_mask_1_seed42.nii.gz",
    "Random2": "results/randomized_mask_2_seed43.nii.gz",
}

# Run on all masks
results = {}
for name, path in mask_paths.items():
    if os.path.exists(path):
        print(f"Processing: {name}")
        medial, lateral = process_mask(path)
        results[name] = {
            "Medial Lowest": medial,
            "Lateral Lowest": lateral
        }
    else:
        results[name] = "File not found"

# Display
for k, v in results.items():
    print(f"\n== {k} ==")
    print(v)
