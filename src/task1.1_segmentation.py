import nibabel as nib
import numpy as np
from skimage import morphology, measure
from scipy.ndimage import binary_closing, binary_opening
import os

def load_nii(path):
    """Loads a NIfTI image and returns image data and affine matrix"""
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header

def save_nii(data, affine, header, path):
    """Saves image data to a NIfTI file"""
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine, header), path)

def segment_bone(ct_volume, threshold=300):
    """
    Segments bone based on intensity thresholding.
    Typical CT Hounsfield values: bone > 300 HU
    """
    bone_mask = ct_volume > threshold  # bone = high intensity
    bone_mask = morphology.remove_small_objects(bone_mask.astype(bool), min_size=1000)
    bone_mask = morphology.remove_small_holes(bone_mask, area_threshold=1000)
    return bone_mask.astype(np.uint8)

def keep_largest_components(mask, num_components=2):
    """Keeps the largest N connected components — expected femur & tibia"""
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    props = sorted(props, key=lambda x: x.area, reverse=True)
    output_mask = np.zeros_like(mask)

    for i in range(min(num_components, len(props))):
        output_mask[labeled == props[i].label] = 1

    return output_mask

if __name__ == "__main__":
    # Step 1: Load CT scan
    input_path = "left_knee.nii.gz"
    ct_volume, affine, header = load_nii(input_path)

    # Step 2: Segment bone
    raw_bone_mask = segment_bone(ct_volume, threshold=300)

    # Step 3: Keep only femur & tibia
    final_bone_mask = keep_largest_components(raw_bone_mask, num_components=2)

    # Step 4: Save result
    os.makedirs("masks", exist_ok=True)
    output_path = "masks/original_mask.nii.gz"
    save_nii(final_bone_mask, affine, header, output_path)

    print("Segmentation complete. Mask saved at:", output_path)

