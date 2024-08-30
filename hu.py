import os
import SimpleITK as sitk
import numpy as np
from skimage.segmentation import slic
from skimage import measure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import filedialog, messagebox

def get_hu_values(image):
    # Default intercept and slope
    default_intercept = 0.0
    default_slope = 1.0
    
    # Check if metadata keys for intercept and slope exist
    if image.HasMetaDataKey("0028|1052") and image.HasMetaDataKey("0028|1053"):
        intercept = float(image.GetMetaData("0028|1052"))
        slope = float(image.GetMetaData("0028|1053"))
    else:
        intercept = default_intercept
        slope = default_slope
        print("Metadata for intercept and slope not found. Using default values.")

    # Convert the image to HU
    hu_image = sitk.Cast(image, sitk.sitkFloat32)
    hu_image = hu_image * slope + intercept
    
    return sitk.GetArrayFromImage(hu_image)

def load_ct_image(file_path):
    ct_image = sitk.ReadImage(file_path)
    ct_array = sitk.GetArrayFromImage(ct_image)
    return ct_image, ct_array

def apply_noise_reduction(image):
    filtered_image = sitk.Median(image, [2, 2, 2])
    filtered_array = sitk.GetArrayFromImage(filtered_image)
    return filtered_image, filtered_array

def downsample_image(array, factor):
    return array[::factor, ::factor, ::factor]

def segment_image(filtered_array, downsample_factor):
    downsampled_array = downsample_image(filtered_array, downsample_factor)
    segments = slic(downsampled_array, n_segments=100, compactness=10)
    labels = measure.label(segments)
    return labels

def calculate_tumor_volume(ct_image, labels):
    spacing = ct_image.GetSpacing()
    volume_per_voxel = np.prod(spacing)
    tumor_volume = np.sum(labels > 0) * volume_per_voxel
    tumor_volume_cc = tumor_volume / 1000
    return tumor_volume_cc

# Step 4: TNM Staging
def classify_tnm(size):
    if size <= 15:
        return 'T1'
    elif 15 < size <= 21:
        return 'T2'
    elif 21 < size <= 28:
        return 'T3'
    else:
        return 'T4'

def extract_metadata(image):
    metadata = {}
    metadata['Spacing'] = image.GetSpacing()
    metadata['Dimensions'] = image.GetSize()
    metadata['Direction'] = image.GetDirection()
    if image.HasMetaDataKey('0018|0050'):
        metadata['Slice Thickness'] = image.GetMetaData('0018|0050')
    else:
        metadata['Slice Thickness'] = 'N/A'
    if image.HasMetaDataKey('0018|5100'):
        metadata['Patient Position'] = image.GetMetaData('0018|5100')
    else:
        metadata['Patient Position'] = 'N/A'
    if image.HasMetaDataKey('0028|1050'):
        metadata['Window Center'] = image.GetMetaData('0028|1050')
    else:
        metadata['Window Center'] = 'N/A'
    if image.HasMetaDataKey('0028|1051'):
        metadata['Window Width'] = image.GetMetaData('0028|1051')
    else:
        metadata['Window Width'] = 'N/A'

    return metadata

def tumor_properties(labels):
    properties = measure.regionprops(labels)
    if properties:
        largest_tumor = max(properties, key=lambda x: x.area)
        tumor_location = largest_tumor.centroid
        tumor_size = largest_tumor.major_axis_length, largest_tumor.minor_axis_length
        tumor_perimeter = largest_tumor.perimeter
        tumor_area = largest_tumor.area
        return tumor_location, tumor_size, tumor_perimeter, tumor_area
    else:
        return None, None, None, None

def main():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(filetypes=[("MHD files", "*.mhd")])
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return
    
    ct_image, ct_array = load_ct_image(file_path)
    hu_array = get_hu_values(ct_image)
    filtered_image, filtered_array = apply_noise_reduction(ct_image)
    labels = segment_image(filtered_array, 2)
    tumor_volume_cc = calculate_tumor_volume(ct_image, labels)
    tumor_stage = classify_tnm(tumor_volume_cc)
    metadata = extract_metadata(ct_image)
    tumor_location, tumor_size, tumor_perimeter, tumor_area = tumor_properties(labels)
    
    print(f"Tumor Volume: {tumor_volume_cc:.2f} cc")
    print(f"Tumor Stage: {tumor_stage}")
    print(f"Tumor Location: {tumor_location}")
    print(f"Tumor Size: {tumor_size}")
    print(f"Tumor Perimeter: {tumor_perimeter}")
    print(f"Tumor Area: {tumor_area}")
    print("Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
