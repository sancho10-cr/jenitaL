import tkinter as tk
from tkinter import filedialog, messagebox
import SimpleITK as sitk
import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load CT Images
def load_ct_image(file_path):
    if file_path.lower().endswith('.mhd'):
        ct_image = sitk.ReadImage(file_path)
    else:
        raise ValueError("Unsupported file format. Please select MHD files.")
    
    ct_array = sitk.GetArrayFromImage(ct_image)
    return ct_image, ct_array

# Convert Pixel Values to Hounsfield Units (HU)
def get_hu_values(image):
    print("Raw image sample data (before HU conversion):")
    print(sitk.GetArrayFromImage(image)[:5, :5, :5])  # Print a small sample of raw data
    
    if image.HasMetaDataKey("0028|1052") and image.HasMetaDataKey("0028|1053"):
        intercept = float(image.GetMetaData("0028|1052"))
        slope = float(image.GetMetaData("0028|1053"))
    else:
        print("Metadata for intercept and slope not found. Using default values.")
        intercept = -1024  # Default intercept
        slope = 1  # Default slope

    hu_image = sitk.Cast(image, sitk.sitkFloat32)
    hu_image = hu_image * slope + intercept
    hu_array = sitk.GetArrayFromImage(hu_image)
    
    print(f"Intercept: {intercept}, Slope: {slope}")
    return hu_array

# Apply Noise Reduction
def apply_noise_reduction(image):
    filtered_image = sitk.Median(image, [2, 2, 2])
    filtered_array = sitk.GetArrayFromImage(filtered_image)
    return filtered_image, filtered_array

# Segment Lung Region
def segment_lung(hu_array):
    # Print minimum and maximum HU values
    print(f"Min HU value: {hu_array.min()}, Max HU value: {hu_array.max()}")
    
    # Threshold the image to get the lung region
    lung_mask = (hu_array > -1000) & (hu_array < -300)
    
    if lung_mask.sum() == 0:
        raise ValueError("Lung mask is empty. Check the HU range for segmentation.")
    
    # Label the connected components
    labeled_mask, num_labels = measure.label(lung_mask, connectivity=3, return_num=True)
    regions = measure.regionprops(labeled_mask)
    
    print(f"Number of regions found: {num_labels}")

    # Select the two largest regions (assuming they are the lungs)
    if len(regions) > 2:
        regions = sorted(regions, key=lambda x: x.area, reverse=True)[:2]
    
    if len(regions) == 0:
        raise ValueError("No valid lung regions found.")
    
    lung_mask = np.zeros_like(labeled_mask, dtype=bool)
    for region in regions:
        lung_mask[labeled_mask == region.label] = True
    
    return lung_mask

# Segment Tumor within Lung
def segment_tumor(lung_mask, hu_array):
    if lung_mask.sum() == 0:
        raise ValueError("Lung mask is empty. Cannot segment tumor without lung mask.")
    
    # Apply thresholding to segment the tumor within the lung
    threshold_value = threshold_otsu(hu_array[lung_mask])
    tumor_mask = (hu_array > threshold_value) & lung_mask

    # Label the connected components
    labeled_mask, num_labels = measure.label(tumor_mask, connectivity=3, return_num=True)
    regions = measure.regionprops(labeled_mask)
    
    print(f"Number of tumor regions found: {num_labels}")

    # Create an empty mask
    tumor_mask = np.zeros_like(tumor_mask, dtype=bool)
    
    for region in regions:
        if region.area > 1000:
            tumor_mask[labeled_mask == region.label] = True

    return tumor_mask

# Calculate Tumor Volume
def calculate_tumor_volume(ct_image, tumor_mask):
    spacing = ct_image.GetSpacing()
    volume_per_voxel = np.prod(spacing)
    tumor_volume = np.sum(tumor_mask) * volume_per_voxel
    tumor_volume_cc = tumor_volume / 1000
    return tumor_volume_cc

# Extract Metadata
def extract_metadata(image):
    metadata = {}
    metadata['Spacing'] = image.GetSpacing()
    metadata['Dimensions'] = image.GetSize()
    metadata['Direction'] = image.GetDirection()
    metadata['Slice Thickness'] = image.GetMetaData('0018|0050') if image.HasMetaDataKey('0018|0050') else 'N/A'
    metadata['Patient Position'] = image.GetMetaData('0018|5100') if image.HasMetaDataKey('0018|5100') else 'N/A'
    metadata['Window Center'] = image.GetMetaData('0028|1050') if image.HasMetaDataKey('0028|1050') else 'N/A'
    metadata['Window Width'] = image.GetMetaData('0028|1051') if image.HasMetaDataKey('0028|1051') else 'N/A'
    return metadata

# Display 3D Image
def display_3d_image(lung_mask, tumor_mask):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract surfaces of the lung
    lung_verts, lung_faces, _, _ = measure.marching_cubes(lung_mask, level=0)
    lung_mesh = Poly3DCollection(lung_verts[lung_faces], alpha=0.2, color='blue')
    ax.add_collection3d(lung_mesh)

    # Extract surfaces of the tumor
    tumor_verts, tumor_faces, _, _ = measure.marching_cubes(tumor_mask, level=0)
    tumor_mesh = Poly3DCollection(tumor_verts[tumor_faces], alpha=0.6, color='red')
    ax.add_collection3d(tumor_mesh)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    ax.set_xlim(0, lung_mask.shape[2])
    ax.set_ylim(0, lung_mask.shape[1])
    ax.set_zlim(0, lung_mask.shape[0])

    plt.title('3D Visualization of Lung and Tumor')
    plt.tight_layout()
    plt.show()

# Select and Process File
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("MHD files", "*.mhd")])
    if file_path:
        try:
            ct_image, ct_array = load_ct_image(file_path)
            hu_array = get_hu_values(ct_image)
            print(f"HU array shape: {hu_array.shape}")
            filtered_image, filtered_array = apply_noise_reduction(ct_image)
            lung_mask = segment_lung(hu_array)
            print(f"Lung mask shape: {lung_mask.shape}")
            tumor_mask = segment_tumor(lung_mask, hu_array)
            tumor_volume_cc = calculate_tumor_volume(ct_image, tumor_mask)
            metadata = extract_metadata(ct_image)
            result_text = (f"Tumor Volume: {tumor_volume_cc:.2f} cc\n"
                           f"Metadata:\n"
                           f"Spacing: {metadata['Spacing']}\n"
                           f"Dimensions: {metadata['Dimensions']}\n"
                           f"Direction: {metadata['Direction']}\n"
                           f"Slice Thickness: {metadata['Slice Thickness']}\n"
                           f"Patient Position: {metadata['Patient Position']}\n"
                           f"Window Center: {metadata['Window Center']}\n"
                           f"Window Width: {metadata['Window Width']}\n")
            result_label.config(text=result_text)

            # Display 3D visualization of lung and tumor
            display_3d_image(lung_mask, tumor_mask)

        except Exception as e:
            messagebox.showerror("Error", str(e))

# Building the UI using Tkinter
root = tk.Tk()
root.title("Lung Tumor Segmentation and Visualization")

frame = tk.Frame(root)
frame.pack(pady=20, padx=20)

select_button = tk.Button(frame, text="Select CT image File", command=select_file)
select_button.pack()

result_label = tk.Label(frame, text="", justify=tk.LEFT, anchor="w", wraplength=600)
result_label.pack(pady=20)

root.mainloop()
