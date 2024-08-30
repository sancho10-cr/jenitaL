import tkinter as tk
from tkinter import filedialog, messagebox
import SimpleITK as sitk
import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Step 1: Load CT Images
def load_ct_image(file_path):
    if file_path.lower().endswith('.mhd'):
        ct_image = sitk.ReadImage(file_path)
    else:
        raise ValueError("")
    
    ct_array = sitk.GetArrayFromImage(ct_image)
    return ct_image, ct_array

# Step 1.2: Convert Pixel Values to Hounsfield Units (HU)
def get_hu_values(image):
    if image.HasMetaDataKey("0028|1052") and image.HasMetaDataKey("0028|1053"):
        intercept = float(image.GetMetaData("0028|1052"))
        slope = float(image.GetMetaData("0028|1053"))
        hu_image = sitk.Cast(image, sitk.sitkFloat32)
        hu_image = hu_image * slope + intercept
        return sitk.GetArrayFromImage(hu_image)
    else:
        print("Metadata for intercept and slope not found. Using default values.")
        intercept = -1024  # Default intercept
        slope = 1  # Default slope
        hu_array = sitk.GetArrayFromImage(image)
        hu_array = hu_array * slope + intercept
        return hu_array

# Step 1.3: Apply Noise Reduction
def apply_noise_reduction(image):
    filtered_image = sitk.Median(image, [2, 2, 2])
    filtered_array = sitk.GetArrayFromImage(filtered_image)
    return filtered_image, filtered_array

# Step 2: Segmentation (adapted for 3D)
def segment_image(filtered_array):
    # Apply Otsu's thresholding to segment the tumor region in 3D
    threshold_value = threshold_otsu(filtered_array)
    tumor_mask = filtered_array > threshold_value

    # Check the segmented region
    print(f"Threshold value: {threshold_value}")
    print(f"Segmented volume shape: {tumor_mask.shape}")
    print(f"Non-zero elements in tumor mask: {np.count_nonzero(tumor_mask)}")

    # Label the connected components in the 3D mask
    labels = measure.label(tumor_mask, connectivity=3)
    return labels

# Step 3: Tumor Volume Calculation
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

# Step 5: Extract Metadata
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

# Step 6: Calculate Tumor Size, Perimeter, and Area
def calculate_tumor_properties(labels, spacing):
    regions = measure.regionprops(labels)

    if len(regions) > 0:
        region = regions[0]
        tumor_area = region.area * np.prod(spacing[:2])
        tumor_size = region.major_axis_length * spacing[0]  # Assuming spacing[0] == spacing[1]

        return tumor_size, tumor_area
    else:
        return 0, 0

# Custom function to calculate 3D perimeter
def calculate_3d_perimeter(labels):
    perimeter = 0
    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = labels == label
        padded_mask = np.pad(label_mask, pad_width=1, mode='constant', constant_values=0)
        for dim in range(3):
            diff = np.diff(padded_mask, axis=dim)
            perimeter += np.sum(np.abs(diff))
    return perimeter

# Main function to process CT images
def process_ct_image(file_path):
    ct_image, ct_array = load_ct_image(file_path)
    hu_array = get_hu_values(ct_image)
    filtered_image, filtered_array = apply_noise_reduction(ct_image)
    labels = segment_image(filtered_array)
    tumor_volume_cc = calculate_tumor_volume(ct_image, labels)
    tumor_stage = classify_tnm(tumor_volume_cc)
    metadata = extract_metadata(ct_image)
    tumor_size, tumor_area = calculate_tumor_properties(labels, ct_image.GetSpacing())
    tumor_perimeter = calculate_3d_perimeter(labels)
    
    return tumor_volume_cc, tumor_stage, metadata, labels, tumor_size, tumor_perimeter, tumor_area

# Function to display 3D image and results
def display_3d_image(labels):
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract surfaces of the labeled tumor
        verts, faces, _, _ = measure.marching_cubes(labels, level=0, step_size=1)

        if len(verts) == 0:
            raise RuntimeError("No surfaces found. The tumor might be too small or not segmented properly.")

        # Display 3D tumor surface
        mesh = Poly3DCollection(verts[faces], alpha=0.4)
        face_color = [0.8, 0.1, 0.1]  # Red color for the tumor
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        ax.set_xlim(0, labels.shape[2])
        ax.set_ylim(0, labels.shape[1])
        ax.set_zlim(0, labels.shape[0])

        plt.title('3D Visualization of Lung Tumor')
        plt.tight_layout()
        plt.show()
    except RuntimeError as e:
        messagebox.showerror("Error", f"No surface found: {str(e)}")

# Function to select and process file
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("MHD files", "*.mhd")])
    if file_path:
        try:
            tumor_volume_cc, tumor_stage, metadata, labels, tumor_size, tumor_perimeter, tumor_area = process_ct_image(file_path)
            result_text = (f"Tumor Volume: {tumor_volume_cc:.2f} cc\n"
                           f"Tumor Size: {tumor_size:.2f} mm\n"
                           f"Tumor Perimeter: {tumor_perimeter:.2f} mm\n"
                           f"Tumor Area: {tumor_area:.2f} mm²\n"
                           f"Metadata:\n"
                           f"Spacing: {metadata['Spacing']}\n"
                           f"Dimensions: {metadata['Dimensions']}\n"
                           f"Direction: {metadata['Direction']}\n"
                           f"Slice Thickness: {metadata['Slice Thickness']}\n"
                           f"Patient Position: {metadata['Patient Position']}\n"
                           f"Window Center: {metadata['Window Center']}\n"
                           f"Window Width: {metadata['Window Width']}\n")
            result_label.config(text=result_text)

            # Display 3D tumor visualization
            display_3d_image(labels)

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
