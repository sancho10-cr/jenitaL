import os
import tkinter as tk
from tkinter import filedialog, messagebox
import SimpleITK as sitk
import numpy as np
from skimage.segmentation import slic
from skimage import measure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt

# Function to load CT Images
def load_ct_image(file_path):
    if file_path.lower().endswith('.mhd'):
        ct_image = sitk.ReadImage(file_path)
    elif file_path.lower().endswith('.zraw'):
        # Assuming ZRAW is a raw data file format, we need to read it differently
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        ct_image = sitk.GetImageFromArray(np.frombuffer(raw_data, dtype=np.int16).reshape((512, 512, 512)))
    else:
        raise ValueError("Unsupported file format. Please select MHD or ZRAW files.")

    ct_array = sitk.GetArrayFromImage(ct_image)
    return ct_image, ct_array

# Function to convert Pixel Values to Hounsfield Units (HU)
def get_hu_values(image):
    if image.HasMetaDataKey("0028|1052") and image.HasMetaDataKey("0028|1053"):
        intercept = float(image.GetMetaData("0028|1052"))
        slope = float(image.GetMetaData("0028|1053"))
        hu_image = sitk.Cast(image, sitk.sitkFloat32)
        hu_image = hu_image * slope + intercept
        return sitk.GetArrayFromImage(hu_image)
    else:
        print("Metadata for intercept and slope not found. Skipping HU conversion.")
        return sitk.GetArrayFromImage(image)

# Function to apply Noise Reduction
def apply_noise_reduction(image):
    filtered_image = sitk.Median(image, [2, 2, 2])
    filtered_array = sitk.GetArrayFromImage(filtered_image)
    return filtered_image, filtered_array

# Function to downsample Image
def downsample_image(array, factor):
    return array[::factor, ::factor, ::factor]

# Function for Segmentation
def segment_image(filtered_array, downsample_factor):
    downsampled_array = downsample_image(filtered_array, downsample_factor)
    segments = slic(downsampled_array, n_segments=100, compactness=10)
    labels = measure.label(segments)
    return labels

# Function to calculate Tumor Volume
def calculate_tumor_volume(ct_image, labels):
    spacing = ct_image.GetSpacing()
    volume_per_voxel = np.prod(spacing)
    tumor_volume = np.sum(labels > 0) * volume_per_voxel
    tumor_volume_cc = tumor_volume / 1000
    return tumor_volume_cc

# Function for TNM Staging
def classify_tnm(volume):
    if volume <= 3:
        return 'T1'
    elif 3 < volume <= 7:
        return 'T2'
    elif 7 < volume:
        return 'T3'
    else:
        return 'T4'

# Function to extract Metadata
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

# Function to calculate Tumor Properties
def calculate_tumor_properties(ct_image, labels):
    properties = measure.regionprops(labels)
    if properties:
        tumor = properties[0]
        tumor_location = tumor.centroid
        tumor_size = tumor.major_axis_length
        tumor_perimeter = tumor.perimeter
        tumor_area = tumor.area
        return {
            'Location': tumor_location,
            'Size': tumor_size,
            'Perimeter': tumor_perimeter,
            'Area': tumor_area
        }
    else:
        return {
            'Location': 'N/A',
            'Size': 'N/A',
            'Perimeter': 'N/A',
            'Area': 'N/A'
        }

# Function to train Random Forest classifier
def train_rf_classifier(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

# Function to process CT image
def process_ct_image(file_path):
    ct_image, ct_array = load_ct_image(file_path)
    hu_array = get_hu_values(ct_image)
    filtered_image, filtered_array = apply_noise_reduction(ct_image)
    labels = segment_image(filtered_array, 2)
    tumor_volume_cc = calculate_tumor_volume(ct_image, labels)
    tumor_stage = classify_tnm(tumor_volume_cc)
    metadata = extract_metadata(ct_image)
    tumor_properties = calculate_tumor_properties(ct_image, labels)
    return tumor_volume_cc, tumor_stage, metadata, tumor_properties

# Function to load training data
def load_training_data(folder_path):
    X = []
    y = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.mhd'):
            file_path = os.path.join(folder_path, filename)
            tumor_volume_cc, _, _, _ = process_ct_image(file_path)
            X.append([tumor_volume_cc])  # Example feature, modify as needed
            # Assuming you have a label or target variable
            y.append(classify_tnm(tumor_volume_cc))  # Example label, modify as needed
    return np.array(X), np.array(y)

# Function to select and process file for testing
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("MHD files", "*.mhd"), ("ZRAW files", "*.zraw")])
    if file_path:
        try:
            tumor_volume_cc, tumor_stage, metadata, tumor_properties = process_ct_image(file_path)
            result_text = (f"Tumor Volume: {tumor_volume_cc:.2f} cc\n"
                           f"Tumor Stage: {tumor_stage}\n"
                           f"Tumor Location: {tumor_properties['Location']}\n"
                           f"Tumor Size: {tumor_properties['Size']}\n"
                           f"Tumor Perimeter: {tumor_properties['Perimeter']}\n"
                           f"Tumor Area: {tumor_properties['Area']}\n"
                           f"Metadata:\n"
                           f"Spacing: {metadata['Spacing']}\n"
                           f"Dimensions: {metadata['Dimensions']}\n"
                           f"Direction: {metadata['Direction']}\n"
                           f"Slice Thickness: {metadata['Slice Thickness']}\n"
                           f"Patient Position: {metadata['Patient Position']}\n"
                           f"Window Center: {metadata['Window Center']}\n"
                           f"Window Width: {metadata['Window Width']}\n")
            result_label.config(text=result_text)
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Building the UI using Tkinter
root = tk.Tk()
root.title("CT Image Processor")

frame = tk.Frame(root)
frame.pack(pady=20, padx=20)

train_folder_button = tk.Button(frame, text="Train Model from Folder", command=lambda: train_model_from_folder(frame))
train_folder_button.pack()

select_button = tk.Button(frame, text="Select CT image File", command=select_file)
select_button.pack()

result_label = tk.Label(frame, text="", justify=tk.LEFT, anchor="w", wraplength=600)
result_label.pack(pady=20)

def train_model_from_folder(frame):
    folder_path = filedialog.askdirectory()
    if folder_path:
        try:
            X_train, y_train = load_training_data(folder_path)
            rf_classifier = train_rf_classifier(X_train, y_train)
            messagebox.showinfo("Training Complete", "Model trained successfully using MHD files in the folder.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

root.mainloop()
