import os
import tkinter as tk
from tkinter import filedialog, messagebox
import SimpleITK as sitk
import numpy as np
from skimage.segmentation import slic
from skimage import measure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt

# Step 1: Load CT Images
def load_ct_image(file_path):
    if file_path.lower().endswith('.mhd'):
        ct_image = sitk.ReadImage(file_path)
    elif file_path.lower().endswith('.zraw'):
        # Assuming ZRAW is a raw data file format, we need to read it differently
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        ct_image = sitk.GetImageFromArray(np.frombuffer(raw_data, dtype=np.int16).reshape((512, 512, 512)))
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
        print("Metadata for intercept and slope not found. Skipping HU conversion.")
        return sitk.GetArrayFromImage(image)

# Step 1.3: Apply Noise Reduction
def apply_noise_reduction(image):
    filtered_image = sitk.Median(image, [2, 2, 2])
    filtered_array = sitk.GetArrayFromImage(filtered_image)
    return filtered_image, filtered_array

# Step 1.4: Downsample Image
def downsample_image(array, factor):
    return array[::factor, ::factor, ::factor]

# Step 2: Segmentation
def segment_image(filtered_array, downsample_factor):
    downsampled_array = downsample_image(filtered_array, downsample_factor)
    segments = slic(downsampled_array, n_segments=100, compactness=10)
    labels = measure.label(segments)
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

# Additional Feature Calculations
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

# Machine Learning classification 
def train_and_evaluate_ml_classifiers(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_rf)
    classification_rep = classification_report(y_test, y_pred_rf)
    return accuracy, classification_rep

# Main function to process CT images
def process_ct_image(file_path):
    ct_image, ct_array = load_ct_image(file_path)
    hu_array = get_hu_values(ct_image)
    filtered_image, filtered_array = apply_noise_reduction(ct_image)
    labels = segment_image(filtered_array, 2)
    tumor_volume_cc = calculate_tumor_volume(ct_image, labels)
    tumor_stage = classify_tnm(tumor_volume_cc)
    metadata = extract_metadata(ct_image)
    tumor_properties = calculate_tumor_properties(ct_image, labels)
    
    # Example: Assuming you have feature vectors X and corresponding labels y for ML
    X = np.random.rand(100, 10)  # Example feature matrix
    y = np.random.randint(0, 2, size=100)  # Example labels
    
    # Uncomment this if you want to train and evaluate classifier on the fly
    accuracy, classification_rep = train_and_evaluate_ml_classifiers(X, y)
    
    return tumor_volume_cc, tumor_stage, metadata, tumor_properties, accuracy, classification_rep

# Function to select and process file
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("MHD files", "*.mhd"), ("ZRAW files", "*.zraw")])
    if file_path:
        try:
            tumor_volume_cc, tumor_stage, metadata, tumor_properties, accuracy, classification_rep = process_ct_image(file_path)
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
                           f"Window Width: {metadata['Window Width']}\n"
                           f"Classification Report:\n{classification_rep}\n"
                           f"Accuracy: {accuracy * 100:.2f}%\n")
            result_label.config(text=result_text)

            # Display tumor image
            plt.figure(figsize=(8, 6))
            plt.imshow(labels[labels.shape[0] // 2], cmap='gray')
            plt.title('Tumor Image')
            plt.axis('off')
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))

# Building the UI using Tkinter
root = tk.Tk()
root.title("CT Image Processor")

frame = tk.Frame(root)
frame.pack(pady=20, padx=20)

select_button = tk.Button(frame, text="Select CT image File For RF", command=select_file)
select_button.pack()

result_label = tk.Label(frame, text="", justify=tk.LEFT, anchor="w", wraplength=600)
result_label.pack(pady=20)

root.mainloop()
