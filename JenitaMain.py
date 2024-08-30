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
from glob import glob
from matplotlib import pyplot as plt

# Step 1: Load and Process All CT Images
def load_ct_image(file_path):
    if file_path.lower().endswith('.mhd'):
        ct_image = sitk.ReadImage(file_path)
    else:
        raise ValueError("Unsupported file format")

    ct_array = sitk.GetArrayFromImage(ct_image)
    return ct_image, ct_array

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
    metadata['Slice Thickness'] = image.GetMetaData('0018|0050') if image.HasMetaDataKey('0018|0050') else 'N/A'
    metadata['Patient Position'] = image.GetMetaData('0018|5100') if image.HasMetaDataKey('0018|5100') else 'N/A'
    metadata['Window Center'] = image.GetMetaData('0028|1050') if image.HasMetaDataKey('0028|1050') else 'N/A'
    metadata['Window Width'] = image.GetMetaData('0028|1051') if image.HasMetaDataKey('0028|1051') else 'N/A'
    return metadata

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

# Function to extract features from tumor properties
def extract_features(tumor_properties):
    return [
        tumor_properties['Size'],
        tumor_properties['Perimeter'],
        tumor_properties['Area'],
        # Add more features as needed
    ]

# Function to extract label from file path
def extract_label(file_path):
    # Implement your logic to extract labels
    # For example, based on file naming or associated metadata
    return 1 if 'cancer' in file_path.lower() else 0

# Load all MHD images and extract features/labels
def load_all_mhd_images(folder_path):
    file_paths = glob(os.path.join(folder_path, "*.mhd"))
    all_features = []
    all_labels = []
    for file_path in file_paths:
        try:
            ct_image, ct_array = load_ct_image(file_path)
            hu_array = get_hu_values(ct_image)
            filtered_image, filtered_array = apply_noise_reduction(ct_image)
            labels = segment_image(filtered_array, 2)
            tumor_properties = calculate_tumor_properties(ct_image, labels)
            
            features = extract_features(tumor_properties)
            all_features.append(features)
            
            label = extract_label(file_path)
            all_labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return np.array(all_features), np.array(all_labels)

# Train model on all data
def train_model_on_all_data(folder_path):
    X, y = load_all_mhd_images(folder_path)
    accuracy, classification_rep = train_and_evaluate_ml_classifiers(X, y)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_rep)
    return accuracy, classification_rep, X, y

def train_and_evaluate_ml_classifiers(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_rf)
    classification_rep = classification_report(y_test, y_pred_rf)
    return accuracy, classification_rep

# Prediction function
def predict_new_image(model, file_path):
    ct_image, ct_array = load_ct_image(file_path)
    hu_array = get_hu_values(ct_image)
    filtered_image, filtered_array = apply_noise_reduction(ct_image)
    labels = segment_image(filtered_array, 2)
    tumor_properties = calculate_tumor_properties(ct_image, labels)
    
    features = extract_features(tumor_properties)
    prediction = model.predict([features])
    return prediction[0]

# Function to select folder and train model
def select_folder_and_train_model():
    folder_path = filedialog.askdirectory()
    if folder_path:
        try:
            global trained_model, X, y
            accuracy, classification_rep, X, y = train_model_on_all_data(folder_path)
            messagebox.showinfo("Training Completed", f"Model trained with accuracy: {accuracy * 100:.2f}%")
            select_button.config(text="Select CT image File For Prediction", command=select_file_for_prediction)
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Function to select file for prediction
def select_file_for_prediction():
    file_path = filedialog.askopenfilename(filetypes=[("MHD files", "*.mhd")])
    if file_path:
        try:
            prediction = predict_new_image(trained_model, file_path)
            result_label.config(text=f"Prediction: {'Cancerous' if prediction == 1 else 'Non-cancerous'}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Building the UI using Tkinter
root = tk.Tk()
root.title("CT Image Processor")

frame = tk.Frame(root)
frame.pack(pady=20, padx=20)

select_button = tk.Button(frame, text="Select Folder with CT Images For Training", command=select_folder_and_train_model)
select_button.pack()

result_label = tk.Label(frame, text="", justify=tk.LEFT, anchor="w", wraplength=600)
result_label.pack(pady=20)

root.mainloop()
