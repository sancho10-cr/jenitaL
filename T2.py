import os
import SimpleITK as sitk
import numpy as np
from skimage.segmentation import slic
from skimage import measure
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load CT Images
def load_ct_image(file_path):
    ct_image = sitk.ReadImage(file_path)
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
def classify_tnm(volume):
    if volume <= 3:
        return 'T1'
    elif 3 < volume <= 7:
        return 'T2'
    elif 7 < volume:
        return 'T3'
    else:
        return 'T4'

# Step 5: Machine Learning Classification (Optional)
def train_and_evaluate_ml_classifiers(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train decision tree classifier
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    y_pred_dt = dt_classifier.predict(X_test)
    print(f"Predicted stages with Decision Tree: {y_pred_dt}")

    # Train random forest classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    print(f"Predicted stages with Random Forest: {y_pred_rf}")

# Main function
def main():
    # Directory containing the MHD files
    directory = os.path.dirname(__file__)
    downsample_factor = 2  # Downsample factor to reduce memory usage
    
    # Process each MHD file in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.mhd'):
            print(f"Processing file: {file_name}")
            file_path = os.path.join(directory, file_name)
            
            # Load CT image
            ct_image, ct_array = load_ct_image(file_path)

            # Convert to Hounsfield Units
            hu_array = get_hu_values(ct_image)

            # Apply noise reduction
            filtered_image, filtered_array = apply_noise_reduction(ct_image)

            # Segment the image
            labels = segment_image(filtered_array, downsample_factor)

            # Calculate tumor volume
            tumor_volume_cc = calculate_tumor_volume(ct_image, labels)
            print(f"Tumor Volume: {tumor_volume_cc:.2f} cc")

            # TNM classification
            tumor_stage = classify_tnm(tumor_volume_cc)
            print(f"Tumor Stage: {tumor_stage}")

    # Machine Learning classification (optional)
    # Example dataset
    X = np.array([[2.37, 5.72, 1.74], [6.91, 9.97, 2.97], [11.63, 13.9, 3.85], [13.85, 14.15, 4.2], [21.95, 25.28, 5.29], [56.44, 31.89, 8.48]])
    y = np.array(['T1', 'T1', 'T2', 'T2', 'T3', 'T4'])

    train_and_evaluate_ml_classifiers(X, y)

if __name__ == "__main__":
    main()
