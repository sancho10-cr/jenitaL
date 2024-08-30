# lung_tumor_classification.py

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Data
print("Loading data...")
data = pd.read_csv('candidates.csv')
print("Data loaded successfully!")

# Step 2: Data Cleaning
print("Checking for missing values...")
if data.isnull().sum().any():
    raise ValueError("Missing values found in the data. Please clean the data.")
print("No missing values found!")

# Convert columns to appropriate data types if needed
print("Converting data types...")
data['coordX'] = data['coordX'].astype(float)
data['coordY'] = data['coordY'].astype(float)
data['coordZ'] = data['coordZ'].astype(float)
data['class'] = data['class'].astype(int)
print("Data types converted successfully!")

# Step 3: Data Exploration
# Check the distribution of the classes
print("Checking class distribution...")
class_distribution = data['class'].value_counts()
print("Class Distribution:\n", class_distribution)

# Optional: Visualize the data points
print("Plotting data...")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['coordX'], data['coordY'], data['coordZ'], c=data['class'], cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
plt.title("3D Scatter Plot of Tumor Coordinates")
plt.xlabel("coordX")
plt.ylabel("coordY")
ax.set_zlabel("coordZ")
plt.show()
print("Data plotted successfully!")

# Step 4: Train-Test Split
print("Splitting data into training and test sets...")
X = data[['coordX', 'coordY', 'coordZ']]
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split successfully!")

# Step 5: Model Selection and Training
print("Training the model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)
print("Model trained successfully!")

# Predict on the test set
print("Making predictions...")
y_pred = clf.predict(X_test)

# Evaluate the model
print("Evaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:\n", classification_report(y_test, y_pred))

print("Script execution completed!")
# End of script
