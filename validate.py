import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained model
model = FusionLiteCNN()
model.load_state_dict(torch.load('fusion_model.pth'))
model.eval()

# Generate synthetic validation dataset (replace with real data)
def generate_validation_data(num_samples=1000):
    np.random.seed(42)
    
    # Synthetic NDVI images (64x64)
    X_img = np.random.uniform(low=-1, high=1, size=(num_samples, 1, 64, 64))
    
    # Synthetic sensor data
    X_sensor = np.column_stack([
        np.random.uniform(0, 100, num_samples),  # moisture
        np.random.uniform(10, 50, num_samples),  # temp
        np.random.uniform(0, 100, num_samples),  # humidity
        np.random.uniform(3, 9, num_samples)     # pH
    ])
    
    # Generate labels based on heuristic rules (replace with real labels)
    y = np.zeros(num_samples)
    for i in range(num_samples):
        ndvi_mean = X_img[i].mean()
        moisture = X_sensor[i,0]
        temp = X_sensor[i,1]
        
        if ndvi_mean > 0.6 and moisture > 60 and temp < 35:
            y[i] = 0  # healthy
        elif ndvi_mean < 0.3 or moisture < 30 or temp > 40:
            y[i] = 2  # drought
        else:
            y[i] = 1  # unhealthy
            
    return X_img, X_sensor, y

# Generate validation data
X_img, X_sensor, y_true = generate_validation_data(1000)

# Convert to tensors
X_img_tensor = torch.FloatTensor(X_img)
X_sensor_tensor = torch.FloatTensor(X_sensor)

# Get model predictions
with torch.no_grad():
    outputs = model(X_img_tensor, X_sensor_tensor)
    y_pred = torch.argmax(outputs, dim=1).numpy()

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_true, y_pred, 
                           target_names=["Healthy", "Unhealthy", "Drought"]))

# Confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Healthy", "Unhealthy", "Drought"],
                yticklabels=["Healthy", "Unhealthy", "Drought"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_true, y_pred)

# Case analysis of misclassified samples
def analyze_errors(X_img, X_sensor, y_true, y_pred):
    errors = np.where(y_true != y_pred)[0]
    print(f"\nAnalyzing {len(errors)} misclassified samples:")
    
    for i in errors[:5]:  # Print first 5 errors
        print(f"\nSample {i}:")
        print(f"Actual: {['Healthy','Unhealthy','Drought'][int(y_true[i])]}")
        print(f"Predicted: {['Healthy','Unhealthy','Drought'][int(y_pred[i])]}")
        print(f"NDVI mean: {X_img[i].mean():.2f}")
        print(f"Sensors - Moisture: {X_sensor[i,0]:.1f}%, Temp: {X_sensor[i,1]:.1f}°C, " +
              f"Humidity: {X_sensor[i,2]:.1f}%, pH: {X_sensor[i,3]:.1f}")

analyze_errors(X_img, X_sensor, y_true, y_pred)