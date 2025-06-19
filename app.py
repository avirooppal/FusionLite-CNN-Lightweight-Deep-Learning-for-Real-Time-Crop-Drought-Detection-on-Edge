import streamlit as st
import numpy as np
import ee
import torch
import json
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict

# Initialize GEE
def initialize_gee():
    try:
        ee.Initialize(project='ndvi-imagery')
        return True
    except Exception as e:
        st.error(f"GEE initialization failed: {str(e)}")
        if st.button("Authenticate GEE"):
            ee.Authenticate()
            st.rerun()
        return False

# --- Model Definition ---
class FusionLiteCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten()
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4, 16), 
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32)
        )
        self.classifier = torch.nn.Linear(64*16*16 + 32, 3)  # 3 classes

    def forward(self, img, sensors):
        img_feat = self.cnn(img)
        sensor_feat = self.mlp(sensors)
        return self.classifier(torch.cat([img_feat, sensor_feat], dim=1))

# --- Load Pre-trained Model ---
@st.cache_resource
def load_model():
    model = FusionLiteCNN()
    # Create state dict with proper tensor types
    state_dict = torch.load('fusion_model.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- GEE NDVI Fetch ---
def get_ndvi_image(coords):
    try:
        polygon = ee.Geometry.Polygon(coords)
        image = ee.ImageCollection('COPERNICUS/S2_SR') \
                 .filterBounds(polygon) \
                 .filterDate('2024-01-01', '2024-06-01') \
                 .sort('CLOUDY_PIXEL_PERCENTAGE') \
                 .first()
        
        if image.bandNames().size().getInfo() == 0:
            st.error("No image found for the given location and date range")
            return None, None

        ndvi = image.normalizedDifference(['B8', 'B4'])
        
        url = ndvi.getThumbURL({
            'min': -1, 'max': 1,
            'palette': ['red', 'yellow', 'green'],
            'region': polygon,
            'dimensions': 512
        })
        
        arr = ndvi.sampleRectangle(region=polygon).get('NDVI').getInfo()
        arr = np.array(arr, dtype=np.float32)
        
        arr = np.nan_to_num(arr, nan=0)
        arr = np.clip(arr, -1, 1)
        
        if arr.size > 0:
            arr = cv2.resize(arr, (64, 64), interpolation=cv2.INTER_LINEAR)
            return arr, url
        else:
            st.error("Empty NDVI array returned from GEE")
            return None, None
            
    except Exception as e:
        st.error(f"Error fetching NDVI: {str(e)}")
        return None, None

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    st.title("🌾 FusionLite-CNN: Crop Health Monitor")
    
    if not initialize_gee():
        st.stop()
    
    # --- Sidebar Controls ---
    st.sidebar.header("Input Parameters")
    
    coord_input = st.sidebar.text_input(
        "Farm Coordinates (JSON)", 
        '[[77.59,12.97],[77.60,12.97],[77.60,12.98],[77.59,12.98]]'
    )
    
    try:
        coords = json.loads(coord_input)
        if not isinstance(coords, list) or len(coords) < 3:
            st.sidebar.error("Please enter valid polygon coordinates (at least 3 points)")
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON format for coordinates")
    
    use_real_data = st.sidebar.checkbox("Use Google Earth Engine", True)
    
    st.sidebar.subheader("Sensor Data")
    moisture = st.sidebar.slider("Soil Moisture (%)", 0, 100, 30)
    temp = st.sidebar.slider("Temperature (°C)", 10, 50, 25)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 70)
    ph = st.sidebar.slider("Soil pH", 3.0, 9.0, 6.5, step=0.1)
    
    # --- Main Display ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("NDVI Input")
        if use_real_data:
            if st.button("Fetch Live NDVI"):
                try:
                    coords = json.loads(coord_input)
                    ndvi_arr, ndvi_url = get_ndvi_image(coords)
                    
                    if ndvi_arr is not None and ndvi_url is not None:
                        st.session_state.ndvi = ndvi_arr
                        st.image(ndvi_url, caption="Sentinel-2 NDVI", use_container_width=True)
                        
                        # --- Dynamic NDVI Color Scaling ---
                        ndvi_display = st.session_state.ndvi.copy()
                        vmin = np.percentile(ndvi_display, 5)
                        vmax = np.percentile(ndvi_display, 95)
                        if (vmax - vmin) < 0.2:
                            vmin, vmax = -1, 1
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(ndvi_display, cmap='RdYlGn', vmin=vmin, vmax=vmax)
                        plt.colorbar(im, ax=ax, label='NDVI Value', extend='both')
                        ax.set_title(f'NDVI (Dynamic Range: {vmin:.2f} to {vmax:.2f})')
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            label = st.selectbox("Simulate NDVI", ["healthy", "unhealthy", "drought"])
            ndvi = np.random.uniform(
                0.6 if label=="healthy" else 0.3 if label=="unhealthy" else 0.0,
                0.9 if label=="healthy" else 0.6 if label=="unhealthy" else 0.3,
                (64, 64)
            )
            st.session_state.ndvi = ndvi
            # --- Dynamic NDVI Color Scaling ---
            ndvi_display = st.session_state.ndvi.copy()
            vmin = np.percentile(ndvi_display, 5)
            vmax = np.percentile(ndvi_display, 95)
            if (vmax - vmin) < 0.2:
                vmin, vmax = 0, 1
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(ndvi_display, cmap='RdYlGn', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, label='NDVI Value', extend='both')
            ax.set_title(f'NDVI (Dynamic Range: {vmin:.2f} to {vmax:.2f})')
            st.pyplot(fig)
    
    with col2:
        st.subheader("Analysis Results")
        if 'ndvi' in st.session_state:
            try:
                model = load_model()
                ndvi = st.session_state.ndvi.astype(np.float32)
                ndvi_tensor = torch.FloatTensor(ndvi).unsqueeze(0).unsqueeze(0).requires_grad_(True)
                sensor_tensor = torch.FloatTensor([
                    [
                        moisture/100.0,
                        (temp-10)/40.0,
                        humidity/100.0,
                        (ph-3.0)/6.0
                    ]
                ]).requires_grad_(True)
                target_layer = model.cnn[3]
                cam_extractor = GradCAM(model, target_layer=target_layer)
                with torch.enable_grad():
                    output = model(ndvi_tensor, sensor_tensor)
                    pred = torch.argmax(output).item()
                    prob = torch.nn.functional.softmax(output, dim=1)[0].detach().cpu().numpy()
                # --- Enhanced Stress Heatmap Visualization ---
                activation_map = cam_extractor(pred, output)
                if activation_map:
                    heatmap = activation_map[0].squeeze().cpu().numpy()
                    heatmap = cv2.resize(heatmap, (ndvi.shape[1], ndvi.shape[0]))
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                    heatmap = np.clip(heatmap*1.5, 0, 1)  # Boost contrast
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im1 = ax.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                    im2 = ax.imshow(heatmap, cmap='jet', alpha=0.4)
                    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
                    cbar1.set_label('NDVI Value', rotation=270, labelpad=15)
                    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.15)
                    cbar2.set_label('Stress Level', rotation=270, labelpad=15)
                    ax.set_title('NDVI with Stress Heatmap Overlay')
                    st.pyplot(fig)
                labels = ["Healthy 🌱", "Unhealthy 🥀", "Drought-Affected 🔥"]
                st.metric("Prediction", labels[pred], f"Confidence: {prob[pred]*100:.1f}%")
                # --- Prediction confidence indicator ---
                st.progress(int(prob[pred]*100))
                st.caption(f"Prediction Confidence: {prob[pred]*100:.1f}%")
                # --- Key stress indicators table ---
                stress_data = {
                    "Indicator": ["NDVI Range", "Moisture Stress", "Thermal Stress"],
                    "Value": [
                        f"{ndvi.min():.2f} to {ndvi.max():.2f}",
                        "High" if moisture < 40 else "Medium" if moisture < 60 else "Low",
                        "High" if temp > 35 else "Medium" if temp > 28 else "Low"
                    ]
                }
                st.table(stress_data)
                # --- Legend for interpretation ---
                with st.expander("How to interpret the results"):
                    st.markdown("""
                    **Color Guide:**
                    - 🟢 Green: Healthy vegetation (NDVI > 0.6)
                    - 🟡 Yellow: Moderate stress (0.3 < NDVI ≤ 0.6)
                    - 🔴 Red: Severe stress (NDVI ≤ 0.3)
                    
                    **Heatmap Intensity:**
                    - 🔵 Blue: Low stress
                    - 🟢 Green: Moderate stress
                    - 🔴 Red: High stress
                    """)
                fig, ax = plt.subplots()
                ax.bar(range(3), prob, tick_label=labels)
                ax.set_ylabel("Probability")
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                if pred == 0:
                    st.success("✅ Advisory: Crops are healthy. Maintain current practices.")
                elif pred == 1:
                    st.warning("⚠️ Advisory: Potential crop stress detected!")
                else:
                    st.error("🚨 Advisory: Severe drought conditions detected!")
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()