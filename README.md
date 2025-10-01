##Advanced Multi-Modal Crop Health Classification System
This project implements a sophisticated multi-modal deep learning system in PyTorch for classifying crop health. It leverages a combination of synthetic multi-spectral imagery, temporal sensor data, and environmental context to achieve robust and accurate predictions. The model incorporates advanced architectural components like residual blocks, attention mechanisms, and an LSTM for temporal analysis.

##Key Features
Multi-Modal Data Fusion: Integrates three distinct data sources for a holistic analysis:

Spectral Data: 5-channel imagery (RGB, NIR, SWIR).

Temporal Sensor Data: 7-day history of sensor readings (temperature, humidity, soil moisture, etc.).

Environmental Context: Static data like season, soil type, and elevation.

Advanced Model Architecture:

A ResNet-style CNN backbone with channel and spatial attention for processing spectral images.

An LSTM network to capture temporal patterns in sensor data.

Cross-modal attention to intelligently weigh the importance of each data modality for the final prediction.

Uncertainty Estimation: The model includes a dedicated head to predict its own uncertainty, helping to identify low-confidence predictions that may require human review.

Robust Training Pipeline:

Focal Loss to handle class imbalance.

WeightedRandomSampler to ensure balanced batches during training.

AdamW optimizer and a Cosine Annealing learning rate scheduler for stable convergence.

Advanced data augmentation techniques for both spectral and sensor data.

Comprehensive Evaluation & Visualization: Generates a suite of detailed plots and reports for in-depth model analysis, including:

Classification reports and confusion matrices.

ROC and Precision-Recall curves.

Uncertainty calibration and distribution plots.

Analysis of attention weights to understand model focus.

Gradient-based feature importance visualization.
