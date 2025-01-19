# CardioVision

### Detailed Explanation of the Four Codes for Image and Heart Analysis

---

### **Code 1: `nemodar2`**
This code processes medical images to extract features related to heart damage (e.g., myocardial infarction or MI). Its primary goal is to analyze and extract insights from medical images.

#### **Key Features:**
1. **Data Loading**:
   - Loads medical images from predefined paths for healthy (normal) and damaged (MI) cases.

2. **Data Augmentation**:
   - Applies transformations like rotation and horizontal flipping to increase the diversity of training data, enhancing the model’s generalization capability.

3. **Feature Extraction**:
   - Converts images to grayscale and applies Otsu thresholding to identify damaged regions (e.g., white areas).
   - Calculates the area of damaged regions and computes the percentage of damaged area relative to the total image area.

4. **Prediction Simulation**:
   - Simulates predictions by adding noise to real values, assuming lower error for healthy images and higher error for MI images.

5. **Visualization**:
   - **Correlation Plots**: Shows how closely the predicted values match the actual values.
   - **Bland-Altman Plots**: Examines the differences between actual and predicted values to evaluate prediction accuracy.

This code serves as the foundation for processing and preparing data for further analysis.

---

### **Code 2: `nemodar`**
This code utilizes a deep learning model, **MobileNetV2**, to classify medical images into healthy and MI categories.

#### **Key Features:**
1. **Data Preparation**:
   - Loads healthy and MI images, resizes them to 224x224 pixels, and preprocesses them for compatibility with MobileNetV2.
   - Splits the data into training (80%) and testing (20%) sets.

2. **Model Construction**:
   - Uses MobileNetV2 (pre-trained on ImageNet) as the base model to extract essential features.
   - Adds custom layers, including `GlobalAveragePooling2D` and a dense layer with a sigmoid activation, for binary classification.

3. **Evaluation**:
   - Computes and plots **ROC curves** for different heart regions (Overall, Apical, Midcavity, Basal).
   - Measures model performance using **AUC** (Area Under the Curve).

This code focuses on building and evaluating a robust classification model for detecting MI regions in heart images.

---

### **Code 3: `TABLE`**
This code provides statistical analysis and generates detailed tables to summarize key metrics from the processed data.

#### **Key Features:**
1. **Image Analysis**:
   - Processes grayscale images to compute average dimensions and intensity levels.
   - Extracts metrics like left ventricle area, damaged area, and percentage of damage.

2. **Statistical Metrics**:
   - Calculates sensitivity, specificity, and AUC for the model’s performance.
   - Computes p-values to evaluate the significance of differences between groups.

3. **Table Generation**:
   - Produces tables summarizing:
     - Patient demographics (e.g., age, weight, cardiac features).
     - Sensitivity, specificity, and AUC for different methods and heart regions.
     - Comparisons between healthy and MI groups.

4. **Excel Export**:
   - Saves all generated tables to an Excel file for easy access and further analysis.

This code complements `nemodar2` and `nemodar` by offering statistical insights into the dataset and model performance.

---

### **Code 4: `tasvir`**
This code uses an **Autoencoder** to analyze, reconstruct, and identify anomalies in MRI images of the heart.

#### **Key Features:**
1. **Data Preparation**:
   - Loads healthy and MI images, resizes them to 128x128 pixels, and normalizes pixel values to [0, 1].

2. **Autoencoder Construction**:
   - **Encoder**: Compresses input data to extract key features using convolutional and max-pooling layers.
   - **Decoder**: Reconstructs the input image from the compressed features using up-sampling and convolutional layers.

3. **Training**:
   - Trains the Autoencoder using healthy images to learn the normal pattern of heart regions.

4. **Reconstruction and Anomaly Detection**:
   - Feeds MI images into the trained Autoencoder for reconstruction.
   - Computes anomaly maps by calculating the absolute difference between the original and reconstructed images.
   - Generates binary masks highlighting damaged regions.

5. **Visualization**:
   - Displays results in a grid format, including:
     - Original healthy and MI images.
     - Anomaly masks.
     - Heatmaps (e.g., hot and jet colormaps) for damaged regions.

This code focuses on anomaly detection by reconstructing images and identifying damaged areas based on reconstruction differences.

---

### **Relationship Between Codes:**
1. **`nemodar2`**: Processes raw images, extracts features, and prepares data for analysis.
2. **`nemodar`**: Builds and evaluates a deep learning model to classify healthy and MI images.
3. **`TABLE`**: Provides detailed statistical insights and generates tables summarizing model performance and dataset characteristics.
4. **`tasvir`**: Reconstructs and detects anomalies in images, specifically focusing on identifying damaged regions.

### **Execution Order**:
1. Run **`nemodar2`** to prepare the data and extract features.
2. Use **`nemodar`** to train and evaluate the classification model.
3. Execute **`TABLE`** for statistical analysis and generating summary tables.
4. Finally, run **`tasvir`** for anomaly detection and detailed visualization of damaged areas.
