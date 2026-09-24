# 🍅 Tomato Leaf Disease Detection

Hybrid Image Processing, Deep Feature Extraction & Machine Learning
An intelligent plant disease detection system that combines image processing, GLCM texture analysis, MobileNetV2 deep features, and machine learning to identify tomato leaf diseases from images.

# 📌 Project Overview
Agriculture plays a vital role in sustaining the global population and supporting economic growth. However, plant diseases can significantly affect crop productivity and quality. 
Detecting diseases at an early stage is therefore important for maintaining crop health and reducing potential crop losses.

Traditional plant disease identification often relies on manual inspection, which can be time-consuming, labor-intensive, and susceptible to human error. 
This project addresses this challenge by developing an automated tomato leaf disease detection system using computer vision, hybrid feature extraction, and machine learning.

The system processes a tomato leaf image through multiple stages. First, image preprocessing techniques such as grayscale conversion and Contrast Limited Adaptive Histogram Equalization (CLAHE) are applied to enhance the image. 
Image characteristics are then analyzed using metrics including PSNR, mean, standard deviation, and entropy.

To represent the visual characteristics of the leaf, the system combines two complementary feature-extraction approaches:

  - GLCM (Gray Level Co-occurrence Matrix) for texture features

  - MobileNetV2 for deep visual features

These features are combined into a hybrid feature vector, which is subsequently used to train and evaluate machine learning classifiers including Random Forest, XGBoost, and LightGBM.

The classifiers are evaluated using accuracy, precision, recall, and F1-score. 
The trained system can then analyze a new tomato leaf image and provide a predicted disease class together with its prediction confidence.

# 🎯 Project Objectives
The main objectives of this project are to:

- Develop an automated tomato leaf disease detection system.
- Improve leaf-image quality through preprocessing and contrast enhancement.
- Extract meaningful texture information using GLCM.
- Extract high-level visual representations using MobileNetV2.
- Combine traditional and deep features into a hybrid representation.
- Compare multiple machine learning classification algorithms.
- Evaluate classification performance using standard evaluation metrics.
- Provide disease predictions with confidence information.
- Support early identification of plant diseases for improved crop monitoring.

# 🔬 Proposed Methodology
The complete system follows the pipeline below:
<img width="1192" height="658" alt="image" src="https://github.com/user-attachments/assets/561b6975-721f-49da-af3b-3ef9700d8f8a" />

# 🖼️ Image Preprocessing
The input tomato leaf image is first processed to improve its quality and make relevant visual patterns easier to analyze.

### Grayscale Conversion
The color image is converted into grayscale to represent the image using intensity information. This simplifies subsequent texture analysis and reduces unnecessary color-channel complexity.

### CLAHE
Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to improve local contrast.

Unlike conventional global histogram equalization, CLAHE operates on local regions of an image and limits contrast amplification to reduce excessive noise enhancement.

This helps reveal subtle patterns and variations within the leaf image.

# 📊 Image Quality Analysis
The enhanced images are analyzed using several statistical and image-quality measurements:

 - PSNR (Peak Signal-to-Noise Ratio)

 - Mean

 - Standard Deviation

 - Entropy

These measurements provide information about image intensity, variation, information content, and the effect of preprocessing.

# 🧩 Hybrid Feature Extraction
A key component of the proposed system is the combination of traditional texture features with deep-learning-based features.

### GLCM Texture Features
The Gray Level Co-occurrence Matrix (GLCM) is used to capture spatial relationships between pixel intensities.

GLCM-based texture information can help represent visual characteristics associated with different disease patterns appearing on leaf surfaces.

### MobileNetV2 Deep Features
MobileNetV2 is used as a deep feature extractor to obtain higher-level visual representations from the leaf images.

The deep features complement the texture information extracted using GLCM.

### Feature Fusion
The two feature sets are combined:

<div align="center">

<pre>
GLCM Features
      +
MobileNetV2 Features
      │
      ▼
Hybrid Feature Vector
</pre>

</div>

The resulting hybrid representation is then provided to the machine learning classifiers.

# 🤖 Machine Learning Models
The project evaluates multiple classification algorithms:

### Random Forest
A tree-based ensemble learning algorithm that combines predictions from multiple decision trees.

### XGBoost
A gradient-boosting algorithm designed to build a strong predictive model through sequentially optimized decision trees.

### LightGBM
A gradient-boosting framework designed for efficient and scalable tree-based learning.

The models are evaluated using the same feature representation so that their classification performance can be compared.

# 📈 Model Evaluation
The classification models are evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified samples |
| Precision | Measures the correctness of positive predictions |
| Recall | Measures how effectively disease samples are identified |
| F1-Score | Combines precision and recall into a single metric |

The evaluation results are used to determine which trained model configuration is used for disease prediction.

# 🔍 Prediction
Once the model has been trained, a new tomato leaf image can be provided to the system.

The application performs the following operations:
<div align="center">

<pre>
Upload Leaf Image
       ↓
Preprocessing
       ↓
Feature Extraction
       ↓
Selected Model
(GLCM / MobileNetV2 / Hybrid)
       ↓
Trained ML Model
       ↓
Disease Classification
       ↓
Prediction + Confidence
</pre>

</div>

The final result provides the predicted disease category together with the model's confidence information.

# 🌱 Expected Benefits
The proposed system provides an automated approach to plant disease identification and can serve as a foundation for computer-vision-based agricultural monitoring.

 - Potential benefits include:

 - Faster preliminary disease identification

 - image-based analysis

 - Reduced dependence on manual visual inspection

 - Consistent feature-based classification

 - Support for early crop-health monitoring

 - Potential integration into smart agriculture applications

# 🚀 Project Status
The project includes a web-based application, trained machine-learning models, image-processing modules, feature-extraction methods, and supporting testing utilities.

The repository also includes the trained model required for prediction, while the large image dataset is excluded from the Git repository because of its size.

# 🚀 Installation & Execution

This section explains how to set up and run the Tomato Leaf Disease Detection System locally.

# 📋 Prerequisites

Before running the project, make sure the following are installed:

 - [Anaconda](https://www.anaconda.com/)
 - [Python 3.8](https://www.python.org/downloads/release/python-380/)
 - [Git](https://git-scm.com/)

A modern web browser

The application is developed and tested using a Conda environment with Python 3.8.

# 📦 1. Download the Dataset

The project requires the Tomato Leaf Disease Dataset for model training and/or dataset-based processing.

The dataset is available on Kaggle:

**Dataset:** [Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/naveedgull/tomato-leaf-disease)

Download and extract the dataset before proceeding with the project setup.

After extraction, place the dataset in the project directory according to the expected dataset structure.

Expected Dataset Structure

```text
tomato/
└── tomato/
    ├── train/
    │   ├── Tomato___Bacterial_spot/
    │   ├── Tomato___Early_blight/
    │   ├── Tomato___healthy/
    │   ├── Tomato___Late_blight/
    │   └── ...
    │
    └── ...
```


> [!NOTE]
> The complete dataset is not included in this GitHub repository because of its large size. Users should download it separately from [Kaggle](https://www.kaggle.com/datasets/naveedgull/tomato-leaf-disease).

# 🐍 2. Open Anaconda Prompt

Open Anaconda Prompt from the Start Menu.

You should see a prompt similar to:
```text
(base) C:\Users\System>
```

# 🔧 3. Create the Conda Environment

Create a new Conda environment using Python 3.8:
```text
conda create -n tf python=3.8
```

When prompted, enter:

```text
y
```

to confirm the installation.

# ✅ 4. Activate the Environment

Activate the newly created environment:

```text
conda activate tf
```

The prompt should now look similar to:

```text
(tf) C:\Users\System>
```

# 📁 5. Clone the Repository

Clone the project from GitHub:

```text
git clone https://github.com/Harshini382/Tomato-Leaf-Disease-Detection.git
```

Move into the project directory:

```text
cd Tomato-Leaf-Disease-Detection
```

Then enter the application directory:

```text
cd tomato
```

Your terminal should now be inside the project directory.

# 🗂️ 6. Verify the Project Structure

The application should contain the following major files and directories:

```text
tomato/
│
├── app.py
├── database.py
├── models.py
├── forms.py
├── image_metrics.py
├── feature_visualization.py
├── retrain_model.py
├── requirements.txt
├── tomato_disease_model.pkl
│
├── methods/
│   ├── image_pipeline.py
│   ├── method1.py
│   ├── method2.py
│   └── method3.py
│
├── static/
│   └── styles.css
│
├── templates/
│   ├── landing.html
│   ├── login.html
│   ├── register.html
│   ├── index.html
│   ├── result.html
│   ├── analysis_report.html
│   └── about.html
│
└── tests/
```

The dataset should be downloaded separately and placed in the appropriate tomato/ dataset directory when training or retraining the model.

# 📚 7. Install Required Python Packages

The project includes a requirements.txt file containing the required dependencies.

Install the dependencies using:

```text
pip install -r requirements.txt
```

If you prefer to install the primary packages individually, the project uses packages including:

 - pip install flask
 - pip install tensorflow
 - pip install keras
 - pip install numpy
 - pip install pandas
 - pip install pillow
 - pip install opencv-python
 - pip install matplotlib
 - pip install scikit-learn
 - pip install flask-wtf
 - pip install flask-sqlalchemy


*Recommendation: Using pip install -r requirements.txt is preferred because it keeps the installation process consistent with the project's dependency list.*

# ▶️ 8. Run the Flask Application

Make sure the Conda environment is active and that you are inside the project directory:

```text
(tf) ...\Tomato-Leaf-Disease-Detection\tomato>
```

Start the Flask application:

```text
python app.py
```

If the application starts successfully, Flask will provide a local address similar to:

**http://127.0.0.1:5000**

# 🌐 9. Open the Application

Open a web browser and visit:

**http://127.0.0.1:5000**


The Tomato Leaf Disease Detection System landing page should appear.

# 👤 10. Register and Login

From the web application:

 - Create a new user account.

 - Enter your login credentials.

 - Sign in to the application.

 - The disease detection dashboard will become available.

# 📷 11. Upload a Tomato Leaf Image

From the dashboard:

 - Select Upload Image.

 - Choose a tomato leaf image from your computer.

 - Upload the image.

**Supported Image Formats**

 - jpg

 - jpeg

 - png

# 🔍 12. Start Disease Prediction

After uploading the image, select:

Predict Disease

The system processes the image through the disease-detection pipeline, including:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Image Enhancement
     ↓
Feature Extraction
     ↓
Deep / Texture Feature Processing
     ↓
Machine Learning Classification
     ↓
Disease Prediction
```

Depending on the selected processing method, the system performs operations such as image preprocessing, feature extraction, and classification.

# 📊 13. View the Prediction Result

After processing is complete, the application displays the prediction results.

The result interface may include:

 - Predicted disease name

 - Prediction confidence / accuracy information

 - Uploaded image preview

 - Feature visualization

 - Image analysis

 - Analysis report

This allows users to examine both the classification result and supporting image-analysis information.

# 🧪 14. Model Training / Retraining

The project also contains scripts for model development and retraining.

The dataset must be available locally before attempting to train or retrain the models.

For example:

```text
python retrain_model.py
```

Additional training scripts included in the repository can be used for experimentation with the project's feature-extraction and classification approaches.

> [!NOTE]
> Training can require significantly more computational resources and time than simply running the already-trained application.


# 🛑 15. Stop the Application

To stop the Flask development server, return to the terminal where the application is running and press:

```text
CTRL + C
```

The Flask server will stop and the terminal prompt will become available again.

# ⚡ Quick Start

For users who have already installed Anaconda and downloaded the dataset:

```cmd
git clone https://github.com/Harshini382/Tomato-Leaf-Disease-Detection.git
cd Tomato-Leaf-Disease-Detection
conda create -n tf python=3.8
conda activate tf
cd tomato
pip install -r requirements.txt
python app.py
```


Then open:

```cmd
http://127.0.0.1:5000
```

# 📝 Important Notes

The dataset is not included in this GitHub repository because of its size.

Download the dataset separately from Kaggle before performing model training or retraining.

The repository contains a pre-trained model for application use.

Make sure the required Python environment and dependencies are installed before launching the application.

For model training, ensure that the dataset directory structure matches the expected structure used by the training scripts.

---

© 2026 Harshini Gengaraj. All Rights Reserved.


