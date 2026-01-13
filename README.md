# Brain MRI Tumor Classification

This repository contains a Computer Vision and Deep Learning project for classifying brain MRI images.
The project aims to automatically detect the presence of brain tumors from MRI scans using a Convolutional Neural Network (CNN).
The project demonstrates fundamental computer vision concepts, image processing techniques, and practical experimentation as part of academic coursework / self-learning.

This work is intended for educational and research purposes and demonstrates how deep learning can be applied to medical image analysis.

---

## 🧠 Dataset Information

- **Dataset Name:** Brain MRI Dataset
- **Domain:** Medical Imaging
- **Image Type:** Brain MRI scans
- **Task:** Image Classification
- **Classes:** 
  - Tumor
  - No Tumor  
  *(or multiple tumor types depending on dataset structure)*

The dataset contains labeled MRI images of the human brain and is widely used for learning and experimentation in medical computer vision.

---

## 🎯 Problem Statement

Manual diagnosis of brain tumors from MRI scans can be time-consuming and requires expert knowledge.
This project explores the use of deep learning models to assist in automatically identifying tumor presence in brain MRI images.

---

## 🏗️ Model Architecture

The model used in this project is a **Convolutional Neural Network (CNN)** designed for image classification tasks.

### Key Components:
- Convolutional layers for feature extraction
- ReLU activation functions
- Pooling layers for dimensionality reduction
- Fully connected (Dense) layers
- Output layer with Softmax/Sigmoid activation

The CNN learns spatial features such as edges, textures, and patterns that are indicative of brain tumors.

---

## 🛠️ Technologies & Tools

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib

---

## 📂 Project Structure

- **CV_Project_Mohammed_Said.ipynb**: Main notebook containing:
  - Data loading and preprocessing
  - Model building and training
  - Evaluation and visualization of results

---

## 🔄 Workflow

1. Load and preprocess MRI images
2. Normalize and resize images
3. Split data into training and validation sets
4. Build CNN model
5. Train the model on MRI images
6. Evaluate performance
7. Visualize results and predictions

---

## 🚀 How to Run the Project

1. Clone the repository:
```bash
  git clone https://github.com/your-username/brain-mri-tumor-classification.git
```

2. Navigate to the project directory:
```bash
  cd brain-mri-tumor-classification
```

3. Install required libraries:
```bash
  pip install numpy opencv-python matplotlib tensorflow
```

4. Launch Jupyter Notebook:
```bash
   jupyter notebook CV_Project_Mohammed_Said.ipynb
```

   ## 📊 Results & Evaluation

The notebook includes comprehensive evaluation of the trained model, including:

- Training and validation accuracy
- Training and validation loss curves
- Sample predictions on test MRI images

The results demonstrate that Convolutional Neural Networks (CNNs) are effective in learning discriminative features from brain MRI scans and can successfully distinguish between different classes.

---

## 📌 Key Learnings

Through this project, the following key concepts were learned:

- Medical image preprocessing techniques
- Applying CNNs to real-world medical datasets
- Handling challenges specific to medical imaging data
- Evaluating and visualizing deep learning model performance

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.  
It is **not designed for medical diagnosis or clinical decision-making**.

---

## 👤 Author

**Mohammed Said**  
Computer Science Undergraduate  
Aspiring Software Engineer | Competitive Programmer | Machine Learning Enthusiast

---

## 📄 License

This project is licensed under the **MIT License**.

You are free to:
- Use the code for educational and research purposes
- Modify and adapt the code
- Share and distribute the project

The only requirement is to give appropriate credit to the original author.

See the `LICENSE` file for more details.

