# DeepSign: CNN-Based Sign Language Recognition System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

DeepSign is a deep learning–powered computer vision system designed to recognize **American Sign Language (ASL)** alphabet gestures from images. The project demonstrates how a Convolutional Neural Network (CNN) can be trained, evaluated, and prepared for real-world inference using a clean, production-style workflow.

This system focuses on robust multi-class image classification, careful model evaluation, and deployable design — making it suitable for portfolios, LinkedIn showcases, and entry-level ML / AI engineering roles.

---

## ⭐ Project Highlights

- ✅ Built a custom CNN model using TensorFlow to classify ASL hand gesture images
- ✅ Solved a **24-class image classification problem** (excluding motion-based letters J and Z)
- ✅ Worked with a realistic image-based dataset organized in class-wise directories
- ✅ Applied proper image preprocessing and normalization for CNN input
- ✅ Evaluated model performance using training/validation accuracy, loss, and confusion matrix
- ✅ Identified the optimal training point based on validation loss trends
- ✅ Saved the trained model for inference and future deployment
- ✅ Designed the project with API-readiness in mind for production use

---

## 🧠 Problem Overview

American Sign Language is a complex and widely used visual language. Automated recognition of ASL gestures can help enable:

- **Assistive communication tools**
- **Human–computer interaction systems**
- **Educational and accessibility applications**

This project focuses on **static ASL alphabet recognition**, where each image represents a single hand gesture corresponding to a letter (A–Z, excluding J and Z).

---

## 📊 Dataset Information

| Property | Details |
|----------|---------|
| **Dataset** | Sign Language MNIST (image-based variant) |
| **Source** | Coursera Lab (derived from the Sign Language MNIST dataset) |
| **Classes** | 24 alphabet letters (A–Z excluding J and Z) |
| **Image Size** | 28 × 28 grayscale |
| **Format** | Image folders (`train/` and `validation/`, class-wise subdirectories) |
| **Training Samples** | ~27,000 |
| **Validation Samples** | ~7,000 |

The dataset follows an MNIST-style structure but is stored as images, closely resembling real-world computer vision pipelines used in industry.

---

## 🏗️ Model Architecture

The CNN architecture follows a standard, industry-accepted design:

```
Input (28x28x1)
    ↓
Convolutional Layers (Feature Extraction)
    ↓
Max-Pooling Layers (Spatial Reduction)
    ↓
Fully Connected (Dense) Layers
    ↓
Softmax Output (24 Classes)
```

This architecture balances performance, simplicity, and interpretability, making it suitable for real-world inference systems.

**Key Components:**
- Convolutional layers for feature extraction
- Max-pooling layers for spatial reduction
- Fully connected (Dense) layers for classification
- Softmax output for multi-class prediction

---

## 📈 Model Evaluation & Training Strategy

**Evaluation Metrics:**
- Training & validation accuracy
- Training & validation loss
- Confusion matrix for class-wise performance analysis

**Training Insights:**
- Training behavior analysis showed **optimal generalization around epoch 24**, based on minimum validation loss
- Subsequent epochs indicated mild overfitting, reinforcing an evaluation-driven stopping decision
- This ensures the model is not only accurate but also reliable and generalizable

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Programming Language** | Python |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Data Processing** | NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Type** | Convolutional Neural Network (CNN) |

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/RansiluRanasinghe/DeepSign-CNN-Based-Sign-Language-Recognition-System.git
cd DeepSign-CNN-Based-Sign-Language-Recognition-System
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
Run the training notebook or script to train the CNN and save the model:
```bash
# If using Jupyter Notebook
jupyter notebook

# Open and run the training notebook
```

### 4️⃣ Inference
The saved model can be loaded to perform predictions on new ASL gesture images.

---

## 📂 Project Structure

```
DeepSign-CNN-Based-Sign-Language-Recognition-System/
│
├── NoteBook/                   # Jupyter notebooks for training
├── data/                       # Dataset (train and validation)
│   ├── train/                  # Training images
│   └── validation/             # Validation images
├── models/                     # Saved trained models
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🔮 Future Improvements

- [ ] Add **FastAPI inference service** for real-time predictions
- [ ] Support **image uploads via REST API**
- [ ] Improve robustness using **data augmentation**
- [ ] Extend to **dynamic gestures** (motion-based recognition)
- [ ] Deploy as a **web or mobile application**

---

## 📌 Why This Project Matters

This project goes beyond basic model training and emphasizes:

✓ **Practical CNN implementation**  
✓ **Clean image-to-model workflow**  
✓ **Evaluation-driven decision making**  
✓ **Production-oriented thinking**

It reflects how computer vision systems are built in real engineering environments, not just academic notebooks.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Connect

**Ransilu Ranasinghe**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ransilu-ranasinghe-a596792ba)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/RansiluRanasinghe)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:dinisthar@gmail.com)

Always open to discussions around:
- Deep learning fundamentals
- Computer vision systems
- ML system design
- Production-ready AI pipelines

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

</div>
