# DeepSign-CNN-Based-Sign-Language-Recognition-System

DeepSign is a deep learning–powered computer vision system designed to recognize American Sign Language (ASL) alphabet gestures from images.
The project demonstrates how a Convolutional Neural Network (CNN) can be trained, evaluated, and prepared for real-world inference using a clean, production-style workflow.

This system focuses on robust multi-class image classification, model evaluation, and deployable design — making it suitable for portfolios, LinkedIn showcases, and entry-level ML / AI engineering roles.

⭐ Project Highlights

-Built a custom CNN model using TensorFlow to classify ASL hand gestures
-Solved a 24-class image classification problem (excluding motion-based letters J and Z)
-Worked with a realistic MNIST-style vision dataset in CSV format
=Applied proper data preprocessing and reshaping for CNN input
-Evaluated model performance using accuracy, loss, and confusion matrix
-Saved the trained model for inference and future deployment
-Designed the project with API-readiness in mind for production use

🧠 Problem Overview

American Sign Language is a complex and widely used visual language.
Automated recognition of ASL gestures can help enable:

-Assistive communication tools
-Human–computer interaction systems
-Educational and accessibility applications
-This project focuses on static ASL alphabet recognition, where each image represents a single hand gesture corresponding to a letter (A–Z, excluding J and Z).

📊 Dataset Information

-Dataset: Sign Language MNIST
-Source: Coursera Lab (MNIST-style ASL dataset)
-Classes: 24 alphabet letters (A–Z excluding J and Z)
-Image Size: 28 × 28 grayscale
-Format: CSV (label + 784 pixel values)
-Training Samples: ~27,000
-Test Samples: ~7,000

The dataset follows the same structure as classic MNIST, making it ideal for CNN-based learning while remaining more challenging and realistic.

🏗️ Model Architecture (Overview)

The CNN architecture follows a standard, industry-accepted design:

-Convolutional layers for feature extraction
-Max-Pooling layers for spatial reduction
-Fully connected (Dense) layers for classification
-Softmax output for multi-class prediction
-This structure balances performance, simplicity, and interpretability, making it suitable for real-world inference systems.

📈 Model Evaluation

-The model was evaluated using:
-Training & validation accuracy
-Training & validation loss
-Confusion matrix to analyze class-wise performance
-This ensures the system is not only accurate but also reliable across different gesture classes.

🛠️ Tech Stack

-Programming Language: Python
-Deep Learning Framework: TensorFlow / Keras
-Data Processing: NumPy, Pandas
-Visualization: Matplotlib, Seaborn

Model Type: Convolutional Neural Network (CNN)

🚀 How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/DeepSign-CNN-Based-Sign-Language-Recognition-System.git
cd DeepSign-CNN-Based-Sign-Language-Recognition-System

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model

Run the training notebook or script to train the CNN and save the model.

4️⃣ Inference

The saved model can be loaded to perform predictions on new ASL gesture images.

🔮 Future Improvements

-Add FastAPI inference service for real-time predictions
-Support image uploads instead of CSV input
-Improve performance with data augmentation
-Extend to dynamic gestures (motion-based recognition)
-Deploy as a web or mobile application

📌 Why This Project Matters

-This project goes beyond model training and focuses on:
-Practical CNN implementation
-Clean data-to-model workflow
-Evaluation-driven development
-Production-oriented thinking
-It reflects how computer vision systems are built in real engineering environments, not just in notebooks.

📂 Repository

👉 GitHub: [[Add your repository link here]](https://github.com/RansiluRanasinghe/DeepSign-CNN-Based-Sign-Language-Recognition-System.git)

🤝 Let’s Connect

-Always open to discussions around:
-Deep learning fundamentals
-Computer vision projects

👉Linked in -> Ransilu Ranasinghe - www.linkedin.com/in/ransilu-ranasinghe-a596792ba

ML system design

Production-ready AI pipelines
