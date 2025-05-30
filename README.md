# Smart Dairy Management Using Cow Muzzle Recognition

This repository contains the implementation of four distinct approaches for cattle muzzle-based biometric identification, developed as part of the capstone project **"Smart Dairy Management Using Cow Muzzle Recognition"** by **Gyanendra Mani** and **Nitesh Kumar** (IIT Ropar, 2025).

## 📁 Folder Structure

1. **`classical-cv-approach/`**  
   Traditional computer vision pipeline using **SIFT** for handcrafted feature extraction and **FLANN** for matching muzzle features.

2. **`transfer-learning-efficientnet/`**  
   Utilizes **EfficientNet-B3**, pretrained on ImageNet, for deep feature extraction. Similarity between images is calculated using **Mahalanobis distance**.

3. **`autoencoder-embedding/`**  
   Employs an **unsupervised convolutional autoencoder** to learn latent representations of muzzle patterns. Matching is done using **cosine similarity** in the embedding space.

4. **`autoencoder-gem-continual/`**  
   Integrates **Gradient Episodic Memory (GEM)** into the autoencoder framework for **continual learning**. Aims to add new identities without retraining from scratch.

## 🛠️ System Architecture

- **Frontend**: React Native (Android)
- **Backend**: Flask REST API
- **Database**: MongoDB with GridFS for image storage
- **Detection**: YOLOv11 (trained via Roboflow)
- **Recognition**: EfficientNet-B3 (deployed model)

## 📊 Key Results

| Method                          | Same-Day Accuracy | Different-Day (After 8 months) Accuracy |
|----------------------------------|-------------------|------------------------|
| Classical CV (SIFT + FLANN)      | 74.07%            | 21.67%                 |
| Transfer Learning (EfficientNet) | 100%              | ~50%                   |
| Autoencoder Embedding            | 70%               | 70%                    |
| Autoencoder + GEM                | Underperformed    | Not recommended        |

## 📘 Project Report

Refer to the detailed methodology, experiments, and analysis in the [BTP-Final-Report.pdf](./BTP-Final-Report.pdf) available in this repository.

## 🔧 Setup Instructions

Each approach has its own folder with a `README.md` file explaining how to install dependencies, run the code, and interpret results. Start by exploring the method folders listed above.

---

For any queries or collaboration requests, feel free to reach out to the me.
