# 🌍 TerraChange

**TerraChange** is a **self-supervised deep learning web application** for **satellite image change detection**, designed to identify meaningful land-surface changes between multi-temporal remote sensing images.

The system leverages **SimCLR-based contrastive learning** with a **ResNet-50 backbone**, followed by a **Siamese U-Net decoder** for pixel-level binary change prediction. The application is deployed using **Streamlit Cloud** and supports interactive inference through a web interface.

---
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Self-Supervised Learning](https://img.shields.io/badge/Self--Supervised-SimCLR-green)
![Backbone](https://img.shields.io/badge/Backbone-ResNet50-lightgrey)

---
## 🚀 Live Demo (Streamlit App)

👉 **Streamlit App:**  
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://terrachange.streamlit.app)


> ⚠️ Note: The first launch may take 1–2 minutes while pretrained model weights are downloaded. This happens only once.

---

## 🧠 Method Overview

1. **Self-Supervised Pretraining (SimCLR)**  
   - Learns robust feature representations from unlabeled satellite images  
2. **Shared Siamese Encoder (ResNet-50)**  
   - Ensures consistent feature extraction across time (T1, T2)  
3. **Feature Differencing**  
   - Suppresses unchanged regions and highlights structural changes  
4. **U-Net Decoder**  
   - Produces dense pixel-level change probability maps  

---

## 📊 Dataset

### LEVIR-CD Dataset

- **Name:** LEVIR-CD (LEVIR Change Detection Dataset)  
- **Description:** High-resolution satellite image pairs with pixel-level change annotations, commonly used for building and urban change detection research.  
- **Resolution:** 1024 × 1024  
- **Use Case:** Urban expansion, land-use change detection  

🔗 **Official Dataset Link:**  
https://justchenhao.github.io/LEVIR/

> ⚠️ The dataset is **not included in this repository** due to size constraints.

---

## 🧪 Model Architecture

- **Encoder:** ResNet-50  
- **Pretraining:** SimCLR (contrastive self-supervised learning)  
- **Change Detection:** Siamese feature differencing  
- **Decoder:** U-Net style multi-scale upsampling  

A visual overview of the architecture is available inside the app under **Model Architecture**.

---

## 🖥️ Tech Stack

- Python  
- PyTorch  
- Streamlit  
- NumPy / OpenCV  
- Self-Supervised Learning (SimCLR)  

---

## ☁️ Deployment Notes

- Large model weights (`.pth`) are **not stored in GitHub**
- Models are hosted externally and **downloaded at runtime**
- This ensures:
  - GitHub size limits are respected  
  - Streamlit Cloud deployment is stable  

---

## 🎯 Applications

- Urban expansion monitoring  
- Disaster damage assessment  
- Environmental and land-use change analysis  
- Remote sensing research & demos  

---

## 📌 Citation / Acknowledgements

- **SimCLR:** Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*  
- **LEVIR-CD Dataset:** Chen & Shi, 2020  

---

## 👤 Author

**[Ankit Halder](https://github.com/DtheFather)**

B.Tech CSE | AI, ML & Cybersecurity Enthusiast  

---

⭐ If you find this project useful, consider giving it a star!
