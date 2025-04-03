# 🧠 AI-Powered Medical Diagnostic System

[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

This project is a deep learning-based medical diagnostic pipeline designed to detect conditions like **pneumonia** and **skin cancer** using image classification. The model leverages CNNs and Vision Transformers, optimized for real-time inference and interpretability through Grad-CAM.

---

## Architecture
![Architecture Image](https://github.com/snkpgithub/AI_Powered_Medical_Diagnostic_System/blob/main/Architecture.png)

## 🚀 Features

- 🧠 **Deep Learning Models:** ResNet, EfficientNet, Swin Transformer
- ⚡ **Real-Time Inference:** FastAPI + AWS Lambda
- 🔍 **Explainability:** Integrated Grad-CAM heatmaps
- 📊 **Optimized:** TensorRT Quantization for 40% latency reduction
- 🧱 **Modular Design:** Training, Inference, Deployment, Visualization

---

## 📂 Folder Structure

```
AI_Powered_Medical_Diagnostic_System/
│
├── src/                  # Model training and inference API
├── notebooks/            # Jupyter notebooks for EDA and prototyping
├── deployment/           # Lambda handler and Dockerfile
├── visualization/        # Grad-CAM visualizer
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignore cache, models, logs
└── README.md             # You're reading it :)
```

---

## 🔧 Setup Instructions

1. **Clone this repo**
   ```bash
   git clone https://github.com/snkpgithub/AI_Powered_Medical_Diagnostic_System.git
   cd AI_Powered_Medical_Diagnostic_System
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API**
   ```bash
   uvicorn src.inference_api:app --reload
   ```

---

## Example Output

![Grad-CAM Example](https://github.com/snkpgithub/AI_Powered_Medical_Diagnostic_System/blob/main/sample_gradcam_output.png) <!-- Replace with real link if available -->

---

## 🧪 Technologies Used

- Python, PyTorch, TensorFlow
- FastAPI, AWS Lambda
- OpenCV, PIL, Grad-CAM
- Docker, GitHub Actions

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- Datasets used for prototyping: [Kaggle Chest X-rays](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- Grad-CAM reference implementation
- AWS Free Tier for deployment testing
