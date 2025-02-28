# 🏥 AI-Powered Medical Diagnosis System  

## 📌 Overview  
This project builds an **AI-driven medical diagnosis system** to detect diseases like **pneumonia and skin cancer** using **CNNs, Vision Transformers, and Grad-CAM explainability**. It includes **real-time inference via FastAPI and deployment on AWS Lambda**.

## 📂 Repository Structure  
📂 medical-diagnosis-ai/
│── 📁 data/ → Medical datasets (X-ray, MRI, skin lesion images)
│── 📁 notebooks/ → Jupyter notebooks for training models
│── 📁 models/ → Trained models (ResNet, Swin Transformer, EfficientNet)
│── 📁 api/ → FastAPI-based prediction API
│── 📁 deployment/ → AWS Lambda & Docker deployment scripts
│── 📁 explainability/ → Grad-CAM visualization scripts
│── 📄 requirements.txt → Dependencies
│── 📄 README.md → Project documentation
│── 📄 app.py → FastAPI endpoint for real-time diagnosis
│── 📄 train.py → Model training script


---

## 📊 Dataset  
- **Chest X-ray Dataset (Pneumonia Detection)** → [Download](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **ISIC Skin Cancer Dataset** → [Download](https://www.isic-archive.com/)  
- **Brain MRI Tumor Classification** → [Download](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  

---

## 🛠️ Installation & Setup  

1️⃣ **Clone the Repository**  
git clone https://github.com/your-username/medical-diagnosis-ai.git
cd medical-diagnosis-ai

2️⃣ Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Download the Dataset (Place in data/ folder)
🏋️ Model Training
Train a CNN-based diagnostic model using train.py:
For Vision Transformer:

🖥️ FastAPI Deployment (Local)
Run the API to predict diseases from medical images
uvicorn app:app --host 0.0.0.0 --port 8000
Send an image to the API for diagnosis
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -F 'file=@test_image.jpg'
☁️ AWS Lambda Deployment
1️⃣ Convert FastAPI app to AWS Lambda using Zappa

pip install zappa  
zappa init  # Configure AWS credentials  
zappa deploy  
2️⃣ Invoke the API on AWS Lambda

curl -X 'POST' 'https://your-api-url.amazonaws.com/dev/predict/' -F 'file=@test_image.jpg'
python explainability/gradcam.py --image test_image.jpg --model resnet50
🚀 Future Enhancements
✅ Add real-time edge deployment on Jetson Nano
✅ Implement multi-disease classification
✅ Fine-tune GPT-4 Vision for automated medical report generation
✨ Contributors
Your Name – Shashank Pandey
Open for contributions! Fork & submit a PR.
📜 License
This project is MIT Licensed.
