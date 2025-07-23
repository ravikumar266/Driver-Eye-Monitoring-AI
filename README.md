# Driver-Eye-Monitoring-AI
eye monitoring project


# 👁️ Driver Eye Monitoring AI

This project is an **AI-powered real-time driver monitoring system** that detects drowsiness or inattention using a Convolutional Neural Network (CNN). It leverages **TensorFlow**, **OpenCV**, **FastAPI**, and **Streamlit** to offer a complete ML pipeline from backend model inference to frontend visualization.

---

## Features

- Deep learning-based eye state detection using a trained CNN.
- Real-time webcam video feed processing with OpenCV.
-  FastAPI backend for serving the ML model.
-  Streamlit frontend for live monitoring and UI interaction.
-  Alerts on drowsy or closed eyes detected.

---

## Tech Stack

| Tool         | Purpose                            |
|--------------|-------------------------------------|
| TensorFlow   | Model training & loading (`model.h5`) |
| FastAPI      | Backend API server                  |
| Streamlit    | Frontend interface                  |
| OpenCV       | Webcam video stream capture         |
| Pillow       | Image pre-processing                |
| Uvicorn      | ASGI server to run FastAPI          |
| Requests     | HTTP calls between frontend & backend |
| Python Multipart | File upload/form support in FastAPI |

---

## Running code 

### open terminal and clone the repo first


 git clone https://github.com/ravikumar266/Driver-Eye-Monitoring-AI.git

cd Driver-Eye-Monitoring-Ai


python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt


cd backend
uvicorn app:app --reload


### open new terminal but not close old terminal of fastapi

cd Driver-Eye-Monitoring-Ai  ## if required

cd frontend


 streamlit run frontend.py


### you can see a option for open cam after enter this  you can see live prediction of eye Monitoring


