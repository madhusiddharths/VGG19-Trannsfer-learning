# Horse or Human Classifier (VGG19 Transfer Learning)

This project implements a binary image classifier to distinguish between **Horses** and **Humans** using Transfer Learning with the **VGG19** architecture. It includes both a training script and a Streamlit-based web application for easy inference.

## Features
- **Transfer Learning**: Utilizes pre-trained VGG19 weights from ImageNet.
- **Automated Dataset Handling**: Automatically downloads and extracts the "Horse or Human" dataset.
- **Interactive Web App**: A clean Streamlit interface for uploading images and viewing classification results with confidence scores.

## Project Structure
- `train.py`: Script to download the dataset, build the model using VGG19, and train it.
- `app.py`: Streamlit application for image classification.
- `requirements.txt`: List of Python dependencies.
- `model_new.h5`: The trained model weights (generated after training).

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd VGG19-Trannsfer-learning
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
To download the dataset and train the model, run:
```bash
python train.py
```
This will save the trained model as `model_new.h5`.

### 2. Running the Web App
Once the model is trained, start the Streamlit app:
```bash
streamlit run app.py
```
Open your browser to the local URL provided (usually `http://localhost:8501`) to start classifying images.

## Technologies Used
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Streamlit](https://streamlit.io/)
- [VGG19](https://keras.io/api/applications/vgg/#vgg19)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

---
*Created for major project for Smart Knower*
