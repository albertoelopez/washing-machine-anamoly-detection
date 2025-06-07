# Washing Machine Anomaly Detection

A machine learning pipeline for detecting anomalies in washing machine operation using sensor data and deploying the model to ESP32 microcontrollers.

## 🚀 Features

- **Data Collection**: Real-time sensor data collection from ESP32
- **Data Processing**: Automated cleaning, feature extraction, and labeling
- **Model Training**: CNN-based anomaly detection model
- **Edge Deployment**: Optimized for ESP32 with TensorFlow Lite
- **Monitoring**: Real-time dashboard for anomaly visualization

## 📁 Project Structure

```
washing-machine-anomaly-detection/
├── data_cleaning_2.py       # Data preprocessing and feature extraction
├── data_collection_1.py     # ESP32 data collection script
├── labeing_tool_3.py        # Interactive data labeling tool
├── train_initial_model_2.1.py  # Model training script
├── convert_to_tenserflow_lite_2.2.py  # Model conversion to TFLite
├── convert_to_C_array_2.3.py # Convert TFLite to C array
├── main.ino                 # ESP32 Arduino code
├── monitoring_dashboard.py  # Flask-based monitoring dashboard
├── quantization_options.md  # Model optimization guide
└── uploading_fine_tuned_model.md  # Deployment instructions
```

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- Arduino IDE with ESP32 board support
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

### Hardware Requirements

- ESP32 development board
- MPU6050 accelerometer
- MLX90614 temperature sensor
- Microphone (for audio analysis)
- MicroSD card (optional, if using external storage)

## 🚦 Usage

### 1. Data Collection
```bash
python data_collection_1.py --duration 60 --label normal --machine-id washer1
```

### 2. Data Labeling
```bash
python labeing_tool_3.py
```

### 3. Model Training
```bash
python train_initial_model_2.1.py
```

### 4. Model Conversion
```bash
python convert_to_tenserflow_lite_2.2.py
python convert_to_C_array_2.3.py
```

### 5. Deploy to ESP32
1. Open `main.ino` in Arduino IDE
2. Select your ESP32 board
3. Upload the sketch

### 6. Start Monitoring Dashboard
```bash
python monitoring_dashboard.py
```

## 🧠 Model Architecture

The model uses a CNN architecture optimized for time-series classification:

```
Input (sensor data) → Conv1D → MaxPooling → Conv1D → Flatten → Dense → Output
```

## 📊 Performance

- Model size: <100KB (quantized)
- Inference time: <50ms on ESP32
- Accuracy: >95% on test set

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
