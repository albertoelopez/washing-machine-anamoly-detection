# Washing Machine Anomaly Detection

An end-to-end system for detecting anomalies in washing machine operation using machine learning on embedded devices.

## 🚀 Features

- Real-time sensor data collection (accelerometer, temperature, audio)
- Advanced signal processing and feature extraction
- Deep learning model for anomaly detection
- Optimized for deployment on ESP32 microcontrollers
- Web-based monitoring dashboard
- Continuous learning pipeline

## 📦 Project Structure

```
washing-machine-anomaly-detection/
├── arduino/                  # ESP32 firmware
│   ├── src/                   # Source files
│   │   ├── config.h           # Configuration constants
│   │   ├── sensors.h          # Sensor reading functions
│   │   ├── tflite_helper.h    # TensorFlow Lite integration
│   │   └── utils.h            # Utility functions
│   ├── platformio.ini         # PlatformIO configuration
│   └── main.ino               # Main Arduino sketch
├── data/                      # Data storage (gitignored)
│   ├── raw/                   # Raw sensor data
│   ├── processed/             # Processed datasets
│   ├── labeled/               # Manually labeled data
│   ├── models/                # Trained models
│   └── logs/                  # Runtime logs
├── docs/                      # Documentation
├── src/                       # Python source code
│   ├── data/                  # Data collection and preprocessing
│   │   ├── __init__.py        # Data directory setup
│   │   ├── data_collection_1.py
│   │   └── data_cleaning_2.py
│   ├── model/                 # Model training and conversion
│   │   ├── train.py
│   │   ├── convert_to_tflite_2.2.py
│   │   └── convert_to_C_array_2.3.py
│   └── web/                   # Web dashboard
│       └── monitoring_dashboard.py
├── tests/                     # Unit and integration tests
├── .gitignore
└── README.md                  # This file
```

## 🛠️ Hardware Requirements

- ESP32 development board (e.g., ESP32-DevKitC)
- MPU6050 6-DoF IMU (accelerometer + gyro)
- MLX90614 IR temperature sensor
- Electret microphone with amplifier (e.g., MAX9814)
- MicroSD card module
- Jumper wires and breadboard
- Power supply (5V, 2A recommended)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PlatformIO (for Arduino development)
- Required Python packages:
  ```
  pip install -r requirements.txt
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/washing-machine-anomaly-detection.git
   cd washing-machine-anomaly-detection
   ```

2. Set up Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install PlatformIO (for Arduino development):
   - Install VS Code: https://code.visualstudio.com/
   - Install PlatformIO IDE extension

## 🏃‍♂️ Quick Start

### 1. Data Collection

```bash
# Connect your ESP32 and run:
python src/data/data_collection_1.py --port /dev/ttyUSB0 --duration 30 --label normal
```

### 2. Data Preprocessing

```bash
python src/data/data_cleaning_2.py
```

### 3. Train the Model

```bash
python src/model/train.py
```

### 4. Convert Model for Edge Deployment

```bash
# Convert to TensorFlow Lite
python src/model/convert_to_tflite_2.2.py

# Convert to C array for Arduino
python src/model/convert_to_C_array_2.3.py
```

### 5. Deploy to ESP32

1. Open the `arduino` folder in VS Code with PlatformIO
2. Copy the generated `model.h` to `arduino/src/`
3. Connect your ESP32
4. Click "Upload" in PlatformIO

### 6. Monitor the System

Start the web dashboard:
```bash
python src/web/monitoring_dashboard.py
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 🤖 Model Architecture

The system uses a Convolutional Neural Network (CNN) for time series classification:

1. Input: 100Hz sensor data (acceleration, temperature, audio)
2. Feature extraction using 1D convolutions
3. Dense layers for classification
4. Output: Anomaly probability

## 📈 Performance

- Model size: < 50KB (quantized)
- Inference time: < 10ms on ESP32
- Accuracy: > 95% on test set

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- TensorFlow Lite for Microcontrollers
- PlatformIO for embedded development
- Adafruit for sensor libraries
