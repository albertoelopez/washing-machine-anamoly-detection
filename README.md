# Washing Machine Anomaly Detection

[![TensorFlow Lite Micro](https://img.shields.io/badge/TensorFlow%20Lite-Micro-orange)](https://github.com/tensorflow/tflite-micro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ESP32](https://img.shields.io/badge/ESP32-ESP--IDF-blue)](https://www.espressif.com/en/products/socs/esp32)

An end-to-end system for detecting anomalies in washing machine operation using machine learning on embedded devices. This project implements a robust and efficient anomaly detection system that can run on resource-constrained hardware like the ESP32.

## Built with

- [TensorFlow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro/tree/main?tab=readme-ov-file#tensorflow-lite-for-microcontrollers)
- [Google's ML on Microcontrollers](https://ai.google.dev/edge/litert/microcontrollers/overview)

## 🚀 Features

- **Real-time Sensor Data Collection**
  - High-frequency accelerometer data (up to 100Hz)
  - Temperature monitoring (ambient and object)
  - Audio analysis for vibration and noise detection
  - Robust error handling and sensor fault detection

- **Advanced Machine Learning**
  - TensorFlow Lite Micro for on-device inference
  - Support for both quantized and float models
  - Configurable anomaly detection thresholds
  - Fallback mechanisms for when ML is unavailable

- **Power Management**
  - Deep sleep mode for power efficiency
  - Configurable sleep timers
  - Automatic wake-on-activity

- **Robust Operation**
  - Watchdog timer for system stability
  - Error recovery mechanisms
  - Detailed logging and debugging
  - Persistent storage of critical data

- **Development Tools**
  - Detailed memory usage reporting
  - TensorFlow Lite profiling
  - Serial debug output configuration
  - Over-the-air (OTA) update ready

## 📦 Project Structure

```
washing-machine-anomaly-detection/
├── arduino/                  # ESP32 firmware
│   ├── src/                   # Source files
│   │   ├── config.h           # Configuration constants and build flags
│   │   ├── sensors.h          # Sensor interface and data collection
│   │   ├── tflite_helper.h    # TensorFlow Lite Micro integration
│   │   └── utils.h            # Utility functions and helpers
│   ├── platformio.ini         # PlatformIO build configuration
│   └── main.ino               # Main application loop and state machine
```

### Key Components

#### `config.h`
- Centralized configuration management
- Feature toggles (e.g., `ENABLE_DEEP_SLEEP`, `ENABLE_TFLITE_PROFILING`)
- Hardware pin definitions
- Performance tuning parameters
- Debug and logging controls

#### `sensors.h`
- Unified sensor interface
- Automatic sensor initialization and validation
- Error detection and recovery
- Power management for sensors
- Calibration routines

#### `tflite_helper.h`
- TensorFlow Lite Micro integration
- Memory management and optimization
- Model loading and validation
- Input/output tensor handling
- Performance profiling and debugging

#### `main.ino`
- Main application loop
- State machine for operation modes
- Watchdog timer integration
- Error handling and recovery
- System monitoring and reporting
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

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/washing-machine-anomaly-detection.git
   cd washing-machine-anomaly-detection
   ```

2. Install PlatformIO:
   ```bash
   # Using pip (recommended)
   pip install -U platformio
   
   # Or using pipx
   pipx install platformio
   ```

3. Configure the project:
   - Update `arduino/platformio.ini` with your board settings
   - Adjust `arduino/src/config.h` for your specific hardware and requirements

4. Install dependencies:
   ```bash
   cd arduino
   pio lib install
   ```

5. Build and upload:
   ```bash
   # Build the project
   pio run
   
   # Upload to connected device
   pio run -t upload
   
   # Monitor serial output
   pio device monitor
   ```

### Configuration Options

#### Power Management
- `ENABLE_DEEP_SLEEP`: Enable/disable deep sleep mode
- `DEEP_SLEEP_TIMEOUT_MS`: Inactivity timeout before entering deep sleep
- `SAMPLE_INTERVAL_MS`: Sensor sampling interval (default: 10ms)
- `INFERENCE_INTERVAL_MS`: ML model inference interval (default: 1000ms)

#### Debugging
- `ENABLE_SERIAL_DEBUG`: Enable detailed debug output
- `ENABLE_TFLITE_PROFILING`: Enable TensorFlow Lite profiling
- `WDT_TIMEOUT_S`: Watchdog timer timeout in seconds (default: 30s)

#### Hardware
- `I2C_SDA`, `I2C_SCL`: I2C pin definitions
- `SD_CS`: SD card chip select pin
- `LED_PIN`: Status LED pin

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

### Documentation
- [TensorFlow Lite Micro Documentation](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32 Technical Reference](https://www.espressif.com/sites/default/files/documentation/esp32_technical_reference_manual_en.pdf)
- [PlatformIO Documentation](https://docs.platformio.org/)

### Related Projects
- [TensorFlow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro)
- [ESP-IDF](https://github.com/espressif/esp-idf)
- [Arduino-ESP32](https://github.com/espressif/arduino-esp32)

### References
- [TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers](https://www.oreilly.com/library/view/tinyml/9781492052036/)
- [ESP32 Deep Sleep Modes](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/sleep_modes.html)
- [TensorFlow Lite Model Optimization](https://www.tensorflow.org/lite/performance/model_optimization)

---

<div align="center">
  <p>Made with ❤️ for embedded ML</p>
  <p>Maintained by the Washing Machine Anomaly Detection Team</p>
</div>

## 📊 Performance Optimization

### Memory Management
- Tensor arena size is configurable via `TENSOR_ARENA_SIZE`
- Memory alignment optimized for ESP32
- Stack and heap usage monitoring

### Power Optimization
- Deep sleep between operations
- Sensor power management
- Clock gating and peripheral control

### ML Model Optimization
- Quantized model support (int8/uint8)
- Operator fusion
- Custom kernels for ESP32

## 🐛 Debugging

### Common Issues
1. **Insufficient Tensor Arena**
   - Symptom: Model fails to initialize or crashes
   - Solution: Increase `TENSOR_ARENA_SIZE` in `config.h`

2. **Sensor Initialization Failures**
   - Check I2C connections and addresses
   - Verify power supply stability
   - Review sensor initialization sequence

3. **Watchdog Resets**
   - Ensure all tasks complete within watchdog timeout
   - Add `esp_task_wdt_reset()` in long-running loops
   - Consider increasing `WDT_TIMEOUT_S` if needed

### Serial Debug Output
```
=== System Status ===
Uptime: 1234s
Heap: 120000 / 320000 bytes free
Min free heap: 85000 bytes
Tensor arena usage: 8.2KB / 12.0KB (68.3%)
Anomaly count: 2
Alert active: NO
====================
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Report bugs by opening an issue
2. Suggest new features or improvements
3. Submit pull requests with bug fixes or new features
4. Improve documentation
5. Share your use cases and success stories

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use descriptive variable and function names
- Add comments for complex logic
- Include error handling for all external calls
- Keep functions small and focused

## 🙏 Acknowledgments

- TensorFlow Lite for Microcontrollers
- PlatformIO for embedded development
- Adafruit for sensor libraries
