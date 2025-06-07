# Basic Anomaly Detection with TensorFlow Lite Micro

This example demonstrates a robust anomaly detection system using TensorFlow Lite Micro on ESP32. It's designed to detect anomalies in sensor data from an IMU (accelerometer and gyroscope). The example follows TensorFlow Lite Micro best practices and is optimized for embedded deployment.

## Features

- Real-time sensor data processing
- TensorFlow Lite Micro for on-device inference
- Configurable anomaly detection threshold
- Visual feedback via built-in LEDs
- Serial output for debugging and monitoring
- Support for both floating-point and quantized models
- Low-power operation with deep sleep support

## Hardware Requirements

- ESP32 development board (ESP32-DevKitC, ESP-EYE, etc.)
- IMU sensor (MPU6050, MPU9250, or similar)
- Optional: External LED for visual feedback

## Software Requirements

- PlatformIO Core or Arduino IDE
- Python 3.7+ (for model training)
- TensorFlow 2.x (for model training)
- Required Python packages (see `requirements.txt`)

## Getting Started

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>/examples/basic_anomaly_detection

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Navigate to the data directory
cd data

# Run the training script
python train_model.py
```

This will generate two model files in the `output` directory:
- `model.h`: Full-precision model
- `quant_model.h`: Quantized model (recommended for ESP32)

### 3. Deploy the Model

1. Copy the generated `quant_model.h` to the `src` directory:
   ```bash
   cp output/quant_model.h ../src/model.h
   ```

2. Update the `platformio.ini` with your board settings and serial port.

3. Connect your hardware:
   - Connect the IMU to the ESP32 (I2C pins)
   - Connect an LED to the alert pin (default: GPIO 4)

4. Build and upload the code:
   ```bash
   pio run -t upload
   ```

5. Monitor the serial output:
   ```bash
   pio device monitor
   ```

## Configuration

You can customize the following parameters in `src/main.cpp`:

- `kTensorArenaSize`: Memory allocated for TensorFlow Lite (increase if you get allocation errors)
- `kNumSamples`: Number of samples per inference
- `kAnomalyThreshold`: Threshold for anomaly detection (0.0 to 1.0)
- `kLedPin` and `kAlertPin`: GPIO pins for status and alert LEDs

## How It Works

1. The ESP32 reads sensor data from the IMU at a fixed interval
2. Data is collected into a buffer of `kNumSamples`
3. The TensorFlow Lite model processes the data to detect anomalies
4. If an anomaly is detected, the alert LED is activated
5. The system enters deep sleep between inference cycles to save power

## Performance

- Model inference time: ~X ms (quantized) / ~Y ms (float)
- Memory usage: Z KB (tensor arena) + model size
- Power consumption: ~A mA during inference, ~B μA in deep sleep

## Troubleshooting

### Model Doesn't Fit in Memory

- Try using a smaller model or reducing `kNumSamples`
- Increase `kTensorArenaSize` if you have enough RAM
- Enable model quantization for smaller footprint

### Poor Detection Accuracy

- Collect more training data
- Adjust the anomaly threshold
- Review the sensor data quality and sampling rate
- Consider a more complex model architecture

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Usage

1. Open the Serial Monitor (115200 baud)
2. The device will start collecting sensor data
3. Normal operation will be indicated by a steady heartbeat LED
4. Anomalies will trigger the alert LED and be logged to serial

## File Structure

- `main.cpp` - Main application code
- `model.h` - Pre-trained TFLite model (converted to C array)
- `platformio.ini` - PlatformIO configuration
- `data/` - Example sensor data and training scripts

## Customization

- Adjust `ANOMALY_THRESHOLD` in `config.h` to change sensitivity
- Modify `SAMPLE_RATE` to change data collection frequency
- Update model architecture in the training script if needed

## License

MIT License - See LICENSE for more information
