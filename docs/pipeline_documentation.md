# Washing Machine Anomaly Detection

## Table of Contents
1. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
2. [Model Development](#model-development)
3. [Deployment](#deployment)
4. [Continuous Learning](#continuous-learning)
5. [Monitoring Dashboard](#monitoring-dashboard)
6. [Complete Workflow](#complete-workflow)

## Data Collection and Preprocessing

### 1. Data Collection Script

```python
# collect_washing_machine_data.py
import serial
import numpy as np
import pandas as pd
from datetime import datetime
import time
import json
import os
from tqdm import tqdm

class WashingMachineDataCollector:
    def __init__(self, port, baudrate=115200, output_dir='collected_data'):
        self.port = port
        self.baudrate = baudrate
        self.output_dir = output_dir
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for data storage"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labeled'), exist_ok=True)
        
    def connect_serial(self):
        """Establish serial connection to ESP32"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port} at {self.baudrate} baud")
            # Wait for Arduino to reset
            time.sleep(2)
            self.ser.flushInput()
            return True
        except Exception as e:
            print(f"Error connecting to {self.port}: {e}")
            return False
    
    # ... [rest of the class methods] ...

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect washing machine sensor data')
    parser.add_argument('--port', type=str, required=True, 
                       help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Duration in minutes')
    parser.add_argument('--label', type=str, default='normal', 
                       choices=['normal', 'unbalanced', 'bearing_wear', 'belt_slip', 'motor_issue'],
                       help='Label for this data collection')
    parser.add_argument('--machine', type=str, default='machine1', 
                       help='Machine identifier')
    
    args = parser.parse_args()
    
    collector = WashingMachineDataCollector(args.port)
    collector.collect_data(
        duration_minutes=args.duration,
        label=args.label,
        machine_id=args.machine
    )
```

### 2. Data Cleaning and Preprocessing

```python
# preprocess_data.py
import os
import numpy as np
import pandas as pd
from scipy import signal
import pywt
from tqdm import tqdm
import json
import glob

class WashingMachineDataPreprocessor:
    def __init__(self, data_dir='collected_data'):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
    
    # ... [class methods] ...

def kurtosis(x):
    """Calculate kurtosis of a signal"""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return np.sum((x - mean)**4) / (n * std**4) - 3

def skew(x):
    """Calculate skewness of a signal"""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return np.sum((x - mean)**3) / (n * std**3)

if __name__ == "__main__":
    preprocessor = WashingMachineDataPreprocessor()
    preprocessor.process_all_data()
```

## Model Development

### 1. Training the Model

```python
# train_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_labeled_data(data_dir='collected_data/labeled'):
    """Load all labeled data files"""
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet') and f.startswith('labeled_')]
    if not files:
        raise FileNotFoundError("No labeled data found. Run the labeling tool first.")
    
    dfs = []
    for file in files:
        df = pd.read_parquet(os.path.join(data_dir, file))
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

# ... [rest of the training code] ...
```

### 2. Model Conversion

```python
# convert_to_tflite.py
import tensorflow as tf
import numpy as np
import joblib
import os

def convert_model(model_path, output_dir='model'):
    """Convert a saved Keras model to TensorFlow Lite"""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Create a representative dataset for quantization
    def representative_data_gen():
        # Load some training data for calibration
        scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
        # Generate random data with the same shape as training data
        for _ in range(100):
            dummy_input = np.random.randn(1, *model.input_shape[1:]).astype(np.float32)
            dummy_input = scaler.transform(dummy_input)
            yield [dummy_input]
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert and save
    tflite_model = converter.convert()
    
    # Save the model
    tflite_model_path = os.path.join(output_dir, 'model_quant.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to {tflite_model_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model_path

if __name__ == "__main__":
    convert_model('model/best_model.h5')
```

## Deployment

### 1. Convert to C Array

```python
# convert_to_c_array.py
def convert_to_c_array(tflite_path, output_path='arduino/model.h'):
    """Convert a TFLite model to a C array"""
    with open(tflite_path, 'rb') as f:
        model_content = f.read()
    
    with open(output_path, 'w') as f:
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("#include <cstdint>\n\n")
        f.write("const unsigned char model_tflite[] = {\n")
        
        # Write 12 bytes per line
        for i in range(0, len(model_content), 12):
            chunk = model_content[i:i+12]
            f.write("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        
        f.write("};\n")
        f.write(f"const unsigned int model_tflite_len = {len(model_content)};\n")
        f.write("\n#endif // MODEL_DATA_H\n")

if __name__ == "__main__":
    convert_to_c_array('model/model_quant.tflite')
```

### 2. Arduino Project Structure

```
washing_machine_monitor/
├── washing_machine_monitor.ino
├── model.h              # Generated by convert_to_c_array.py
└── src/
    ├── config.h         # Configuration constants
    ├── sensors.h        # Sensor reading functions
    ├── tflite_helper.h  # TensorFlow Lite helper functions
    └── utils.h          # Utility functions
```

## Continuous Learning

### 1. Collect New Data

```python
# collect_new_data.py
import pandas as pd
from datetime import datetime
import os

def collect_new_data(port, duration_minutes=60, output_dir='new_data'):
    """Collect new data for continuous learning"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the same data collection script
    from collect_washing_machine_data import WashingMachineDataCollector
    
    collector = WashingMachineDataCollector(port, output_dir=output_dir)
    data = collector.collect_data(duration_minutes=duration_minutes, 
                                 label='unlabeled',
                                 machine_id='production')
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"production_{timestamp}.parquet")
    data.to_parquet(output_file)
    
    return output_file
```

## Monitoring Dashboard

```python
# dashboard.py
from flask import Flask, render_template, jsonify
import pandas as pd
import os
from datetime import datetime, timedelta
import json

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    # Read the latest log file
    log_dir = 'collected_data/logs'
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.csv')])
    
    if not log_files:
        return jsonify({"error": "No log files found"}), 404
    
    latest_log = os.path.join(log_dir, log_files[-1])
    df = pd.read_csv(latest_log)
    
    # Get data from last hour
    one_hour_ago = datetime.now() - timedelta(hours=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    recent = df[df['timestamp'] > one_hour_ago]
    
    return jsonify({
        'timestamps': recent['timestamp'].dt.strftime('%H:%M:%S').tolist(),
        'anomaly_scores': recent['anomaly_score'].tolist(),
        'avg_scores': recent['avg_score'].tolist(),
        'current_status': "NORMAL" if recent['avg_score'].iloc[-1] < ANOMALY_THRESHOLD else "ANOMALY DETECTED"
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
```

## Complete Workflow

### 1. Data Collection
```bash
# Collect normal operation data (30 minutes)
python collect_washing_machine_data.py --port COM3 --duration 30 --label normal --machine washer1

# Collect anomaly data (if possible, 10 minutes each)
python collect_washing_machine_data.py --port COM3 --duration 10 --label unbalanced --machine washer1
```

### 2. Data Preprocessing
```bash
python preprocess_data.py
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Convert to TFLite
```bash
python convert_to_tflite.py
```

### 5. Generate C Array
```bash
python convert_to_c_array.py
```

### 6. Deploy to ESP32
1. Copy `model.h` to your Arduino project
2. Upload the sketch to your ESP32
3. Monitor the serial output for anomalies

### 7. Continuous Learning (Optional)
```bash
# Collect new data
python collect_new_data.py --port COM3 --duration 60

# Retrain model with new data
python retrain_model.py
```

### 8. Monitor with Dashboard
```bash
python dashboard.py
```

This documentation provides a complete guide to implementing the washing machine anomaly detection system, from data collection to deployment and monitoring.
