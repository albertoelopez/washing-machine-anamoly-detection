# Washing Machine Anomaly Detection

## Data Collection and Preprocessing Pipeline

I'll help you set up a complete data collection, cleaning, and labeling pipeline. Here's how we'll approach it:

## 1. Data Collection Script

First, let's create a robust data collection script that handles all sensors:

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
    
    def collect_data(self, duration_minutes=60, label="normal", machine_id="default"):
        """Collect data for specified duration"""
        if not hasattr(self, 'ser') or not self.ser.is_open:
            if not self.connect_serial():
                return None
        
        samples_per_second = 100  # Adjust based on your ESP32 settings
        total_samples = duration_minutes * 60 * samples_per_second
        window_size = 100  # 1 second of data at 100Hz
        
        # Initialize data buffer
        data = {
            'timestamp': [],
            'accel_x': [], 'accel_y': [], 'accel_z': [],
            'temperature': [],
            'audio_rms': [],
            'label': label,
            'machine_id': machine_id
        }
        
        print(f"Collecting {duration_minutes} minutes of {label} data...")
        start_time = time.time()
        
        try:
            with tqdm(total=total_samples, unit='samples') as pbar:
                while len(data['timestamp']) < total_samples:
                    if self.ser.in_waiting:
                        line = self.ser.readline().decode('utf-8').strip()
                        if line.startswith('DATA:'):  # Expected format: DATA,accel_x,accel_y,accel_z,temp,audio_rms
                            try:
                                parts = line.split(',')
                                if len(parts) == 6:  # DATA + 5 values
                                    data['timestamp'].append(time.time())
                                    data['accel_x'].append(float(parts[1]))
                                    data['accel_y'].append(float(parts[2]))
                                    data['accel_z'].append(float(parts[3]))
                                    data['temperature'].append(float(parts[4]))
                                    data['audio_rms'].append(float(parts[5]))
                                    pbar.update(1)
                            except (ValueError, IndexError) as e:
                                print(f"Error parsing line: {line} - {e}")
                    
                    # Small delay to prevent CPU overuse
                    time.sleep(0.001)
                    
        except KeyboardInterrupt:
            print("\nData collection stopped by user")
        except Exception as e:
            print(f"Error during data collection: {e}")
        finally:
            # Save collected data
            if len(data['timestamp']) > 0:
                self.save_data(data, label, machine_id)
            return data
    
    def save_data(self, data, label, machine_id):
        """Save collected data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{machine_id}_{label}_{timestamp}.parquet"
        filepath = os.path.join(self.output_dir, 'raw', filename)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save as Parquet (efficient binary format)
        df.to_parquet(filepath)
        print(f"Saved {len(df)} samples to {filepath}")
        
        # Also save metadata
        metadata = {
            'machine_id': machine_id,
            'label': label,
            'start_time': datetime.fromtimestamp(data['timestamp'][0]).isoformat(),
            'end_time': datetime.fromtimestamp(data['timestamp'][-1]).isoformat(),
            'sample_count': len(df),
            'sample_rate_hz': 100,  # Update this based on your actual sample rate
            'sensors': ['accel_xyz', 'temperature', 'microphone']
        }
        
        metadata_path = filepath.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect washing machine sensor data')
    parser.add_argument('--port', type=str, required=True, help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
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
2. Data Cleaning and Preprocessing
Let's create a script to clean and preprocess the collected data:

python
CopyInsert
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
        
    def load_raw_data(self, file_pattern='*.parquet'):
        """Load all raw data files matching pattern"""
        files = glob.glob(os.path.join(self.raw_dir, file_pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching {file_pattern}")
            
        all_data = []
        for file in tqdm(files, desc="Loading files"):
            try:
                # Load data
                df = pd.read_parquet(file)
                
                # Load corresponding metadata
                metadata_file = file.replace('.parquet', '_metadata.json')
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Add metadata to dataframe
                for key, value in metadata.items():
                    if key not in df.columns:
                        df[key] = value
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not all_data:
            raise ValueError("No valid data files found")
            
        return pd.concat(all_data, ignore_index=True)
    
    def clean_data(self, df):
        """Clean the raw sensor data"""
        print("Cleaning data...")
        
        # 1. Remove duplicate timestamps
        df = df.drop_duplicates('timestamp')
        
        # 2. Handle missing values (linear interpolation)
        df = df.interpolate(method='linear')
        
        # 3. Remove outliers using IQR
        sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'temperature', 'audio_rms']
        for col in sensor_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                df = df[mask].copy()
        
        # 4. Resample to consistent frequency
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('datetime')
            df = df.resample('10ms').mean()  # 100Hz
            df = df.interpolate(method='linear')
            df = df.reset_index()
        
        return df
    
    def extract_features(self, df, window_size=100, step=50):
        """Extract time and frequency domain features"""
        print("Extracting features...")
        
        sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'audio_rms']
        features = []
        labels = []
        metadata = []
        
        # Group by label and machine if available
        group_cols = []
        if 'label' in df.columns:
            group_cols.append('label')
        if 'machine_id' in df.columns:
            group_cols.append('machine_id')
            
        if not group_cols:
            # If no grouping, process entire dataset as one
            groups = [('all', df)]
        else:
            groups = df.groupby(group_cols)
        
        for group_name, group_df in groups:
            # Convert to numpy for faster processing
            data = group_df[sensor_cols].values
            n_samples = len(data)
            
            # Process in windows
            for i in range(0, n_samples - window_size + 1, step):
                window = data[i:i+window_size]
                window_features = self._extract_window_features(window)
                
                features.append(window_features)
                
                # Store metadata
                if isinstance(group_name, tuple):
                    meta = dict(zip(group_cols, group_name))
                else:
                    meta = {group_cols[0]: group_name}
                
                meta['window_start'] = group_df.iloc[i].name if hasattr(group_df, 'index') else i
                metadata.append(meta)
        
        # Create feature matrix
        feature_cols = []
        for sensor in sensor_cols:
            feature_cols.extend([
                f"{sensor}_mean", f"{sensor}_std", f"{sensor}_max", 
                f"{sensor}_min", f"{sensor}_rms", f"{sensor}_kurtosis",
                f"{sensor}_skew", f"{sensor}_energy", f"{sensor}_entropy"
            ])
            
            # FFT features
            for j in range(5):  # First 5 FFT coefficients
                feature_cols.append(f"{sensor}_fft_{j}")
        
        features_df = pd.DataFrame(features, columns=feature_cols)
        metadata_df = pd.DataFrame(metadata)
        
        # Combine features and metadata
        result = pd.concat([metadata_df.reset_index(drop=True), features_df], axis=1)
        return result
    
    def _extract_window_features(self, window):
        """Extract features from a single window of sensor data"""
        features = []
        
        # Time domain features
        for i in range(window.shape[1]):  # For each sensor
            signal = window[:, i]
            
            # Basic statistics
            mean = np.mean(signal)
            std = np.std(signal)
            maximum = np.max(signal)
            minimum = np.min(signal)
            rms = np.sqrt(np.mean(signal**2))
            kurt = kurtosis(signal)
            skew_val = skew(signal)
            energy = np.sum(signal**2)
            
            # Entropy
            hist, _ = np.histogram(signal, bins=10, density=True)
            hist = hist[hist != 0]
            entropy = -np.sum(hist * np.log2(hist))
            
            # Frequency domain (FFT)
            fft_vals = np.abs(np.fft.fft(signal - mean))[:5]  # First 5 coefficients
            fft_vals = np.pad(fft_vals, (0, max(0, 5 - len(fft_vals))), 'constant')
            
            # Combine all features
            window_features = [
                mean, std, maximum, minimum, rms, kurt, skew_val, energy, entropy
            ] + list(fft_vals)
            
            features.extend(window_features)
        
        return np.concatenate(features)
    
    def process_all_data(self, output_file='processed_features.parquet'):
        """Process all raw data files and save features"""
        # Load and clean data
        df = self.load_raw_data()
        df_clean = self.clean_data(df)
        
        # Extract features
        features_df = self.extract_features(df_clean)
        
        # Save processed data
        output_path = os.path.join(self.processed_dir, output_file)
        features_df.to_parquet(output_path)
        print(f"Saved processed features to {output_path}")
        
        # Save column names for later use
        with open(os.path.join(self.processed_dir, 'feature_columns.json'), 'w') as f:
            json.dump(features_df.columns.tolist(), f)
        
        return features_df

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
3. Labeling Tool
Let's create a simple tool to help label your data:

python
CopyInsert
# label_data.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import json

class DataLabelingTool:
    def __init__(self, data_dir='collected_data'):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.labeled_dir = os.path.join(data_dir, 'labeled')
        os.makedirs(self.labeled_dir, exist_ok=True)
        
        # Load processed data
        self.data_files = self._find_data_files()
        self.current_file_idx = 0
        self.current_data = None
        self.current_labels = None
        self.window_size = 1000  # 10 seconds at 100Hz
        
        # Label mapping
        self.label_map = {
            '0': 'normal',
            '1': 'unbalanced',
            '2': 'bearing_wear',
            '3': 'belt_slip',
            '4': 'motor_issue',
            '9': 'unknown'
        }
        
    def _find_data_files(self):
        """Find all processed data files"""
        return glob.glob(os.path.join(self.processed_dir, '*.parquet'))
    
    def load_data(self, file_idx):
        """Load data for labeling"""
        file_path = self.data_files[file_idx]
        self.current_data = pd.read_parquet(file_path)
        self.current_labels = np.full(len(self.current_data), 'unknown')
        self.current_file = os.path.basename(file_path)
        return self.current_data
    
    def plot_segment(self, start_idx):
        """Plot a segment of the data for labeling"""
        plt.close('all')
        fig, axs = plt.subplots(4, 1, figsize=(15, 10))
        fig.suptitle(f"Segment {start_idx//self.window_size + 1}/{(len(self.current_data)//self.window_size)+1}")
        
        end_idx = min(start_idx + self.window_size, len(self.current_data))
        segment = self.current_data.iloc[start_idx:end_idx]
        
        # Plot each sensor
        sensors = ['accel_x', 'accel_y', 'accel_z', 'audio_rms']
        for i, sensor in enumerate(sensors):
            if sensor in segment.columns:
                axs[i].plot(segment.index, segment[sensor])
                axs[i].set_ylabel(sensor)
        
        # Add buttons for labeling
        button_axs = []
        buttons = []
        
        def make_label_callback(label):
            def callback(event):
                self.current_labels[start_idx:end_idx] = label
                print(f"Labeled segment as {label}")
                plt.close()
            return callback
        
        # Create buttons for each label
        for i, (key, label) in enumerate(self.label_map.items()):
            ax = plt.axes([0.1 + i*0.15, 0.02, 0.12, 0.05])
            button = Button(ax, f"{key}: {label}")
            button.on_clicked(make_label_callback(label))
            button_axs.append(ax)
            buttons.append(button)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def label_all_segments(self):
        """Label all segments in the current file"""
        if self.current_data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        for i in range(0, len(self.current_data), self.window_size):
            self.plot_segment(i)
        
        # Add labels to dataframe and save
        self.current_data['label'] = self.current_labels[:len(self.current_data)]
        
        # Save labeled data
        output_file = os.path.join(self.labeled_dir, f"labeled_{self.current_file}")
        self.current_data.to_parquet(output_file)
        print(f"Saved labeled data to {output_file}")
        
        # Save label mapping
        label_map_file = os.path.join(self.labeled_dir, 'label_mapping.json')
        with open(label_map_file, 'w') as f:
            json.dump(self.label_map, f, indent=2)
        
        return output_file

if __name__ == "__main__":
    # Example usage
    labeler = DataLabelingTool()
    
    # Load the first data file
    if labeler.data_files:
        labeler.load_data(0)
        labeler.label_all_segments()
    else:
        print("No data files found. Run preprocessing first.")
4. How to Use This Pipeline
Step 1: Collect Data
bash
CopyInsert
# Collect normal operation data
python collect_washing_machine_data.py --port COM3 --duration 30 --label normal --machine washer1

# Collect data with known issues (if possible)
python collect_washing_machine_data.py --port COM3 --duration 10 --label unbalanced --machine washer1
Step 2: Preprocess the Data
bash
CopyInsert in Terminal
python preprocess_data.py
Step 3: Label the Data
bash
CopyInsert
python label_data.py
# Follow the GUI to label segments of data
Step 4: Train Your Model
After labeling, you can use the labeled data to train your model. The labeled data will be saved in the collected_data/labeled/ directory.

5. Data Directory Structure
CopyInsert
collected_data/
├── raw/                    # Raw sensor data
│   ├── washer1_normal_20230606_1430.parquet
│   ├── washer1_normal_20230606_1430_metadata.json
│   └── ...
├── processed/              # Cleaned and preprocessed data
│   └── processed_features.parquet
└── labeled/               # Manually labeled data
    ├── labeled_washer1_normal_20230606_1430.parquet
    └── label_mapping.json
6. Next Steps
Collect at least 1 hour of normal operation data from your washing machine
Label the data using the labeling tool
Augment your dataset with synthetic anomalies if needed
Proceed to model training
Would you like me to explain any part of this pipeline in more detail or help you get started with the data collection?

Feedback submitted
6/6/2025, 11:57:45 PM
Me

now add the other steps in order

Revert
Complete End-to-End Washing Machine Anomaly Detection Pipeline
Here's the complete workflow with all steps in order, from data collection to deployment:

1. Data Collection & Preparation
1.1 Collect Raw Sensor Data
bash
CopyInsert
# Install required packages
pip install pyserial pandas pyarrow tqdm numpy scipy

# Collect normal operation data (30 minutes)
python collect_washing_machine_data.py --port COM3 --duration 30 --label normal --machine washer1

# Collect anomaly data (if possible, 10 minutes each)
python collect_washing_machine_data.py --port COM3 --duration 10 --label unbalanced --machine washer1
python collect_washing_machine_data.py --port COM3 --duration 10 --label bearing_wear --machine washer1
1.2 Preprocess Data
bash
CopyInsert
# Clean and extract features
python preprocess_data.py
1.3 Label Data
bash
CopyInsert
# Launch the labeling tool
python label_data.py
2. Model Development
2.1 Train Initial Model
python
CopyInsert
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

def prepare_data(df, test_size=0.2, val_size=0.1):
    """Prepare data for training"""
    # Separate features and labels
    X = df.drop(columns=['label', 'machine_id', 'window_start'], errors='ignore')
    y = pd.get_dummies(df['label'])  # Convert to one-hot encoding
    
    # Split into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Further split training set into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=val_size/(1-test_size), 
        random_state=42,
        stratify=y_train
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')
    
    return (X_train_scaled, y_train, X_val_scaled, y_val, 
            X_test_scaled, y_test, scaler)

def create_model(input_shape, num_classes):
    """Create a CNN model for time series classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape[0]//100, 100, 1), 
                              input_shape=(input_shape[0],)),
        
        # First Conv block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Second Conv block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train():
    # Load and prepare data
    df = load_labeled_data()
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(df)
    
    # Create and compile model
    model = create_model(X_train.shape[1:], y_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'model/best_model.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save the final model
    model.save('model/final_model.h5')
    
    return model, history

if __name__ == "__main__":
    train()
2.2 Convert to TensorFlow Lite
python
CopyInsert
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
        # This should match your training data preprocessing
        scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
        # Generate random data with the same shape as training data
        # Replace this with actual data samples for better results
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
2.3 Convert to C Array
python
CopyInsert
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
3. Deploy to ESP32
3.1 Install Required Arduino Libraries
Install the following libraries via Arduino Library Manager:
Arduino_TensorFlowLite
Adafruit_MLX90614
Adafruit_BusIO
Adafruit_Sensor
Adafruit_MPU6050
3.2 Arduino Project Structure
CopyInsert
washing_machine_monitor/
├── washing_machine_monitor.ino
├── model.h              # Generated by convert_to_c_array.py
└── src/
    ├── config.h         # Configuration constants
    ├── sensors.h        # Sensor reading functions
    ├── tflite_helper.h  # TensorFlow Lite helper functions
    └── utils.h          # Utility functions
3.3 Main Arduino Sketch
cpp
CopyInsert
// washing_machine_monitor.ino
#include "src/config.h"
#include "src/sensors.h"
#include "src/tflite_helper.h"
#include "src/utils.h"

// Global variables
float features[NUM_FEATURES] = {0};
float anomaly_scores[ANOMALY_WINDOW] = {0};
int score_index = 0;
bool alert_triggered = false;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  // Initialize hardware
  setupSensors();
  setupTensorFlow();
  setupSDCard();
  
  Serial.println("Washing Machine Monitor Ready");
}

void loop() {
  static unsigned long lastInference = 0;
  static float feature_buffer[FEATURE_WINDOW][NUM_FEATURES];
  static int buffer_index = 0;
  
  // 1. Read sensors
  readSensors(features);
  
  // 2. Add to feature buffer
  memcpy(feature_buffer[buffer_index], features, sizeof(features));
  buffer_index = (buffer_index + 1) % FEATURE_WINDOW;
  
  // 3. Run inference every second
  if (millis() - lastInference >= 1000) {
    lastInference = millis();
    
    // 4. Prepare input tensor
    float* input_data = tflInterpreter->input(0)->data.f;
    for (int i = 0; i < FEATURE_WINDOW; i++) {
      for (int j = 0; j < NUM_FEATURES; j++) {
        *input_data++ = feature_buffer[(buffer_index + i) % FEATURE_WINDOW][j];
      }
    }
    
    // 5. Run inference
    float anomaly_score = runInference();
    
    // 6. Update moving average
    anomaly_scores[score_index] = anomaly_score;
    score_index = (score_index + 1) % ANOMALY_WINDOW;
    
    float avg_score = 0;
    for (int i = 0; i < ANOMALY_WINDOW; i++) {
      avg_score += anomaly_scores[i];
    }
    avg_score /= ANOMALY_WINDOW;
    
    // 7. Check for anomaly
    if (avg_score > ANOMALY_THRESHOLD && !alert_triggered) {
      alert_triggered = true;
      char alert[100];
      snprintf(alert, sizeof(alert), "ALERT! Anomaly detected. Score: %.4f", avg_score);
      Serial.println(alert);
      logToSD(alert);
      blinkLED(5, 200);
    } else if (avg_score <= ANOMALY_THRESHOLD) {
      alert_triggered = false;
    }
    
    // 8. Log data
    char log_entry[256];
    snprintf(log_entry, sizeof(log_entry), 
             "%lu,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f",
             millis(),
             features[0], features[1], features[2],  // accel x,y,z
             features[3], features[4],               // temp, audio
             anomaly_score, avg_score);
    logToSD(log_entry);
  }
}
4. Continuous Learning (Optional)
4.1 Collect New Data
python
CopyInsert
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
4.2 Retrain Periodically
python
CopyInsert
# retrain_model.py
import os
import pandas as pd
import joblib
from train_model import load_labeled_data, prepare_data, create_model

def load_new_data(new_data_dir='new_data', label='normal'):
    """Load newly collected data"""
    files = [f for f in os.listdir(new_data_dir) if f.endswith('.parquet')]
    if not files:
        return None
    
    dfs = []
    for file in files:
        df = pd.read_parquet(os.path.join(new_data_dir, file))
        df['label'] = label  # Assume normal unless labeled otherwise
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def retrain_model():
    """Retrain model with new data"""
    # Load existing model
    model = tf.keras.models.load_model('model/final_model.h5')
    
    # Load existing data
    existing_data = load_labeled_data()
    
    # Load new data
    new_data = load_new_data()
    if new_data is None:
        print("No new data to retrain with")
        return
    
    # Combine datasets
    all_data = pd.concat([existing_data, new_data], ignore_index=True)
    
    # Prepare data
    X_train, y_train, X_val, y_val, _, _, _ = prepare_data(all_data)
    
    # Retrain with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train on combined dataset
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32
    )
    
    # Save retrained model
    model.save('model/retrained_model.h5')
    print("Model retrained and saved")
    
    # Convert to TFLite
    from convert_to_tflite import convert_model
    convert_model('model/retrained_model.h5')

if __name__ == "__main__":
    retrain_model()
5. Monitoring Dashboard (Optional)
Create a simple web dashboard to monitor the washing machine:

python
CopyInsert
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
6. Complete Workflow Summary
Data Collection
Collect sensor data using collect_washing_machine_data.py
Label data using the labeling tool
Model Development
Train initial model: python train_model.py
Convert to TFLite: python convert_to_tflite.py
Generate C array: python convert_to_c_array.py
Deployment
Copy generated model.h to Arduino project
Upload sketch to ESP32
Monitor serial output
Continuous Learning
Collect new data in production
Periodically retrain model with new data
Deploy updated model
Monitoring
Use the web dashboard to monitor in real-time
Set up alerts for anomalies