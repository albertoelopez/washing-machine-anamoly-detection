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