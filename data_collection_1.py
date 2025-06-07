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