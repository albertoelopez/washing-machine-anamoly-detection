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