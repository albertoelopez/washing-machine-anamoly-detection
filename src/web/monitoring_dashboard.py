"""Flask web dashboard for monitoring washing machine status."""
from pathlib import Path
from flask import Flask, render_template, jsonify
import pandas as pd
from datetime import datetime, timedelta
import json
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data import DATA_DIR

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    # Read the latest log file
    log_dir = DATA_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_files = sorted(log_dir.glob('*.csv'))
    
    if not log_files:
        return jsonify({"error": "No log files found"}), 404
    
    latest_log = log_files[-1]
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