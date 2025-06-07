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