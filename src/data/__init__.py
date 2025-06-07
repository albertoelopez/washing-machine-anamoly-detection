"""Data module for washing machine anomaly detection."""
from pathlib import Path

# Define the data directory path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# Create data directory structure if it doesn't exist
for subdir in ['raw', 'processed', 'labeled', 'models', 'logs']:
    (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)