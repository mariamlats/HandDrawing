"""Run explicitly once before starting the webcam application."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_asset import download_model

if __name__ == '__main__':
    path = download_model(ROOT / 'models' / 'hand_landmarker.task')
    print(f'Verified model ready: {path}')
