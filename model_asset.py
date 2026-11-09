"""Pinned official MediaPipe model asset; no model is downloaded on camera startup."""
import hashlib
import os
from pathlib import Path
import tempfile
import requests

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
MAX_MODEL_BYTES = 16 * 1024 * 1024


def verify_model(path):
    digest = hashlib.sha256()
    count = 0
    with Path(path).open('rb') as model:
        for chunk in iter(lambda: model.read(64 * 1024), b''):
            count += len(chunk)
            if count > MAX_MODEL_BYTES:
                raise ValueError('Hand-landmarker model exceeds the size limit.')
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        raise ValueError('Hand-landmarker model checksum does not match the pinned asset.')


def download_model(destination):
    destination = Path(destination)
    if destination.exists():
        verify_model(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with requests.get(MODEL_URL, stream=True, timeout=(10, 60), allow_redirects=False) as response:
            if response.status_code != 200:
                raise RuntimeError('Could not download the official hand-landmarker model.')
            count = 0
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.part', delete=False) as output:
                temporary = Path(output.name)
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    count += len(chunk)
                    if count > MAX_MODEL_BYTES:
                        raise ValueError('Hand-landmarker model exceeds the size limit.')
                    output.write(chunk)
        verify_model(temporary)
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
