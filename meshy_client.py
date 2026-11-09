"""Optional Meshy client with bounded polling and downloads; no GUI dependencies."""

import os
import json
from pathlib import Path
import struct
import tempfile
import time
from urllib.parse import quote, urlsplit

import requests


API_URL = 'https://api.meshy.ai/openapi/v1/image-to-3d'
MAX_MODEL_BYTES = 100 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024


class MeshyError(RuntimeError):
    pass


def _json_response(response):
    # Never include response bodies, bearer keys or signed asset URLs in errors.
    if not 200 <= response.status_code < 300:
        raise MeshyError(f'Meshy request failed (HTTP {response.status_code}).')
    chunks, size = [], 0
    for chunk in response.iter_content(chunk_size=16 * 1024):
        size += len(chunk)
        if size > MAX_JSON_BYTES:
            raise MeshyError('Meshy response exceeds the size limit.')
        chunks.append(chunk)
    try:
        data = json.loads(b''.join(chunks))
    except (ValueError, UnicodeDecodeError) as exc:
        raise MeshyError('Meshy returned invalid JSON.') from exc
    if not isinstance(data, dict):
        raise MeshyError('Meshy returned an unexpected response.')
    return data


def _task_id(value):
    # IDs are opaque; percent-encode them for URL paths and never use them as filenames.
    if (not isinstance(value, str) or not value or len(value) > 512
            or value in ('.', '..') or any(ord(char) < 32 for char in value)):
        raise MeshyError('Meshy returned an invalid task identifier.')
    return value


class MeshyClient:
    def __init__(self, api_key):
        if not api_key:
            raise MeshyError('Configure MESHY_API_KEY before enabling 3D generation.')
        self.api_key = api_key

    def create_task(self, png_data_uri):
        try:
            with requests.post(API_URL, headers={'Authorization': f'Bearer {self.api_key}'},
                               json={'image_url': png_data_uri, 'enable_pbr': True,
                                     'should_remesh': True, 'should_texture': True,
                                     'target_formats': ['glb']},
                               timeout=(10, 90), stream=True, allow_redirects=False) as response:
                return _task_id(_json_response(response).get('result'))
        except requests.RequestException as exc:
            raise MeshyError('Could not create the Meshy task. Check connectivity/account status before retrying.') from exc

    def wait_for_task(self, task_id, on_progress=None, timeout=1200):
        if not 1 <= timeout <= 3600:
            raise ValueError('Polling timeout must be between 1 and 3600 seconds.')
        url = API_URL + '/' + quote(_task_id(task_id), safe='')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                remaining = max(0.1, deadline - time.monotonic())
                with requests.get(url, headers={'Authorization': f'Bearer {self.api_key}'},
                                  timeout=(min(10, remaining), min(30, remaining)),
                                  stream=True, allow_redirects=False) as response:
                    data = _json_response(response)
            except requests.RequestException as exc:
                raise MeshyError('Could not retrieve Meshy task status. No automatic resubmission was made.') from exc
            status = data.get('status')
            if not isinstance(status, str):
                raise MeshyError('Meshy returned an invalid task status.')
            try:
                progress = max(0, min(100, int(data.get('progress', 0) or 0)))
            except (TypeError, ValueError):
                progress = 0
            if on_progress:
                on_progress(status, progress)
            if status == 'SUCCEEDED':
                models = data.get('model_urls')
                if not isinstance(models, dict) or not isinstance(models.get('glb'), str):
                    raise MeshyError('The completed task did not include a GLB model.')
                return models['glb']
            if status in ('FAILED', 'CANCELED', 'CANCELLED'):
                raise MeshyError('The Meshy task did not complete successfully. Check its provider dashboard.')
            time.sleep(min(2, max(0, deadline - time.monotonic())))
        raise MeshyError('Stopped waiting for Meshy. The remote task may still be running; check its dashboard before submitting again.')


def validate_asset_url(url):
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == 'https' and parsed.hostname == 'assets.meshy.ai'
                 and parsed.port in (None, 443) and not parsed.username
                 and not parsed.password and not parsed.fragment)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise MeshyError('Model downloads must use the approved HTTPS Meshy asset host.')
    return url


def download_glb(url, destination, output_directory, max_bytes=MAX_MODEL_BYTES):
    validate_asset_url(url)
    if not 12 <= max_bytes <= MAX_MODEL_BYTES:
        raise ValueError('Invalid model size limit.')
    root = Path(output_directory).resolve()
    target = Path(destination).resolve()
    if target.parent != root or target.suffix.lower() != '.glb':
        raise MeshyError('Models must be saved as GLB files directly inside the output folder.')
    root.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # No bearer credentials are sent to the asset server. Redirects are rejected.
        with requests.get(url, stream=True, timeout=(10, 30), allow_redirects=False) as response:
            if response.status_code != 200:
                raise MeshyError('Model download failed.')
            try:
                declared_length = int(response.headers.get('Content-Length', '0'))
            except (TypeError, ValueError) as exc:
                raise MeshyError('Invalid model response length.') from exc
            if declared_length < 0 or declared_length > max_bytes:
                raise MeshyError('Model exceeds the download size limit.')
            count = 0
            deadline = time.monotonic() + 120
            with tempfile.NamedTemporaryFile(dir=root, suffix='.part', delete=False) as output:
                temporary = Path(output.name)
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if time.monotonic() > deadline:
                        raise MeshyError('Model download timed out.')
                    count += len(chunk)
                    if count > max_bytes:
                        raise MeshyError('Model exceeds the download size limit.')
                    output.write(chunk)
        with temporary.open('rb') as model:
            header = model.read(12)
        if len(header) != 12 or header[:4] != b'glTF':
            raise MeshyError('Downloaded content is not a GLB model.')
        version, length = struct.unpack('<II', header[4:])
        if version != 2 or length != count or count < 20:
            raise MeshyError('Downloaded GLB header or length is invalid.')
        os.replace(temporary, target)
        temporary = None
        return str(target)
    except requests.RequestException as exc:
        raise MeshyError('Model download failed. Check the task in your Meshy dashboard.') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
