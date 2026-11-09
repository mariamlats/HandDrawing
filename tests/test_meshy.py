import json
import struct
from pathlib import Path
from unittest.mock import Mock
import pytest
import meshy_client as client
import model_asset


class Response:
    def __init__(self, data=b'', status=200, headers=None):
        self.content = json.dumps(data).encode() if isinstance(data, dict) else data
        self.status_code = status
        self.headers = headers or {}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def iter_content(self, chunk_size=65536):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    def blocked(*a, **k): raise AssertionError('Unexpected external request')
    monkeypatch.setattr(client.requests, 'get', blocked)
    monkeypatch.setattr(client.requests, 'post', blocked)


def glb():
    return b'glTF' + struct.pack('<II', 2, 24) + struct.pack('<I4s', 4, b'JSON') + b'{}  '


@pytest.mark.parametrize('url', ['http://assets.meshy.ai/model.glb',
    'https://127.0.0.1/model.glb', 'https://assets.meshy.ai.evil.invalid/model.glb',
    'https://user:password@assets.meshy.ai/model.glb', 'file:///etc/passwd',
    'https://assets.meshy.ai:8443/model.glb'])
def test_unapproved_model_urls_are_rejected(url):
    with pytest.raises(client.MeshyError): client.validate_asset_url(url)


def test_streamed_model_is_verified_and_written_atomically(tmp_path, monkeypatch):
    get = Mock(return_value=Response(glb()))
    monkeypatch.setattr(client.requests, 'get', get)
    result = client.download_glb('https://assets.meshy.ai/model.glb?token=example',
                                 tmp_path / 'model.glb', tmp_path)
    assert Path(result).read_bytes() == glb()
    assert get.call_args.kwargs['stream'] and not get.call_args.kwargs['allow_redirects']
    assert 'headers' not in get.call_args.kwargs
    assert not list(tmp_path.glob('*.part'))


@pytest.mark.parametrize('response', [Response(b'not-a-model'), Response(glb(), status=302),
    Response(glb(), headers={'Content-Length': str(client.MAX_MODEL_BYTES + 1)})])
def test_invalid_download_does_not_leave_partial_output(tmp_path, monkeypatch, response):
    monkeypatch.setattr(client.requests, 'get', lambda *a, **k: response)
    with pytest.raises(client.MeshyError):
        client.download_glb('https://assets.meshy.ai/model.glb', tmp_path/'model.glb', tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_stream_limit_is_enforced_without_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr(client.requests, 'get', lambda *a, **k: Response(glb() + b'excess'))
    with pytest.raises(client.MeshyError):
        client.download_glb('https://assets.meshy.ai/model.glb', tmp_path/'model.glb', tmp_path, max_bytes=24)
    assert list(tmp_path.iterdir()) == []


def test_output_cannot_escape_output_directory(tmp_path):
    with pytest.raises(client.MeshyError):
        client.download_glb('https://assets.meshy.ai/model.glb', tmp_path/'../model.glb', tmp_path)


def test_api_requests_reject_redirects_and_encode_opaque_task_ids(monkeypatch):
    post = Mock(return_value=Response({'result': '../opaque/id?query'}))
    get = Mock(return_value=Response({'status':'SUCCEEDED','model_urls':{'glb':'https://assets.meshy.ai/model.glb'}}))
    monkeypatch.setattr(client.requests, 'post', post)
    monkeypatch.setattr(client.requests, 'get', get)
    api = client.MeshyClient('test-key')
    identifier = api.create_task('data:image/png;base64,example')
    api.wait_for_task(identifier)
    assert '%2F' in get.call_args.args[0] and '%3F' in get.call_args.args[0]
    assert post.call_args.kwargs['allow_redirects'] is False
    assert get.call_args.kwargs['allow_redirects'] is False


def test_api_errors_do_not_repeat_private_response_bodies(monkeypatch):
    monkeypatch.setattr(client.requests, 'post', lambda *a, **k: Response(b'private-key-in-body', status=400))
    with pytest.raises(client.MeshyError) as error:
        client.MeshyClient('private-key').create_task('data:')
    assert 'private-key' not in str(error.value)


def test_polling_has_deadline(monkeypatch):
    get = Mock(return_value=Response({'status':'IN_PROGRESS','progress':12}))
    monkeypatch.setattr(client.requests, 'get', get)
    counter = iter(range(100))
    monkeypatch.setattr(client.time, 'monotonic', lambda: next(counter))
    monkeypatch.setattr(client.time, 'sleep', lambda value: None)
    with pytest.raises(client.MeshyError, match='Stopped waiting'):
        client.MeshyClient('test-key').wait_for_task('task', timeout=3)
    assert get.call_count <= 2


def test_json_response_size_is_bounded(monkeypatch):
    monkeypatch.setattr(client.requests, 'post', lambda *a, **k: Response(b'x' * (client.MAX_JSON_BYTES + 1)))
    with pytest.raises(client.MeshyError, match='size limit'):
        client.MeshyClient('test-key').create_task('data:')


def test_model_checksum_rejects_modified_weights(tmp_path):
    file = tmp_path / 'model.task'
    file.write_bytes(b'changed-model')
    with pytest.raises(ValueError, match='checksum'): model_asset.verify_model(file)
