from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import GUI_hand_drawing as drawing


@pytest.fixture(autouse=True)
def reset_state(tmp_path, monkeypatch):
    drawing.clear_canvas()
    drawing.eraser_mask = None
    drawing.meshy_job.update(running=False, error='')
    monkeypatch.setattr(drawing, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(drawing, 'MESHY_ENABLED', False)
    monkeypatch.setattr(drawing, 'MESHY_AUTO_OPEN', False)
    monkeypatch.setattr(drawing, 'MESHY_API_KEY', '')


def test_png_exports_have_unique_names_and_valid_content():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    first, second = drawing.save_png(image), drawing.save_png(image)
    assert first != second
    assert drawing.cv2.imread(first).shape == image.shape


def test_empty_and_single_pixel_svg_exports(tmp_path):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    path = tmp_path / 'empty.svg'
    drawing.vectorize_to_svg(image, str(path))
    assert ET.parse(path).getroot().tag.endswith('svg')
    image[10, 10] = [0, 255, 0]
    quantized, palette = drawing._kmeans_quantize(image)
    assert len(palette) == 1
    drawing.vectorize_to_svg(image, str(path))
    ET.parse(path)


def test_coloured_shape_exports_svg_paths(tmp_path):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[10:40, 10:40] = [0, 255, 0]
    path = tmp_path / 'shape.svg'
    drawing.vectorize_to_svg(image, str(path))
    assert ET.parse(path).findall('{http://www.w3.org/2000/svg}path')


def test_rdp_handles_long_jagged_strokes_without_recursion():
    points = [(x, x % 2) for x in range(1200)]
    simplified = drawing._rdp(points, 0.01)
    assert simplified[0] == points[0] and simplified[-1] == points[-1]
    assert len(simplified) > 1000


def test_straightening_preserves_end_to_end_line():
    points = [(x, 50) for x in range(0, 200, 2)]
    result = drawing._auto_straighten(points, 4)
    assert len(result) == 2
    assert all(abs(y - 50) <= 1 for _, y in result)


def test_clear_removes_in_progress_stroke_and_mask():
    drawing.eraser_mask = np.ones((32, 32), dtype=np.uint8) * 255
    drawing.start_stroke((1, 1))
    drawing.append_stroke_point((10, 10))
    drawing.check_ui_interaction(20, 220, True, None)
    drawing.finish_stroke()
    assert drawing._current_stroke is None and drawing.strokes == []
    assert not drawing.eraser_mask.any()


def test_eraser_mask_can_be_repainted():
    drawing.eraser_mask = np.ones((32, 32), dtype=np.uint8) * 255
    drawing.unerase_on_draw(None, (16, 16), 6)
    assert drawing.eraser_mask[16, 16] == 0
    assert drawing.eraser_mask[0, 0] == 255


def test_disabled_meshy_does_not_save_or_submit(monkeypatch):
    save = Mock()
    monkeypatch.setattr(drawing, 'save_png_and_svg', save)
    assert not drawing.start_meshy_3d_from_canvas(np.zeros((8, 8, 3), dtype=np.uint8))
    save.assert_not_called()


def test_job_is_reserved_before_thread_start(monkeypatch):
    monkeypatch.setattr(drawing, 'MESHY_ENABLED', True)
    monkeypatch.setattr(drawing, 'MESHY_API_KEY', 'test-only-key')
    monkeypatch.setattr(drawing, 'save_png_and_svg', lambda image: ('drawing.png', 'drawing.svg'))
    thread = Mock()
    monkeypatch.setattr(drawing.threading, 'Thread', thread)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert drawing.start_meshy_3d_from_canvas(image)
    assert not drawing.start_meshy_3d_from_canvas(image)
    assert thread.call_count == 1


def test_downloaded_models_are_not_auto_opened(monkeypatch):
    open_process = Mock()
    monkeypatch.setattr(drawing.subprocess, 'run', open_process)
    drawing._open_file_or_url('anything.glb')
    open_process.assert_not_called()


def test_failed_camera_is_released(tmp_path, monkeypatch):
    model = tmp_path / 'hand.task'
    model.write_bytes(b'test-placeholder-not-loaded')
    monkeypatch.setattr(drawing, 'MODEL_PATH', model)
    camera = Mock()
    camera.read.return_value = (False, None)
    monkeypatch.setattr(drawing.cv2, 'VideoCapture', lambda *a: camera)
    close = Mock()
    monkeypatch.setattr(drawing.cv2, 'destroyAllWindows', close)
    drawing.main()
    camera.release.assert_called_once()
    close.assert_called_once()
