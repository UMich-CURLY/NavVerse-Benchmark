import numpy as np
import pytest
from PIL import Image, ImageDraw

from baselines.utils.modern_video import (
    ModernVideoRenderer,
    _compact_action,
    _fit_instruction,
    capture_modern_video_sample,
)


@pytest.mark.parametrize("with_third,with_bev", [(False, False), (True, False), (True, True)])
def test_compact_renderer_uses_only_available_views(with_third, with_bev):
    pov = np.full((120, 160, 3), [20, 90, 180], dtype=np.uint8)
    third = np.full((120, 160, 3), [180, 90, 20], dtype=np.uint8)
    bev = np.full((180, 180, 3), [40, 120, 70], dtype=np.uint8)
    sample = capture_modern_video_sample(pov, third if with_third else None, 7, "TURN_LEFT_SMALL")
    renderer = ModernVideoRenderer(
        "Exit the room, cross the square, keep the ambulance on your left, and stop by the camera bag.",
        "innout_amsterdam_13",
        "InternVLA-N1",
    )

    frame = renderer.render(sample, bev if with_bev else None)

    height, width, _ = frame.shape
    assert height % 8 == width % 8 == 0
    assert frame.dtype == np.uint8
    expected = {"pov": pov}
    if with_third:
        expected["third"] = third
    if with_bev:
        expected["bev"] = bev
    assert renderer.view_boxes.keys() == expected.keys()
    image_area = 0
    for name, (left, top, right, bottom) in renderer.view_boxes.items():
        image_area += (right - left) * (bottom - top)
        assert np.allclose(frame[(top + bottom) // 2, (left + right) // 2], expected[name][0, 0], atol=3)
    assert image_area / (height * width) > 0.75


def test_missing_frame_does_not_resize_an_existing_video():
    pov = np.zeros((120, 160, 3), dtype=np.uint8)
    renderer = ModernVideoRenderer("Find the door", "scene_1", "Oracle")
    first = renderer.render(capture_modern_video_sample(pov, pov, 1, "FORWARD"))
    missing = renderer.render(capture_modern_video_sample(pov, None, 2, "STOP"))
    assert first.shape == missing.shape


def test_long_instruction_is_wrapped_without_dropping_words():
    instruction = " ".join(f"waypoint{index}" for index in range(80))
    canvas = Image.new("RGB", (1200, 300))
    draw = ImageDraw.Draw(canvas)

    _, lines, _ = _fit_instruction(draw, instruction, 1100, 220)

    assert " ".join(lines).split() == instruction.split()


def test_waypoint_action_is_compact_for_fixed_status_badge():
    action = "32 waypoints, next waypoint: [210.1, 160.9, -0.2]"

    assert _compact_action(action) == "32 WAYPOINTS"
    assert _compact_action("TURN_RIGHT_SMALL") == "RIGHT · 15°"
