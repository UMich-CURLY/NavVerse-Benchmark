import pytest

from baselines.gpt.policy import ACTION_SCHEMA, parse_action_response


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('{"action":"FORWARD","step_size":1.25}', ("FORWARD", 1.25)),
        ('{"action":"TURN_LEFT","step_size":90}', ("TURN_LEFT", 90.0)),
        ('{"action":"TURN_RIGHT","step_size":45}', ("TURN_RIGHT", 45.0)),
        ('{"action":"STOP","step_size":0}', ("STOP", 0.0)),
    ],
)
def test_parse_action_response_uses_one_action_sized_parameter(payload, expected):
    assert parse_action_response(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        '{"action":"FORWARD","step_size":0}',
        '{"action":"FORWARD","step_size":5.1}',
        '{"action":"TURN_LEFT","step_size":4}',
        '{"action":"TURN_RIGHT","step_size":181}',
        '{"action":"STOP","step_size":1}',
        '{"action":"STOP","step_size":true}',
    ],
)
def test_parse_action_response_rejects_invalid_step_sizes(payload):
    with pytest.raises(ValueError):
        parse_action_response(payload)


def test_action_schema_requires_action_and_step_size_only():
    assert ACTION_SCHEMA["required"] == ["action", "step_size"]
    assert set(ACTION_SCHEMA["properties"]) == {"action", "step_size"}
