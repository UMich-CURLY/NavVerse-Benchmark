import pytest

from navverse.episode import VLNEpisode, episode_instruction, episode_target_type


def test_mixed_targets_select_the_same_instruction_type():
    episodes = [
        VLNEpisode({
            "scene_id": "indoor",
            "episode_id": 1,
            "target_type": "obj",
            "instructions": [
                {"instruction_type": "target", "instruction_content": "lamp"},
                {"instruction_type": "vln", "instruction_content": "Walk down the hall to the lamp."},
            ],
        }),
        VLNEpisode({
            "scene_id": "outdoor",
            "episode_id": 2,
            "target_type": "place",
            "instructions": [
                {"instruction_type": "target", "instruction_content": "bus stop"},
            ],
        }),
    ]

    assert [episode_target_type(ep) for ep in episodes] == ["obj", "place"]
    assert [episode_instruction(ep, "target") for ep in episodes] == ["lamp", "bus stop"]
    assert episode_instruction(episodes[0], "vln") == "Walk down the hall to the lamp."
    with pytest.raises(ValueError, match="needs exactly one nonempty vln instruction"):
        episode_instruction(episodes[1], "vln")
