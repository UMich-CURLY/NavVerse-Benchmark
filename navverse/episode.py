import json
import os
import re
from copy import deepcopy


# Trailing _<digits> on an objnav class label is the sub-class id we strip to
# get the broad shop class (e.g. "restaurant_7" -> "restaurant", "fast_food_5"
# -> "fast_food", "convenience_store_4" -> "convenience_store").
_SHOP_CLASS_TRAILING_INSTANCE_RE = re.compile(r"_\d+$")
_DEFAULT_EPISODE_SUBFOLDERS = ("grCommercial", "vcp", "innout")
_LEGACY_DEFAULT_FILENAMES = {"test_generator.json"}


def _broad_shop_class(objnav: str) -> str:
    return _SHOP_CLASS_TRAILING_INSTANCE_RE.sub("", objnav or "")


def _is_store_scene(scene_id: str) -> bool:
    return "store" in (scene_id or "")


def episode_target_type(episode):
    target_type = episode.get("target_type")
    if target_type is not None and target_type not in {"obj", "place"}:
        raise ValueError(f"Invalid target_type {target_type!r} in {episode['episode_label']}")
    if target_type in {"obj", "place"}:
        return target_type
    goals = episode.get("goals", [])
    index = episode.get("closest_goal_idx", 0)
    goal_type = goals[index].get("type") if 0 <= index < len(goals) else None
    return "place" if goal_type == "place" or episode.get("task_type") == "placenav" else "obj"


def episode_instruction(episode, instruction_type):
    if instruction_type not in {"target", "vln"}:
        raise ValueError(f"Invalid instruction_type {instruction_type!r}")
    instructions = episode.get("instructions")
    if instructions is not None:
        matches = [item["instruction_content"] for item in instructions
                   if item.get("instruction_type") == instruction_type]
        if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0].strip():
            raise ValueError(f"Episode {episode['episode_label']} needs exactly one nonempty {instruction_type} instruction")
        return matches[0]
    if instruction_type == "vln":
        content = episode.get("instruction")
    elif episode_target_type(episode) == "place":
        content = episode.get("placenav_goal") or episode.get("instruction")
    else:
        content = episode.get("objnav")
        if _is_store_scene(episode.get("scene_id", "")):
            content = _broad_shop_class(content)
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Episode {episode['episode_label']} has no {instruction_type} instruction")
    return content

class VLNEpisode(dict):
    def __init__(self, data=None, **kwargs):
        """ Pass in data or kwargs to update the default episode data """
        super().__init__()
        self.update({
            "scene_id": "test",
            "scene_type": "test",
            "episode_id": 0,
            "path": "test.usd",
            "scene_scale": 1.0,
            "collider": True,
            "align_ground": True,
            "navmesh_preset": "default",
            "objnav": "table",
            "instruction": "",
            "closest_goal_idx": 0,
            "goals": [
                {
                    "instance": "table",
                    "type": "object",
                    "location": [-50.37,-19.45, 0.6],
                    "radius": 0.5,
                    "reference_path": [
                        [-52.37,-19.45, 0.6],
                        [-52.37,-3.48, 0.6],
                        [-60.77,-3.48, 0.6]
                    ]
                }
            ],
            "start_position": [0.0, 0.0, 0.6],
            "start_rotation": [1.0, 0.0, 0.0, 0.0],
        })
        if data is not None:
            self.update(deepcopy(data))
        if kwargs:
            self.update(deepcopy(kwargs))
            # print(f'Overwriting episode data with kwargs: {kwargs}')
    def copy(self):
        return deepcopy(self)
    
    @property
    def episode_label(self):
        """ episode label = scene_id + episode_id """
        return f"{self['scene_id']}_{self['episode_id']}"

    @property
    def scene_id(self):
        return self['scene_id']

    @property
    def episode_id(self):
        return self['episode_id']

    @property
    def episode_info(self):
        return json.dumps(self, indent=4)

    def __getitem__(self, key):
        if key == "episode_label":
            return self.episode_label
        if key == "episode_info":
            return self.episode_info
        return super().__getitem__(key)

    @classmethod
    def from_json(self, json_path):
        data = json.load(open(json_path))
        if not isinstance(data, list):
            return []
        episodes = [VLNEpisode(x) for x in data]
        return episodes

    @classmethod
    def from_json_folder(self, json_folder):
        json_paths = episode_json_paths(json_folder)
        episodes = []
        for json_path in json_paths:
            try:
                episodes.extend(self.from_json(json_path))
            except Exception as e:
                print(f"Error loading episode from {json_path}: {e}")
                continue
        episodes.sort(key=lambda x: (x.scene_id, x.episode_id))
        _expand_store_class_goals(episodes)
        return episodes


def episode_json_paths(json_folder):
    json_folder = os.path.abspath(json_folder)
    json_paths = [
        os.path.join(json_folder, name)
        for name in os.listdir(json_folder)
        if name.endswith(".json") and os.path.isfile(os.path.join(json_folder, name))
    ]
    if os.path.basename(json_folder) == "episodes":
        for subfolder in _DEFAULT_EPISODE_SUBFOLDERS:
            subfolder_path = os.path.join(json_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            json_paths.extend(
                os.path.join(subfolder_path, name)
                for name in os.listdir(subfolder_path)
                if name.endswith(".json") and os.path.isfile(os.path.join(subfolder_path, name))
            )
        legacy_path = os.path.join(json_folder, "legacy")
        if os.path.isdir(legacy_path):
            json_paths.extend(
                os.path.join(legacy_path, name)
                for name in os.listdir(legacy_path)
                if (name.startswith("vc_") or name in _LEGACY_DEFAULT_FILENAMES)
                and name.endswith(".json")
                and os.path.isfile(os.path.join(legacy_path, name))
            )
    return sorted(json_paths)


def find_episode_json(json_folder, scene_id):
    filename = f"{scene_id}.json"
    matches = [path for path in episode_json_paths(json_folder) if os.path.basename(path) == filename]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple episode files found for scene '{scene_id}': {matches}")
    return matches[0]

def load_episode_set(json_folder):
    #detect every file that ends with _set.txt
    set_files = [x for x in os.listdir(json_folder) if x.endswith("_set.txt")]
    episode_set_list = {}
    for set_file in set_files:
        with open(os.path.join(json_folder, set_file), 'r') as f:
            lines = [x.strip() for x in f.read().splitlines()]
            lines = [x for x in lines if not x.startswith("#")]
            episode_set_list[set_file.replace("_set.txt", "")] = lines
    return episode_set_list

def save_episodes(episodes, json_path):
    json.dump(episodes, open(json_path, 'w'), indent=4)


def _expand_store_class_goals(episodes):
    """For store scenes, broaden each episode's goals to include every instance
    sharing the same broad shop class across sibling episodes in that scene.

    Why: store-scene objnav labels carry both a class and a sub-class id (e.g.
    "restaurant_7"), but the instruction sent to the agent is stripped to the
    broad class ("restaurant") by navverse/sim.py. With only the broad-class
    cue, reaching any "restaurant_*" instance is a fair success. DistanceToGoal
    already returns the min over goals[], so appending sibling instances makes
    the metric match the prompt's resolution.
    """
    scene_to_episodes = {}
    for ep in episodes:
        if not _is_store_scene(ep.get("scene_id", "")):
            continue
        scene_to_episodes.setdefault(ep["scene_id"], []).append(ep)

    for scene_id, scene_episodes in scene_to_episodes.items():
        class_to_goals = {}
        for ep in scene_episodes:
            broad = _broad_shop_class(ep.get("objnav", ""))
            if not broad:
                continue
            bucket = class_to_goals.setdefault(broad, {})
            for goal in ep.get("goals", []):
                instance = goal.get("instance", "")
                if instance and instance not in bucket:
                    bucket[instance] = deepcopy(goal)

        for ep in scene_episodes:
            broad = _broad_shop_class(ep.get("objnav", ""))
            if broad not in class_to_goals:
                continue
            original_goals = ep.get("goals", [])
            original_instances = {g.get("instance") for g in original_goals}
            sibling_goals = [
                goal
                for inst, goal in class_to_goals[broad].items()
                if inst not in original_instances
            ]
            if sibling_goals:
                ep["goals"] = list(original_goals) + sibling_goals
