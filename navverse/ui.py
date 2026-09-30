import asyncio
from pathlib import Path
import random
import time
import json
import os
import numpy as np
import torch

import omni.kit.app
import omni.ui as ui
from isaacsim.gui.components import ui_utils
from isaacsim.gui.components.style import get_style
import isaacsim.core.utils.prims as prim_utils
import omni.usd
import isaacsim.core.utils.bounds as bounds_utils
import isaaclab.sim as sim_utils
import navverse.navmesh as navmesh_utils
from navverse.episode import VLNEpisode
from navverse.utils.vis import visualize_points, visualize_curve
from scipy.spatial.transform import Rotation as R

# check ui_utils.py from https://github.com/isaac-sim/IsaacSim/blob/main/source/extensions/isaacsim.gui.components/isaacsim/gui/components/ui_utils.py


LABEL_WIDTH = 120
SPACING = 4

str_func = [lambda x: x.as_string, lambda x, y: x.set_value(y)]
int_func = [lambda x: x.as_int, lambda x, y: x.set_value(y)]
float_func = [lambda x: x.as_float, lambda x, y: x.set_value(y)]
bool_func = [lambda x: x.as_bool, lambda x, y: x.set_value(y)]
choice_func = lambda item_list: [
    lambda x: item_list[x.get_item_value_model().as_int],
    lambda x, y: x.get_item_value_model().set_value(item_list.index(y))
]
xyz_func = [lambda x: [x[i].as_float for i in range(3)], lambda x, y: [x[i].set_value(y[i]) for i in range(3)]]

def set_text(x, y):
    x.text = y
label_func = [lambda x: x.text, lambda x, y: set_text(x, y)]

class BaseUI:

    def __init__(self, vln_sim, window_name = "NavVerse Settings"):
        self.env = vln_sim.manager_env

        # create window for UI
        dock_preference = getattr(ui.DockPreference, "RIGHT_BOTTOM", ui.DockPreference.RIGHT_TOP)
        self.self = ui.Window(
            window_name, width=300, height=500, visible=True, dock_preference=dock_preference
        )
        # dock once Kit has registered the UI windows
        asyncio.ensure_future(self._dock_window(window_title=self.self.title))

        # keep a dictionary of stacks so that child environments can add their own UI elements
        # this can be done by using the `with` context manager
        self.ui_elements = dict()
        # create main frame
        self.ui_elements["main_frame"] = self.self.frame
        with self.ui_elements["main_frame"]:
            self.ui_elements["main_stack"] = ui.VStack(style=get_style(), spacing=5, height=0)
    

    def get_ui_value(self, map, key):
        try:
            getter_func, setter_func = map[key][1]
            # print(f"[INFO]: Get {key} value: {getter_func(self.ui_elements[key])}")
            return getter_func(self.ui_elements[key])
        except:
            print(f"[ERROR]: Failed to get value for {key}")
            import traceback
            traceback.print_exc()

    def set_ui_value(self, map, key, value):
        try:
            getter_func,  setter_func = map[key][1]
            setter_func(self.ui_elements[key], value)
            # print(f"[INFO]: Set {key} value: {value}")
        except:
            print(f"[ERROR]: Failed to set value for {key}")
            import traceback
            traceback.print_exc()

    def create_frame(self, name, collapsed=False):
        with ui.CollapsableFrame(name, collapsed=collapsed):
            return ui.VStack(style=get_style(), spacing=5, height=0)
            

    def __del__(self):
        """Destructor for the window."""
        # destroy the window
        if self.self is not None:
            self.self.visible = False
            self.self.destroy()
            self.self = None

    async def _dock_window(self, window_title):
        """Docks the custom UI window to the property window."""
        for _ in range(10):
            if ui.Workspace.get_window(window_title) and ui.Workspace.get_window("Property"):
                break
            await self.env.sim.app.next_update_async()

        custom_window = ui.Workspace.get_window(window_title)
        property_window = ui.Workspace.get_window("Property")
        if custom_window and property_window:
            custom_window.dock_in(property_window, ui.DockPosition.SAME, 1.0)
            custom_window.focus()

class BenchmarkUI(BaseUI):
    def __init__(self, vln_sim):
        super().__init__(vln_sim)
        # set ref to vln_sim
        self.vln_sim = vln_sim
        self.episode_label_list = vln_sim.episode_label_list
        self.episode_list = vln_sim.episode_list
        self.manager_env = vln_sim.manager_env
        self.env = vln_sim.env
        self.ui_episode = None
        self._last_runtime_info_update_s = 0.0
        # initialize ui
        self.build_ui()
        # map ui_episode fields to ui_elements
        # format: {ui_name: (config_name, [getter_func, setter_func])}
        instruction_name = "objnav" if self.vln_sim.args.task_type=="objnav" else "instruction"
        self.ui_map = {
            "episode_label": ("episode_label", choice_func(self.episode_label_list)),
            "instruction": (instruction_name, str_func),
            "episode_info": ("episode_info", label_func),
            "start_position": ("start_position", xyz_func),
        }
        # add callback to sim
        self.vln_sim.add_callback('client_episode_changed', lambda episode_label: self.update_ui("episode_label", episode_label))
        self.vln_sim.add_callback('reset_finished', lambda: self.refresh_runtime_info(force=True))
        self.vln_sim.add_callback('step_finished', self.refresh_runtime_info)

    def build_ui(self):
        with self.ui_elements["main_stack"]:
            with self.create_frame("Debug Tools"):
                self.ui_elements["episode_label"] = ui_utils.dropdown_builder(
                    "Episode Id",
                    items=self.episode_label_list,
                    on_clicked_fn=lambda x: self.update_ui("episode_label", x)
                )
                self.ui_elements["instruction"] = ui_utils.str_builder(
                    "Instruction",
                    default_val="instruction",
                )
                self.ui_elements["start_position"] = ui_utils.xyz_builder(
                    "Start Position",
                    default_val=[0.0, 0.0, 0.0]
                )
                ui_utils.btn_builder("Load Scene", text="Load", on_clicked_fn=lambda: self.save_settings("episode_runtime"))
                ui_utils.btn_builder("Follow Reference Path", text="Start", on_clicked_fn=self.follow_reference_path)
                ui_utils.btn_builder("Stop Following", text="Stop", on_clicked_fn=lambda: self.vln_sim.clear_waypoints())
                ui_utils.btn_builder("Switch StopCalled State", text="Switch", on_clicked_fn=lambda: self.env.set_stop_called(self.vln_sim.robot_index, not self.env.is_stop_called[self.vln_sim.robot_index]))
            with self.create_frame("Action Controls"):
                with ui.HStack(spacing=SPACING, height=28):
                    ui.Button("Forward", clicked_fn=lambda: self.run_discrete_action("FORWARD"))
                    ui.Button("Back", clicked_fn=lambda: self.run_discrete_action("BACKWARD"))
                    ui.Button("Stop", clicked_fn=lambda: self.run_discrete_action("STOP"))
                with ui.HStack(spacing=SPACING, height=28):
                    ui.Button("Left", clicked_fn=lambda: self.run_discrete_action("LEFT"))
                    ui.Button("Right", clicked_fn=lambda: self.run_discrete_action("RIGHT"))
                    ui.Button("Turn L", clicked_fn=lambda: self.run_discrete_action("TURN_LEFT"))
                    ui.Button("Turn R", clicked_fn=lambda: self.run_discrete_action("TURN_RIGHT"))
            with self.create_frame("Metrics Info"):
                self.ui_elements["metrics_info"] = ui_utils.ui.Label(
                    "No metrics yet",
                    style_type_name_override="Label::label",
                    word_wrap=True,
                    alignment=ui_utils.ui.Alignment.LEFT_TOP,
                )
            with self.create_frame("Sim Info"):
                self.ui_elements["sim_info"] = ui_utils.ui.Label(
                    "No sim info yet",
                    style_type_name_override="Label::label",
                    word_wrap=True,
                    alignment=ui_utils.ui.Alignment.LEFT_TOP,
                )
            with self.create_frame("Episode Info"):
                self.ui_elements["episode_info"] = ui_utils.ui.Label(
                    "Episode Info",
                    style_type_name_override="Label::label",
                    word_wrap=True,
                    alignment=ui_utils.ui.Alignment.LEFT_TOP,
                )

    def update_ui(self, settings_type, selected_value=None):
        if settings_type == "episode_label":
            if self.ui_episode is not None and selected_value==self.ui_episode["episode_label"]:
                return
            print(f"[UI] UI is updating to episode {selected_value}")
            self.ui_episode = self.vln_sim.episode_list[self.vln_sim.episode_label_list.index(selected_value)]
            for key, (value, _) in self.ui_map.items():
                self.set_ui_value(self.ui_map, key, self.ui_episode[value])
        # elif settings_type == "episode_info":
        #     self.set_ui_value(self.ui_map, "episode_info", self.ui_episode.episode_info)


    def follow_reference_path(self):
        try:
            episode = self.vln_sim.current_episode or self.ui_episode
            if episode is None:
                print("[UI] Cannot follow reference path: no episode is loaded")
                return
            if hasattr(self.vln_sim.env, "set_stop_called"):
                self.vln_sim.env.set_stop_called(self.vln_sim.robot_index, False)
            self.vln_sim.set_ref_waypoints(episode)
            self.refresh_runtime_info(force=True)
        except Exception as exc:
            print(f"[UI] Failed to follow reference path: {exc}")
            import traceback
            traceback.print_exc()

    def run_discrete_action(self, action):
        try:
            self.vln_sim.set_discrete_waypoint_action(action)
            self.refresh_runtime_info(force=True)
        except Exception as exc:
            print(f"[UI] Failed to run discrete action {action}: {exc}")
            import traceback
            traceback.print_exc()

    def refresh_runtime_info(self, force=False):
        now = time.time()
        if not force and now - self._last_runtime_info_update_s < 0.25:
            return
        self._last_runtime_info_update_s = now
        try:
            state = self.vln_sim._state_payload()
        except Exception as exc:
            self.ui_elements["metrics_info"].text = f"Failed to read metrics: {exc}"
            self.ui_elements["sim_info"].text = f"Failed to read sim info: {exc}"
            return

        metrics = state.get("metrics")
        if not metrics and self.vln_sim.info is not None:
            metrics = self.vln_sim.info.get("measurements")
        self.ui_elements["metrics_info"].text = self._format_metrics(metrics or {})
        self.ui_elements["sim_info"].text = self._format_sim_info(state)

    def _format_metrics(self, metrics):
        if not metrics:
            return "No metrics yet"
        preferred_keys = [
            "sim_duration",
            "distance_to_goal",
            "progress",
            "path_length",
            "success",
            "spl",
            "soft_success",
            "soft_spl",
            "oracle_success",
        ]
        keys = [key for key in preferred_keys if key in metrics]
        keys.extend(sorted(key for key in metrics if key not in keys))
        return "\n".join(f"{key}: {self._format_value(metrics[key])}" for key in keys)

    def _format_sim_info(self, state):
        return "\n".join([
            f"state: {state.get('sim_state', 'unknown')}",
            f"episode: {state.get('episode_label', 'none')}",
            f"step: {state.get('sim_step', 0)}",
            f"step fps: {self._format_value(state.get('sim_fps', 0.0))}",
            f"sim/wall: {self._format_value(state.get('sim_to_wall_ratio', 0.0))}x",
            f"sim time: {self._format_value(state.get('sim_time', 0.0))}s",
            f"step dt: {self._format_value(state.get('step_dt', 0.0), precision=3)}s",
            f"waypoints: {state.get('waypoint_count', 0)}",
        ])

    def _format_value(self, value, precision=2):
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{precision}f}"
        if isinstance(value, (list, tuple)) and len(value) <= 4:
            return "[" + ", ".join(self._format_value(x, precision=precision) for x in value) + "]"
        return str(value)

    def save_settings(self, settings_type):
        episode_label = self.get_ui_value(self.ui_map, "episode_label")
        self.ui_episode = self.vln_sim.episode_list[self.episode_label_list.index(episode_label)].copy()
        if settings_type == "episode_runtime":
            for key, (value, _) in self.ui_map.items():
                if key in ["episode_info", "episode_label"]:
                    continue
                self.ui_episode[value] = self.get_ui_value(self.ui_map, key)
            self.vln_sim.reset(self.ui_episode)

class TaskGeneratorUI(BaseUI):
    def __init__(self, vln_sim, task_generator):
        super().__init__(vln_sim)
        self.vln_sim = vln_sim
        self.task_generator = task_generator
        self.args = vln_sim.args
        self.episode_label_list = vln_sim.episode_label_list
        self.episode_list = vln_sim.episode_list
        self.manager_env = vln_sim.manager_env
        self.env = vln_sim.env
        self.ui_episode = None
        self.scene_config = None
        self.scene_folder = Path(self.args.scene_folder)
        self.astar_plan = None
        self.astar_plan_running = False
        self._astar_plan_update_sub = None
        self._astar_plan_refresh_sub = None
        self._astar_settle_callback = None
        self._astar_settle_state = None
        # initialize ui
        self.build_ui()
        # map ui_episode fields to ui_elements
        # format: {ui_name: (config_name, [getter_func, setter_func])}
        self.ui_config_map = {
            "scene_id": ("scene_id", choice_func(task_generator.scene_id_list)),
            "usd_path": ("path", [
                lambda x: x.as_string.replace(self.args.scene_folder+"/", "").replace(self.args.scene_folder, ""),
                lambda x, y: x.set_value(str(self.scene_folder / y))
            ]),
            "navmesh_preset": ("navmesh_preset", choice_func(task_generator.navmesh_preset_list)),
            "scene_scale": ("scene_scale", float_func),
            "ceiling_height": ("ceiling_height", float_func),
            "collider": ("collider", bool_func),
            "align_ground": ("align_ground", bool_func),
            "episode_number": ("episode_number", int_func),
            "rule_pattern": ("rule_pattern", choice_func(task_generator.rule_pattern_list)),
        }
        navmesh_settings_keys = [self._camel_to_snake(x) for x in self.task_generator.navmesh_interface.settings.keys()]
        self.navmesh_settings_map = {f"navmesh_settings_{x}": (x, float_func) for x in navmesh_settings_keys}
        # add callback to sim
        def episode_changed_callback(x):
            self.update_ui("scene_id", self.vln_sim.next_episode.scene_id)
            self.update_ui("navmesh_preset", self.vln_sim.next_episode["navmesh_preset"])
        self.vln_sim.add_callback('client_episode_changed', episode_changed_callback)

    def build_ui(self):
        with self.ui_elements["main_stack"]:
            with self.create_frame("Scene Settings"):
                self.ui_elements["scene_id"] = ui_utils.dropdown_builder(
                    "Scene Id",
                    items=self.task_generator.scene_id_list,
                    on_clicked_fn=lambda x: self.update_ui("scene_id", x)
                )
                ui_utils.btn_builder("Load Scene", text="Load", on_clicked_fn=lambda: self.load_scene())
                ui_utils.btn_builder("Save Scene Settings", text="Save", on_clicked_fn=lambda: self.save_settings("scene"))
                ui_utils.btn_builder("ReLoad Task Config", text="ReLoad", on_clicked_fn=lambda: self.reload_config())
                self.ui_elements["usd_path"] = ui_utils.str_builder(
                    "USD Path",
                    use_folder_picker=True,
                    default_val=self.args.scene_folder,
                    folder_dialog_title="Select Scene USD",
                    bookmark_label="Scene Folder",
                    bookmark_path=self.args.scene_folder,
                )
                self.ui_elements["scene_scale"] = ui_utils.combo_floatfield_slider_builder(
                    "Scene Scale",
                    default_val=1.0,
                    min=0.01,
                    max=100,
                    step=0.01,
                )[0]
                self.ui_elements["ceiling_height"] = ui_utils.combo_floatfield_slider_builder(
                    "Ceiling Clip Range",
                    default_val=1.0,
                    min=0,
                    max=100,
                    step=0.1,
                )[0]
                self.ui_elements["ceiling_height"].add_value_changed_fn(lambda x: self.task_generator.update_bev_camera_clip(self.ui_episode["scene_id"], "ceiling", x.as_float))
                self.ui_elements["collider"] = ui_utils.cb_builder("Collider", default_val=True)
                self.ui_elements["align_ground"] = ui_utils.cb_builder("Align Ground", default_val=True)
            with self.create_frame("Navmesh Settings", collapsed=True):
                self.ui_elements["navmesh_preset"] = ui_utils.dropdown_builder(
                    "Navmesh Preset",
                    items=self.task_generator.navmesh_preset_list,
                    on_clicked_fn=lambda x: self.update_ui("navmesh_preset", x)
                )
                for key, value in self.task_generator.navmesh_interface.settings.items():
                    key = self._camel_to_snake(key)
                    self.ui_elements[f"navmesh_settings_{key}"] = ui_utils.combo_floatfield_slider_builder(
                        key,
                        default_val=value,
                        min=0.01*value,
                        max=100*value,
                        step=0.001,
                    )[0]
                ui_utils.btn_builder("Navmesh Config", text="Save", on_clicked_fn=lambda: self.save_settings("navmesh_config"))
            with self.create_frame("Navmesh Tools"):
                ui_utils.btn_builder("Save Geometry", text="Save", on_clicked_fn=lambda: self.save_geometry())
                ui_utils.btn_builder("Setup Geometry", text="Setup", on_clicked_fn=lambda: self.setup_navmesh())
                ui_utils.btn_builder("Build Navmesh", text="Build", on_clicked_fn=lambda: self.build_navmesh())
                ui_utils.btn_builder("Load Navmesh", text="Load", on_clicked_fn=lambda: self.load_navmesh())
                ui_utils.btn_builder("Save Navmesh", text="Save", on_clicked_fn=lambda: self.save_navmesh())
                ui_utils.btn_builder("Test Navmesh", text="Test", on_clicked_fn=lambda: self.test_navmesh())
                ui_utils.btn_builder("Print Robot Position", text="Print", on_clicked_fn=lambda: self.print_robot_position())
                ui_utils.btn_builder("Teleport Robot", text="Teleport", on_clicked_fn=lambda: self.teleport_robot())
                ui_utils.btn_builder("Generate Cube", text="Generate", on_clicked_fn=lambda: self.generate_cube())
                ui_utils.btn_builder("Clear Visualization", text="Clear", on_clicked_fn=lambda: self.clear_visualization())
            with self.create_frame("Astar Tools"):
                ui_utils.btn_builder("Astar plan", text="Astar plan", on_clicked_fn=lambda: self.start_astar_plan_to_goal_object())
                ui_utils.btn_builder("Astar exec", text="Astar exec", on_clicked_fn=lambda: self.exec_astar_plan())
                ui_utils.btn_builder("Navmesh exec", text="Navmesh exec", on_clicked_fn=lambda: self.exec_navmesh_plan())
                ui_utils.btn_builder("Stop exec", text="Stop exec", on_clicked_fn=lambda: self.stop_exec())
                self.ui_elements["astar_status"] = ui_utils.ui.Label(
                    "No A* plan",
                    style_type_name_override="Label::label",
                    word_wrap=True,
                    alignment=ui_utils.ui.Alignment.LEFT_TOP,
                )
            with self.create_frame("Episode Settings"):
                # episode number
                self.ui_elements["episode_number"] = ui_utils.int_builder("Episode Number", default_val=30)
                self.ui_elements["rule_pattern"] = ui_utils.dropdown_builder(
                    "Rule Pattern",
                    items=self.task_generator.rule_pattern_list,
                    on_clicked_fn=lambda x: self.save_settings("scene_runtime")
                )
                ui_utils.btn_builder("Generate Episode", text="Generate", on_clicked_fn=lambda: self.generate_episodes())
                ui_utils.btn_builder("Stop Generation", text="Stop", on_clicked_fn=lambda: self.stop_generation())
                ui_utils.btn_builder("Get Information", text="Get", on_clicked_fn=lambda: self.get_information())
                ui_utils.btn_builder("Toggle Ceiling", text="Toggle", on_clicked_fn=lambda: self.task_generator.toggle_ceiling(self.ui_episode["scene_id"]))
                ui_utils.btn_builder("Get World Size", text="Get", on_clicked_fn=lambda: self.task_generator.get_world_bb(self.ui_episode["scene_id"]))
                ui_utils.btn_builder("Create BEV Map", text="Create", on_clicked_fn=lambda: self.task_generator.create_bev_map(self.ui_episode["scene_id"], clip_range="ceiling"))
                ui_utils.btn_builder("Save Occupancy Map", text="Save", on_clicked_fn=lambda: self.task_generator.create_bev_map(self.ui_episode["scene_id"], clip_range="robot", file_name="height_map"))
            with self.create_frame("Info"):
                self.ui_elements["task_config"] = ui_utils.ui.Label(
                    "Task Config",
                    style_type_name_override="Label::label",
                    word_wrap=True,
                    alignment=ui_utils.ui.Alignment.LEFT_TOP,
                )
    def _snake_to_camel(self, s):
        parts = s.split('_')
        return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])
    
    def _camel_to_snake(self, s):
        return ''.join(['_' + char.lower() if char.isupper() else char for char in s]).lstrip('_')

    def reload_config(self):
        scene_id = self.ui_episode["scene_id"]
        self.episode_ui = self.vln_sim.episode_list[self.episode_label_list.index(scene_id)]
        self.update_ui("scene_id", scene_id)
        self.update_ui("navmesh_preset", self.ui_episode["navmesh_preset"])
        print("[UI] Task config reloaded")

    def update_ui(self, settings_type, selected_value):
        if settings_type == "navmesh_preset":
            preset_name = selected_value
            print(f"[UI] Update navmesh_preset to {preset_name}")
            for key, (value, _) in self.navmesh_settings_map.items():
                self.set_ui_value(self.navmesh_settings_map, key, self.task_generator.task_config['navmesh'][value])
        elif settings_type == "scene_id":
            scene_id = selected_value
            self.task_generator._set_runtime_config(scene_id)
            self.scene_config = self.task_generator.episode_data(scene_id)
            self.scene_config["episode_number"] = self.task_generator.cfg.flow.get("episode_number", 50)
            self.scene_config["rule_pattern"] = self.task_generator.cfg.flow.get("rule_pattern", "name")
            self.ui_episode = VLNEpisode(self.scene_config)
            for key, (value, _) in self.ui_config_map.items():
                self.set_ui_value(self.ui_config_map, key, self.scene_config[value])
            print("[UI] Loaded goal rules:", self.task_generator.cfg.sample.get("goal_rules", {}))
            print("[UI] Loaded excluded paths:", self.scene_config["navmesh_exclude"])


    def save_settings(self, settings_type):
        if settings_type == "navmesh_runtime":
            for key, (value, _) in self.navmesh_settings_map.items():
                value_camel = self._snake_to_camel(value)
                self.task_generator.navmesh_interface.settings[value_camel] = self.get_ui_value(self.navmesh_settings_map, key)
        elif settings_type == "navmesh_config":
            for key, (value, _) in self.navmesh_settings_map.items():
                self.task_generator.task_config['navmesh'][value] = self.get_ui_value(self.navmesh_settings_map, key)
            self.task_generator.save_config()
        elif settings_type == "scene":
            for key, (value, _) in self.ui_config_map.items():
                self.scene_config[value] = self.get_ui_value(self.ui_config_map, key)
                self.task_generator.update_config(self.scene_config)
            self.task_generator.save_config()
        elif settings_type == "scene_runtime":
            for key, (value, _) in self.ui_config_map.items():
                self.ui_episode[value] = self.get_ui_value(self.ui_config_map, key)

    def load_scene(self):
        self.save_settings("scene_runtime")
        self.vln_sim.reset(self.ui_episode)

    def _current_scene_id(self):
        if self.ui_episode is not None:
            return self.ui_episode["scene_id"]
        return self.scene_config["scene_id"]

    def _current_navmesh_scene_config(self):
        scene_config = self.task_generator.episode_data(self._current_scene_id())
        if self.ui_episode is not None:
            for _, (config_name, _) in self.ui_config_map.items():
                if config_name in self.ui_episode:
                    scene_config[config_name] = self.ui_episode[config_name]
        return scene_config

    def _navmesh_path(self):
        return str(self.scene_folder / f"navmesh/{self._current_scene_id()}_navmesh.bin")

    def _navmesh_preprocess_config(self):
        preprocess_config = self.task_generator._navmesh_preprocess_config()
        self.task_generator.navmesh_interface.preprocess_config = preprocess_config
        return preprocess_config

    def _expected_navmesh_metadata(self):
        scene_config = self._current_navmesh_scene_config()
        preprocess_config = self._navmesh_preprocess_config()
        expected_metadata = {
            "scene_id": scene_config["scene_id"],
            "scene_type": scene_config.get("scene_type"),
            "settings": self.task_generator.navmesh_interface.serialized_settings(),
        }
        preprocess_key = navmesh_utils.navmesh_preprocess_cache_key(scene_config.get("scene_type"), preprocess_config)
        if preprocess_key is not None:
            expected_metadata["preprocess"] = preprocess_key
        return expected_metadata

    def _navmesh_cache_matches_runtime(self, navmesh_path):
        meta_path = f"{navmesh_path}.meta.json"
        if not os.path.exists(navmesh_path):
            return False
        if not os.path.exists(meta_path):
            print("[UI] Navmesh metadata is missing; rebuilding navmesh cache.")
            return False
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        expected_metadata = self._expected_navmesh_metadata()
        return all(metadata.get(k) == v for k, v in expected_metadata.items())

    def _navmesh_include_paths(self):
        return self._current_navmesh_scene_config().get("navmesh_include", ["/World/ground/terrain"])

    def _navmesh_exclude_paths(self):
        return self._current_navmesh_scene_config().get("navmesh_exclude", [])
    
    def save_geometry(self):
        self.task_generator.navmesh_interface.save_geometry(
            self._navmesh_include_paths(),
            self._navmesh_exclude_paths(),
            self.manager_env.scene.stage,
            self.scene_folder / "navmesh",
            scene_type=self._current_navmesh_scene_config().get("scene_type"),
            preprocess_config=self._navmesh_preprocess_config(),
        )
    
    def setup_navmesh(self):
        self.save_settings("scene_runtime")
        selected_paths = self._navmesh_include_paths()
        start_time = time.time()
        self.task_generator.navmesh_interface.setup_navmesh(
            selected_paths,
            self._navmesh_exclude_paths(),
            self.manager_env.scene.stage,
            scene_type=self._current_navmesh_scene_config().get("scene_type"),
            preprocess_config=self._navmesh_preprocess_config(),
        )
        print(f"[INFO]: Navmesh geometry setup time: {time.time() - start_time:.2f} seconds")

    def build_navmesh(self):
        self.save_settings("scene_runtime")
        self.save_settings("navmesh_runtime")
        start_time = time.time()
        scene_config = self._current_navmesh_scene_config()
        if scene_config.get("scene_type") == "innout":
            self.task_generator.navmesh_interface.build_navmesh_innout(
                self._navmesh_include_paths(),
                self._navmesh_exclude_paths(),
                self.manager_env.scene.stage,
            )
        else:
            self.task_generator.navmesh_interface.setup_navmesh(
                self._navmesh_include_paths(),
                self._navmesh_exclude_paths(),
                self.manager_env.scene.stage,
                scene_type=scene_config.get("scene_type"),
                preprocess_config=self._navmesh_preprocess_config(),
            )
            self.task_generator.navmesh_interface.build_navmesh()
        print(f"[INFO]: Navmesh build time: {time.time() - start_time:.2f} seconds")

    def load_navmesh(self):
        self.save_settings("scene_runtime")
        self.save_settings("navmesh_runtime")
        navmesh_path = self._navmesh_path()
        if self._navmesh_cache_matches_runtime(navmesh_path):
            self.task_generator.navmesh_interface.load_navmesh(navmesh_path)
            scene_config = self._current_navmesh_scene_config()
            self.task_generator.navmesh_interface.configure_building_exclusion(
                self._navmesh_include_paths(),
                self.manager_env.scene.stage,
                scene_type=scene_config.get("scene_type"),
            )
        else:
            print("[UI] Navmesh cache metadata is stale or missing, rebuilding before load...")
            self.build_navmesh()
            self.save_navmesh()
        self.test_navmesh(teleport=False)
        self.teleport_robot()

    def test_navmesh(self, teleport=True):
        self.task_generator.navmesh_interface.visualize_navmesh()
        points = self.task_generator.navmesh_interface.sample_random_points(1000)
        if points is not None:
            visualize_points(points, prim_path="/World/RandomPoints", width=0.2)
            for i in range(50):
                path = self.task_generator.navmesh_interface.find_paths(points[2*i], points[2*i+1])
                visualize_curve(path, prim_path=f"/World/Path_{i}", width=0.1)
        if teleport:
            self.teleport_robot()

    def save_navmesh(self):
        self.save_settings("scene_runtime")
        self.save_settings("navmesh_runtime")
        os.makedirs(self.scene_folder / "navmesh", exist_ok=True)
        navmesh_path = self._navmesh_path()
        self.task_generator.navmesh_interface.save_navmesh(navmesh_path, metadata=self._expected_navmesh_metadata())

    def print_robot_position(self):
        try:
            pos = self.manager_env.scene["robot"].data.root_state_w[0, 0:3].detach().cpu().numpy()
        except Exception as exc:
            print(f"[UI] Failed to read robot position: {exc}")
            return
        x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
        print(f"[UI] Current robot position: x={x:.3f}, y={y:.3f}, z={z:.3f}")

    def _reset_robot_for_teleport(self, robot):
        if hasattr(robot.data, "default_joint_pos") and hasattr(robot, "write_joint_position_to_sim"):
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = torch.zeros_like(robot.data.joint_vel)
            robot.write_joint_position_to_sim(joint_pos)
            robot.write_joint_velocity_to_sim(joint_vel)
            if hasattr(robot, "set_joint_position_target"):
                robot.set_joint_position_target(joint_pos)
            if hasattr(robot, "set_joint_velocity_target"):
                robot.set_joint_velocity_target(joint_vel)
        if hasattr(self.env, "_reset_low_level_policy_state"):
            self.env._reset_low_level_policy_state()

    def _write_robot_teleport_state(self, robot, root_state):
        if getattr(self.manager_env.cfg, "control_mode", None) == "kinematic_displacement" and hasattr(robot, "write_root_pose_to_sim"):
            robot.write_root_pose_to_sim(root_state[:, :7])
            if hasattr(robot, "write_root_velocity_to_sim"):
                robot.write_root_velocity_to_sim(root_state[:, 7:])
        else:
            robot.write_root_state_to_sim(root_state)
        self._reset_robot_for_teleport(robot)
        robot.write_data_to_sim()

    def teleport_robot(self):
        # self.ui_episode["start_position"] = navmesh_interface.sample_random_points(1)[0]
        # env.reset(self.ui_episode)
        with torch.inference_mode():
            robot = self.manager_env.scene["robot"]
            robot_root_state = robot.data.default_root_state.clone()
            random_pos = self.task_generator.navmesh_interface.sample_random_points(robot_root_state.shape[0])
            if random_pos is None:
                print("[UI] Cannot teleport robot: navmesh is not built or loaded")
                return
            random_pos[:, 2] += getattr(self.manager_env.cfg, "robot_spawn_height", 0.6)
            robot_root_state[:, 0:3] = torch.tensor(random_pos, device=robot.device, dtype=robot_root_state.dtype)
            robot_root_state[:, 7:] = 0.0
            self._write_robot_teleport_state(robot, robot_root_state)
            self.manager_env.sim.forward()

    def generate_cube(self):
        prim_selection = omni.usd.get_context().get_selection()
        selected_prim_paths = prim_selection.get_selected_prim_paths()
        # create a cube for first selected prim
        if len(selected_prim_paths) > 0:
            prim_path = selected_prim_paths[0]
            cube_path = "/World/Cube"
            # remove cube if it exists
            if prim_utils.get_prim_at_path(cube_path):
                prim_utils.delete_prim(cube_path)
            bb_cache = bounds_utils.create_bbox_cache()
            min_x, min_y, min_z, max_x, max_y, max_z = bounds_utils.compute_combined_aabb(bb_cache, prim_paths=[prim_path])
            cfg_cube = sim_utils.CuboidCfg(size=[1.0, 1.0, 1.0])
            position = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]
            cfg_cube.func(cube_path, cfg_cube, translation=list(position))
        else:
            print("[ERROR]: No prim selected")

    def clear_visualization(self):
        self.vln_sim.env.remove_prim("/World/RandomPoints")
        self.vln_sim.env.remove_prim("/World/Path_*")
        self.vln_sim.env.remove_prim("/World/AstarReferencePath")
        self.vln_sim.env.remove_prim("/World/AstarPlannedPath")
        self.vln_sim.env.remove_prim("/World/AstarPlanEndpoints")
        self.vln_sim.env.remove_prim("/World/ground/navmeshmesh")

    def _set_astar_status(self, text):
        print(f"[UI] {text}")
        if "astar_status" in self.ui_elements:
            self.ui_elements["astar_status"].text = text

    def _sync_task_generator_scene_runtime(self):
        self.save_settings("scene_runtime")
        self.save_settings("navmesh_runtime")
        scene_id = self.ui_episode["scene_id"]
        self.task_generator._set_runtime_config(scene_id)
        scene_config = self.task_generator.episode_data(scene_id)
        for _, (config_name, _) in self.ui_config_map.items():
            if config_name in self.ui_episode:
                scene_config[config_name] = self.ui_episode[config_name]
        self.task_generator.cfg.scene.update({
            key: value
            for key, value in scene_config.items()
            if key in self.task_generator.cfg.scene
        })
        self.task_generator.rule_pattern = self.task_generator.cfg.flow.get("rule_pattern", "name")
        self.vln_sim.runtime_config_path = self.task_generator.cfg.scene.get("config")
        return scene_config

    def _astar_ui_path_length_bounds(self, scene_id):
        astar_config = self.vln_sim._astar_config(scene_id)
        default_max = 45.0 if self.task_generator.rule_pattern == "vc_store" else 50.0
        self.task_generator._set_runtime_config(scene_id)
        default_max = float(self.task_generator.cfg.sample.get("max_path_length", default_max))
        min_length = float(astar_config.ui.min_path_length_m)
        max_length = default_max if astar_config.ui.max_path_length_m is None else float(astar_config.ui.max_path_length_m)
        return min_length, max_length

    def _astar_ui_active_termination_reason(self):
        if getattr(self.args, "disable_termination", False):
            return None
        try:
            reset_buf = self.env.termination_manager.compute()
            if not bool(reset_buf.any()):
                return None
            terms = self.env.termination_manager.get_active_iterable_terms(0)
            if not terms:
                return "unknown"
            return max(terms, key=lambda item: item[1])[0]
        except Exception:
            return None

    def _astar_ui_reset_termination_state(self):
        try:
            self.env.reset_termination_states([self.vln_sim.robot_index])
        except Exception:
            pass
        for candidate in (self.env, getattr(self.env, "env", None), self.manager_env):
            episode_length_buf = getattr(candidate, "episode_length_buf", None)
            if episode_length_buf is None:
                continue
            try:
                episode_length_buf[self.vln_sim.robot_index] = 0
            except Exception:
                pass
        try:
            self.env.termination_manager.reset()
        except Exception:
            pass
        try:
            self.env.measure_manager.reset_measures()
        except Exception:
            pass
        self.vln_sim.done = None
        if isinstance(getattr(self.vln_sim, "info", None), dict):
            self.vln_sim.info.pop("terminations", None)

    def _astar_ui_robot_tilt_ok(self, robot):
        try:
            quat_wxyz = robot.data.root_quat_w[0].detach().cpu().numpy().astype(float)
            roll, pitch, _ = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_euler("xyz")
        except Exception:
            return True
        astar_config = self.vln_sim._astar_config()
        max_tilt = np.deg2rad(float(astar_config.ui.settle_max_tilt_deg))
        return abs(float(roll)) <= max_tilt and abs(float(pitch)) <= max_tilt

    def _sample_astar_goal_object_plan(self, scene_id, max_attempts=100):
        default_start_clearance = self.task_generator.start_clearance_default(scene_id, mode="ui")
        astar_config = self.vln_sim._astar_config(scene_id)
        start_clearance_value = astar_config.ui.start_clearance_m
        if start_clearance_value is None:
            start_clearance_value = default_start_clearance
        start_clearance = float(start_clearance_value)
        goal_clearance = float(astar_config.ui.goal_clearance_m)
        min_path_length, max_path_length = self._astar_ui_path_length_bounds(scene_id)
        goal_checks_per_start = max(1, int(astar_config.ui.goal_checks_per_start))

        plan = self.task_generator.sample_goal_object_plan(
            scene_id,
            {
                "sample_mode": "ui_astar_goal_object",
                "trajectory_planner": "navmesh",
                "max_attempts": max_attempts,
                "goal_checks_per_start": goal_checks_per_start,
                "start_clearance_m": start_clearance,
                "goal_clearance_m": goal_clearance,
                "min_path_length_m": min_path_length,
                "max_path_length_m": max_path_length,
                "start_sample_attempts": 40,
                "start_sample_batch_size": 128,
            },
        )
        if plan.get("sampled"):
            reference_path = np.asarray(plan["reference_path"], dtype=float)
            print(
                f"[UI] A* sample accepted: goal={plan['goal_name']}, path={float(plan['reference_path_length_m']):.1f}m, "
                f"start_clearance={float(plan['start_clearance_m']):.2f}m, goal_clearance={float(plan['goal_clearance_m']):.2f}m"
            )
            plan["start_position"] = np.asarray(plan["start_position"], dtype=float)
            plan["goal_position"] = np.asarray(plan["goal_position"], dtype=float)
            plan["goal_radius"] = float(plan.get("goal_radius", 0.0) or 0.0)
            plan["reference_path"] = reference_path
            plan["reference_path_length"] = float(plan["reference_path_length_m"])
            plan["start_clearance"] = float(plan["start_clearance_m"])
            plan["goal_clearance"] = float(plan["goal_clearance_m"])
            return plan

        reject_counts = plan.get("reject_counts", {})
        if reject_counts:
            top_rejections = ", ".join(
                f"{key}={value}" for key, value in sorted(reject_counts.items(), key=lambda item: item[1], reverse=True)[:4]
            )
        else:
            top_rejections = "none"
        raise RuntimeError(
            "Failed to find a stable reachable A* sample "
            f"after {max_attempts} start attempts "
            f"(length={min_path_length:.1f}-{max_path_length:.1f}m, "
            f"start_clearance>={start_clearance:.2f}m, goal_clearance>={goal_clearance:.2f}m, "
            f"rejects={top_rejections})"
        )

    def _teleport_robot_to_astar_start(self, start_pos, reference_path):
        start_pos = np.asarray(start_pos, dtype=float)
        if len(reference_path) > 1:
            yaw = float(np.arctan2(reference_path[1][1] - reference_path[0][1], reference_path[1][0] - reference_path[0][0]))
        else:
            yaw = 0.0
        quat_xyzw = R.from_euler("z", yaw).as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        robot = self.manager_env.scene["robot"]
        with torch.inference_mode():
            root_state = robot.data.default_root_state.clone()
            root_pos = start_pos.copy()
            root_pos[2] += getattr(self.manager_env.cfg, "robot_spawn_height", 0.6)
            root_state[:, 0:3] = torch.tensor(root_pos, device=robot.device, dtype=root_state.dtype)
            root_state[:, 3:7] = torch.tensor(quat_wxyz, device=robot.device, dtype=root_state.dtype)
            root_state[:, 7:] = 0.0
            self._write_robot_teleport_state(robot, root_state)
        self.vln_sim.clear_waypoints()
        return yaw

    def _reference_path_to_navmesh_waypoints(self, reference_path):
        reference_path = np.asarray(reference_path, dtype=float)
        if len(reference_path) < 2:
            raise RuntimeError("Cached A* plan has no navmesh reference trajectory")
        return [[float(point[0]), float(point[1]), float("inf")] for point in reference_path[1:]]

    def _set_exec_stop_called(self, value):
        if hasattr(self.env, "set_stop_called"):
            self.env.set_stop_called(self.vln_sim.robot_index, bool(value))

    def _visualize_astar_plan(self, plan, waypoints):
        self.vln_sim.env.remove_prim("/World/AstarReferencePath")
        self.vln_sim.env.remove_prim("/World/AstarPlannedPath")
        self.vln_sim.env.remove_prim("/World/AstarPlanEndpoints")

        start_pos = np.asarray(plan["start_position"], dtype=float)
        astar_config = self.vln_sim._astar_config(plan.get("scene_id"))
        vis_z = float(start_pos[2] + astar_config.ui.visual_z_offset_m)
        reference_path = np.asarray(plan["reference_path"], dtype=float).copy()
        reference_path[:, 2] = vis_z
        astar_curve = np.asarray([[start_pos[0], start_pos[1], vis_z]] + [[wp[0], wp[1], vis_z] for wp in waypoints], dtype=float)
        endpoints = np.asarray(
            [
                [start_pos[0], start_pos[1], vis_z],
                [plan["goal_position"][0], plan["goal_position"][1], vis_z],
            ],
            dtype=float,
        )
        visualize_curve(reference_path, prim_path="/World/AstarReferencePath", color=(0.0, 0.45, 1.0), width=0.12)
        visualize_curve(astar_curve, prim_path="/World/AstarPlannedPath", color=(1.0, 0.1, 0.0), width=0.18)
        visualize_points(endpoints, prim_path="/World/AstarPlanEndpoints", color=(0.0, 1.0, 0.0), width=0.5)
        print(
            f"[UI] Visualized A* plan at z={vis_z:.2f}: "
            "/World/AstarReferencePath, /World/AstarPlannedPath, /World/AstarPlanEndpoints"
        )

    def _cancel_astar_settle_check(self):
        if self._astar_settle_callback is not None:
            try:
                self.vln_sim.remove_callback("step_finished", self._astar_settle_callback)
            except ValueError:
                pass
        self._astar_settle_callback = None
        self._astar_settle_state = None

    def _complete_astar_plan(self, plan):
        try:
            scene_id = plan["scene_id"]
            self.vln_sim.astar_ui_scene_id = scene_id
            self.vln_sim.visualize_waypoints = False
            self._teleport_robot_to_astar_start(plan["start_position"], plan["reference_path"])
            waypoints = self.vln_sim.generate_astar_path(
                scene_id,
                plan["reference_path"],
                astar_overrides={"bev_mode": "local"},
                goal_position=plan["goal_position"],
                goal_radius=plan.get("goal_radius"),
            )
            if len(waypoints) == 0:
                raise RuntimeError("A* planner returned no waypoints")

            self._visualize_astar_plan(plan, waypoints)
            plan["waypoints"] = waypoints
            self.astar_plan = plan
            self._set_astar_status(
                f"A* plan ready: {plan['goal_name']} ({len(waypoints)} waypoints, "
                f"navmesh ref {plan['reference_path_length']:.1f}m)"
            )
            self._schedule_astar_post_update("_astar_plan_refresh_sub", self._refresh_after_astar_plan)
        except Exception as e:
            self.astar_plan = None
            self._set_astar_status(f"A* plan failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.astar_plan_running = False
            self._cancel_astar_settle_check()

    def _astar_settle_reject_reason(self, before):
        robot = self.manager_env.scene["robot"]
        reason = self._astar_ui_active_termination_reason()
        if reason in {"bad_orientation", "terrain_out_of_bounds", "illegal_contact"}:
            return reason
        after = robot.data.root_state_w[0, :3].detach().cpu().numpy().astype(float)
        if not np.isfinite(after).all():
            return "non_finite_robot_pose"
        xy_drift = float(np.linalg.norm(after[:2] - before[:2]))
        z_drop = float(before[2] - after[2])
        astar_config = self.vln_sim._astar_config()
        max_xy_drift = float(astar_config.ui.settle_max_xy_drift_m)
        max_z_drop = float(astar_config.ui.settle_max_z_drop_m)
        if xy_drift > max_xy_drift:
            return f"xy_drift:{xy_drift:.2f}>{max_xy_drift:.2f}"
        if z_drop > max_z_drop:
            return f"z_drop:{z_drop:.2f}>{max_z_drop:.2f}"
        if not self._astar_ui_robot_tilt_ok(robot):
            return "bad_orientation"
        return None

    def _start_astar_settle_check(self, plan, attempts_left):
        astar_config = self.vln_sim._astar_config(plan["scene_id"])
        settle_frames = int(astar_config.ui.settle_check_frames)
        if settle_frames <= 0:
            self._complete_astar_plan(plan)
            return

        self._cancel_astar_settle_check()
        robot = self.manager_env.scene["robot"]
        robot_state = self.vln_sim._snapshot_robot_sim_state()
        self._astar_ui_reset_termination_state()
        self._teleport_robot_to_astar_start(plan["start_position"], plan["reference_path"])
        before = robot.data.root_state_w[0, :3].detach().cpu().numpy().astype(float)
        self._astar_settle_state = {
            "plan": plan,
            "attempts_left": attempts_left,
            "robot_state": robot_state,
            "before": before,
            "frames_seen": 0,
            "settle_frames": settle_frames,
        }
        self._set_astar_status(f"Settling A* start for {settle_frames} sim frames...")

        def _on_step_finished():
            state = self._astar_settle_state
            if state is None:
                return
            state["frames_seen"] += 1
            reason = self._astar_settle_reject_reason(state["before"])
            if reason is None and state["frames_seen"] < state["settle_frames"]:
                return

            self._cancel_astar_settle_check()
            if reason is None:
                self._set_astar_status("A* start settled; generating path...")
                self._schedule_astar_post_update(
                    "_astar_plan_update_sub",
                    lambda: self._complete_astar_plan(state["plan"]),
                )
                return

            self.vln_sim._restore_robot_sim_state(state["robot_state"])
            self._astar_ui_reset_termination_state()
            if state["attempts_left"] <= 0:
                self.astar_plan = None
                self.astar_plan_running = False
                self._set_astar_status(f"A* plan failed: start did not settle ({reason})")
                return
            self._set_astar_status(f"A* start rejected ({reason}); sampling another...")
            self._schedule_astar_post_update(
                "_astar_plan_update_sub",
                lambda: self._attempt_astar_plan_candidate(state["plan"]["scene_id"], state["attempts_left"] - 1),
            )

        self._astar_settle_callback = _on_step_finished
        self.vln_sim.add_callback("step_finished", _on_step_finished)

    def _attempt_astar_plan_candidate(self, scene_id, attempts_left):
        if not self.astar_plan_running:
            return
        try:
            plan = self._sample_astar_goal_object_plan(scene_id, max_attempts=1)
        except Exception as e:
            if attempts_left > 0:
                self._set_astar_status(f"A* candidate rejected ({e}); sampling another...")
                self._schedule_astar_post_update(
                    "_astar_plan_update_sub",
                    lambda: self._attempt_astar_plan_candidate(scene_id, attempts_left - 1),
                )
                return
            self.astar_plan = None
            self.astar_plan_running = False
            self._set_astar_status(f"A* plan failed: {e}")
            import traceback
            traceback.print_exc()
            return
        self._complete_astar_plan(plan)

    def _schedule_astar_post_update(self, attr_name, callback):
        existing_subscription = getattr(self, attr_name, None)
        if existing_subscription is not None:
            try:
                existing_subscription.unsubscribe()
            except Exception:
                pass
            setattr(self, attr_name, None)

        def _run_once(_event):
            subscription = getattr(self, attr_name, None)
            if subscription is None:
                return
            setattr(self, attr_name, None)
            try:
                subscription.unsubscribe()
            except Exception:
                pass
            callback()

        app_interface = omni.kit.app.get_app_interface()
        subscription = app_interface.get_post_update_event_stream().create_subscription_to_pop(_run_once)
        setattr(self, attr_name, subscription)

    def _refresh_after_astar_plan(self):
        try:
            self.manager_env.sim.forward()
            self.manager_env.sim.render()
        except Exception as exc:
            print(f"[UI] A* post-plan viewport refresh failed: {exc}")

    def start_astar_plan_to_goal_object(self):
        if self.astar_plan_running:
            self._set_astar_status("A* plan is already running...")
            return
        self.astar_plan_running = True
        self._cancel_astar_settle_check()
        self._set_astar_status("A* plan queued...")

        def _run_plan():
            self.plan_astar_to_goal_object()

        try:
            self._schedule_astar_post_update("_astar_plan_update_sub", _run_plan)
        except Exception as exc:
            print(f"[UI] Failed to queue A* plan on Kit update stream, running directly: {exc}")
            _run_plan()

    def plan_astar_to_goal_object(self):
        try:
            scene_config = self._sync_task_generator_scene_runtime()
            scene_id = scene_config["scene_id"]
            self._set_astar_status(f"Planning A* in {scene_id}...")
            self.task_generator.check_navmesh(scene_id)
            astar_config = self.vln_sim._astar_config(scene_id)
            attempts = max(1, int(astar_config.ui.max_candidate_attempts))
            self._attempt_astar_plan_candidate(scene_id, attempts - 1)
        except Exception as e:
            self.astar_plan = None
            self.astar_plan_running = False
            self._cancel_astar_settle_check()
            self._set_astar_status(f"A* plan failed: {e}")
            import traceback
            traceback.print_exc()

    def exec_astar_plan(self):
        if self.astar_plan is None:
            self._set_astar_status("No cached A* plan. Click Astar plan first.")
            return
        self.vln_sim.astar_ui_scene_id = self.astar_plan["scene_id"]
        self.vln_sim.visualize_waypoints = False
        self.vln_sim._configure_astar_follower_for_scene(self.astar_plan["scene_id"])
        self.vln_sim._apply_astar_goal_stop_radius(self.astar_plan.get("goal_radius"))
        self.vln_sim.waypoint_path_planner = "astar"
        self._teleport_robot_to_astar_start(self.astar_plan["start_position"], self.astar_plan["reference_path"])
        self._set_exec_stop_called(False)
        self.vln_sim.env.remove_prim("/World/WaypointPath")
        self.vln_sim.env.remove_prim("/World/Arrow")
        self.vln_sim.set_waypoints(self.astar_plan["waypoints"], verbose=True)
        self.vln_sim.commands_source = "ui_astar_goal_object"
        self._set_astar_status(
            f"Executing A* plan to {self.astar_plan['goal_name']} "
            f"({len(self.astar_plan['waypoints'])} waypoints)"
        )

    def exec_navmesh_plan(self):
        if self.astar_plan is None:
            self._set_astar_status("No cached A* plan. Click Astar plan first.")
            return
        try:
            waypoints = self._reference_path_to_navmesh_waypoints(self.astar_plan["reference_path"])
        except Exception as e:
            self._set_astar_status(f"Navmesh exec failed: {e}")
            return

        self.vln_sim.astar_ui_scene_id = self.astar_plan["scene_id"]
        self.vln_sim.visualize_waypoints = False
        self.vln_sim._configure_astar_follower_for_scene(self.astar_plan["scene_id"])
        self.vln_sim._apply_astar_goal_stop_radius(self.astar_plan.get("goal_radius"))
        self.vln_sim.waypoint_path_planner = "navmesh"
        self._teleport_robot_to_astar_start(self.astar_plan["start_position"], self.astar_plan["reference_path"])
        self._set_exec_stop_called(False)
        self.vln_sim.env.remove_prim("/World/WaypointPath")
        self.vln_sim.env.remove_prim("/World/Arrow")
        self.vln_sim.set_waypoints(waypoints, verbose=True)
        self.vln_sim.commands_source = "ui_navmesh_goal_object"
        self._set_astar_status(
            f"Executing navmesh ref to {self.astar_plan['goal_name']} "
            f"({len(waypoints)} waypoints)"
        )

    def stop_exec(self):
        self._cancel_astar_settle_check()
        try:
            self.vln_sim._handle_stop_command()
            self.vln_sim.waypoint_path_planner = None
            self.vln_sim.astar_ui_scene_id = None
            self._set_astar_status("Stopped waypoint execution")
        except Exception as e:
            self._set_astar_status(f"Stop exec failed: {e}")

    def generate_episodes(self):
        self.save_settings("scene")
        self.save_settings("scene_runtime")
        self.save_settings("navmesh_config")
        self.save_settings("navmesh_runtime")
        self.task_generator.generate_episodes(self.ui_episode["scene_id"])
    
    def stop_generation(self):
        self.task_generator.stop_generation()

    def get_information(self):
        # Get current robot world pose
        pos = self.manager_env.scene["robot"].data.root_state_w[0, 0:3].cpu().numpy()
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        print(f"[UI] Scene type: {self.scene_config['scene_type']}")

        # Build base directory by combining scene_folder and path (drop filename)
        json_path = str(self.scene_folder / os.path.dirname(self.scene_config["path"])) if isinstance(self.scene_config.get("path"), str) else str(self.scene_folder)
        print(f"[UI] JSON path: {json_path}")
        try:
            from navverse.utils.vc_location_info_utils import CityDataReader
            reader = CityDataReader(json_path)
        except Exception as e:
            print(f"[ERROR] Failed to init CityDataReader: {e}")
            return

        # Query nearest road and nearby interesting points
        nearest_road = reader.get_nearest_road(x, y)
        nearby_points = reader.get_points_in_radius(x, y, radius=200.0)

        print("[UI] Current Position:", {"x": x, "y": y, "z": z})
        if nearest_road:
            rn = nearest_road.get("name", "unknown")
            rt = nearest_road.get("type", "unknown")
            dist = nearest_road.get("distance_to_road", None)
            cp = nearest_road.get("closest_point_on_road", None)
            print(f"[UI] Nearest road: name={rn}, type={rt}, distance={dist:.2f}m" if dist is not None else f"[UI] Nearest road: name={rn}, type={rt}")
            if cp is not None:
                print(f"[UI] Closest point on road: ({cp[0]:.2f}, {cp[1]:.2f})")
        else:
            print("[UI] No nearby road found")

        print(f"[UI] Nearby points within 200m: {len(nearby_points)}")
        for p in nearby_points[:5]:
            name = p.get("name", "-")
            ptype = p.get("type", "-")
            cat = p.get("category", "-")
            d = p.get("distance_from_center", None)
            if d is not None:
                print(f"  - {name} ({ptype}/{cat}), {d:.1f}m")
            else:
                print(f"  - {name} ({ptype}/{cat})")
        
        # --- Sample one episode and aggregate road/POI names along its trajectory ---
        episodes_file = f"episodes/{self.ui_episode['scene_id']}.json"
        if not os.path.exists(episodes_file):
            print(f"[UI] Episodes file not found: {episodes_file}; skipping trajectory info aggregation")
            return
        
        try:
            with open(episodes_file, 'r') as f:
                episodes = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load episodes from {episodes_file}: {e}")
            return
        
        if not episodes:
            print("[UI] No episodes in file; skipping trajectory info aggregation")
            return

        # Pick a random episode
        sampled_episode = random.choice(episodes)
        # Collect all reference path points from all goals
        trajectory_points = []
        for g in sampled_episode.get("goals", []):
            ref_path = g.get("reference_path", [])
            if isinstance(ref_path, list):
                trajectory_points.extend(ref_path)

        if not trajectory_points:
            print("[UI] Sampled episode has no reference_path; skipping info aggregation")
            return

        road_names = set()
        poi_names = set()

        for p in trajectory_points:
            if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                continue
            px, py = float(p[0]), float(p[1])
            # nearest road
            nearest_road = reader.get_nearest_road(px, py)
            if nearest_road:
                rn = nearest_road.get("name")
                if rn:
                    road_names.add(rn)
            # nearby POIs within 200m
            nearby_points = reader.get_points_in_radius(px, py, radius=200.0)
            for poi in nearby_points or []:
                name = poi.get("name")
                if name:
                    poi_names.add(name)

        print(f"[UI] Sampled episode id: {sampled_episode.get('episode_id')}")
        print(f"[UI] Roads along trajectory ({len(road_names)}): {sorted(road_names)}")
        print(f"[UI] POIs along trajectory ({len(poi_names)}): {sorted(poi_names)}")
