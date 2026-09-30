"""Thin runtime integration used by the main NavVerse environment wrapper."""

from dataclasses import dataclass

import numpy as np

from navverse_tools.terrain_eval.terrain import (
    ControlledTerrainManager,
    TERRAIN_ROOT_PATH,
)


def configure_env_cfg(env_cfg, args):
    """Apply optional evaluation-only wheel losses before the environment is built."""
    if getattr(args, "robot_name", None) != "turtlebot":
        return
    wheels = env_cfg.scene.robot.actuators["wheels"]
    fields = {
        "dynamic_friction": getattr(
            args, "controlled_turtlebot_joint_dynamic_friction", None
        ),
        "viscous_friction": getattr(
            args, "controlled_turtlebot_joint_viscous_friction", None
        ),
    }
    for name, value in fields.items():
        if value is None:
            continue
        if value < 0.0:
            raise ValueError(f"Turtlebot joint {name} must be non-negative")
        setattr(wheels, name, float(value))


def _apply_physics_material(stage, args):
    friction = getattr(args, "controlled_material_friction", None)
    static = getattr(args, "controlled_material_static_friction", None)
    dynamic = getattr(args, "controlled_material_dynamic_friction", None)
    if static is None and dynamic is None and friction is None:
        return
    static = friction if static is None else static
    dynamic = friction if dynamic is None else dynamic
    if static is None or dynamic is None:
        raise ValueError("Both controlled static and dynamic friction must be specified")

    restitution = float(getattr(args, "controlled_material_restitution", 0.0))
    stiffness = float(
        getattr(args, "controlled_material_compliant_stiffness", 0.0)
    )
    damping = float(getattr(args, "controlled_material_compliant_damping", 0.0))
    values = (float(static), float(dynamic), restitution, stiffness, damping)
    if min(values) < 0.0:
        raise ValueError("Controlled material coefficients must be non-negative")
    if damping > 0.0 and stiffness <= 0.0:
        raise ValueError("Compliant damping requires positive compliant stiffness")

    from pxr import PhysxSchema, UsdPhysics, UsdShade

    scope = getattr(args, "controlled_material_scope", "default")
    material_path = (
        "/World/NavVerseControlledGroundMaterial"
        if scope == "terrain"
        else "/physicsScene/defaultMaterial"
    )
    bind_paths = ()
    if scope == "terrain":
        bind_paths = ("/World/ground/terrain",)
        if stage.GetPrimAtPath(TERRAIN_ROOT_PATH).IsValid():
            bind_paths += (TERRAIN_ROOT_PATH,)
        if not stage.GetPrimAtPath(material_path).IsValid():
            UsdShade.Material.Define(stage, material_path)

    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise RuntimeError(f"Controlled physics material prim not found: {material_path}")
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    static_attr = material_api.CreateStaticFrictionAttr()
    dynamic_attr = material_api.CreateDynamicFrictionAttr()
    restitution_attr = material_api.CreateRestitutionAttr()
    static_attr.Set(float(static))
    dynamic_attr.Set(float(dynamic))
    restitution_attr.Set(restitution)

    physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    combine_attr = physx_api.CreateFrictionCombineModeAttr()
    stiffness_attr = physx_api.CreateCompliantContactStiffnessAttr()
    damping_attr = physx_api.CreateCompliantContactDampingAttr()
    combine_attr.Set(getattr(args, "controlled_material_friction_combine_mode", "multiply"))
    stiffness_attr.Set(stiffness)
    damping_attr.Set(damping)

    for prim_path in bind_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Controlled material bind prim not found: {prim_path}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(material_prim),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )

    actual = (
        float(static_attr.Get()),
        float(dynamic_attr.Get()),
        float(stiffness_attr.Get()),
        float(damping_attr.Get()),
    )
    expected = (float(static), float(dynamic), stiffness, damping)
    if any(abs(current - target) > 1e-6 for current, target in zip(actual, expected)):
        raise RuntimeError(f"Failed to apply controlled physics material: {actual}")


@dataclass
class TerrainEvaluationRuntime:
    args: object
    terrain: ControlledTerrainManager | None = None

    @classmethod
    def from_args(cls, args):
        manifest = getattr(args, "controlled_terrain_manifest", "")
        terrain_type = getattr(args, "controlled_terrain_type", "")
        if bool(manifest) != bool(terrain_type):
            raise ValueError(
                "Controlled terrain manifest and terrain type must be provided together"
            )
        manager = None
        if manifest:
            manager = ControlledTerrainManager(
                manifest,
                terrain_type,
                terrain_profile=getattr(
                    args, "controlled_terrain_profile", "standard"
                ),
            )
        return cls(args=args, terrain=manager)

    def configure_episode(self, stage, episode, env_cfg):
        if self.terrain is not None:
            self.terrain.configure_episode(stage, episode["episode_label"])
        _apply_physics_material(stage, self.args)
        spawn_height = getattr(env_cfg, "controlled_robot_spawn_height", None)
        if self.terrain is None or spawn_height is None:
            return episode
        configured = dict(episode)
        configured["robot_spawn_height"] = float(spawn_height)
        return configured

    def measurements(self, position_xy=None):
        if self.terrain is None:
            return {}
        if position_xy is not None:
            position_xy = np.asarray(position_xy, dtype=np.float64)
            if np.all(np.isfinite(position_xy[:2])):
                return self.terrain.update_progress(position_xy[:2])
        return self.terrain.metric_values()
