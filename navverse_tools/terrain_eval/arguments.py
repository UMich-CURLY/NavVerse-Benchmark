"""CLI arguments consumed by the terrain-evaluation runtime hook."""

from navverse_tools.terrain_eval.terrain import TERRAIN_PROFILES, TERRAIN_TYPES


def add_terrain_eval_args(parser):
    group = parser.add_argument_group("Terrain evaluation")
    group.add_argument(
        "--controlled_terrain_manifest",
        default="",
        help="Frozen manifest shared by every terrain and embodiment cell.",
    )
    group.add_argument(
        "--controlled_terrain_type",
        default="",
        choices=("", *TERRAIN_TYPES),
        help="Terrain inserted at the three manifest placements.",
    )
    group.add_argument(
        "--controlled_terrain_profile",
        default="standard",
        choices=tuple(TERRAIN_PROFILES),
        help="Geometry profile for the inserted terrain patches.",
    )
    group.add_argument(
        "--controlled_episode_timeout_s",
        type=float,
        default=None,
        help="Optional per-episode simulation timeout.",
    )
    group.add_argument(
        "--controlled_waypoint_follower_profile",
        default="",
        choices=("", "task_generation"),
        help="Optional follower profile for stored reference paths.",
    )
    group.add_argument(
        "--controlled_reference_execution",
        default="",
        choices=("", "task_generation_internal"),
        help="Execute the stored path internally with task-generation semantics.",
    )
    group.add_argument("--controlled_material_friction", type=float, default=None)
    group.add_argument(
        "--controlled_material_scope",
        choices=("default", "terrain"),
        default="default",
    )
    group.add_argument("--controlled_material_static_friction", type=float, default=None)
    group.add_argument("--controlled_material_dynamic_friction", type=float, default=None)
    group.add_argument("--controlled_material_restitution", type=float, default=0.0)
    group.add_argument(
        "--controlled_material_compliant_stiffness", type=float, default=0.0
    )
    group.add_argument(
        "--controlled_material_compliant_damping", type=float, default=0.0
    )
    group.add_argument(
        "--controlled_material_friction_combine_mode",
        choices=("average", "min", "multiply", "max"),
        default="multiply",
    )
    group.add_argument(
        "--controlled_turtlebot_joint_dynamic_friction", type=float, default=None
    )
    group.add_argument(
        "--controlled_turtlebot_joint_viscous_friction", type=float, default=None
    )
