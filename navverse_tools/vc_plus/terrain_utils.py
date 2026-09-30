#!/usr/bin/env python3
"""Terrain height queries shared by VC+ generation steps."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


class TerrainHeightQuery:
    def __init__(self, city_dir: Path | None) -> None:
        self.city_dir = city_dir
        self.available = False
        self._tree: cKDTree | None = None
        self._coords: np.ndarray | None = None
        self._heights: np.ndarray | None = None
        self._mesh_triangles: np.ndarray | None = None
        self._mesh_grid: dict[tuple[int, int], list[int]] = {}
        self._mesh_cell_size = 10.0
        if city_dir is None:
            return
        path = city_dir / "height_field.npz"
        if path.exists():
            data = np.load(path)
            coords = np.asarray(data["plane_coord"], dtype=float).reshape(-1, 2)
            heights = np.asarray(data["terrain_alt"], dtype=float).reshape(-1)
            valid = np.isfinite(coords).all(axis=1) & np.isfinite(heights)
            if np.any(valid):
                coords = coords[valid].copy()
                coords[:, 1] *= -1.0
                self._coords = coords
                self._heights = heights[valid]
                self._tree = cKDTree(coords)
                self.available = True
        self._load_terrain_mesh(city_dir / "terrain.glb")

    def _load_terrain_mesh(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            import trimesh

            scene = trimesh.load(path, force="scene", process=False)
            vertices = []
            faces = []
            offset = 0
            for node in scene.graph.nodes_geometry:
                transform, geometry_name = scene.graph[node]
                geometry = scene.geometry[geometry_name]
                node_vertices = trimesh.transform_points(np.asarray(geometry.vertices, dtype=float), transform)
                vertices.append(node_vertices)
                faces.append(np.asarray(geometry.faces, dtype=np.int64) + offset)
                offset += len(node_vertices)
            if not vertices or not faces:
                return
            glb_vertices = np.vstack(vertices)
            mapped = np.column_stack([glb_vertices[:, 0], -glb_vertices[:, 2], glb_vertices[:, 1]])
            triangles = mapped[np.vstack(faces)]
            xy_a = triangles[:, 1, :2] - triangles[:, 0, :2]
            xy_b = triangles[:, 2, :2] - triangles[:, 0, :2]
            twice_xy_area = np.abs(xy_a[:, 0] * xy_b[:, 1] - xy_a[:, 1] * xy_b[:, 0])
            z_values = np.sort(mapped[:, 2])
            z_threshold = float("-inf")
            if len(z_values) > 1:
                gaps = np.diff(z_values)
                gap_index = int(np.argmax(gaps))
                if gaps[gap_index] >= 2.0:
                    z_threshold = float((z_values[gap_index] + z_values[gap_index + 1]) * 0.5)
            keep = (twice_xy_area > 1e-10) & (triangles[:, :, 2].max(axis=1) >= z_threshold)
            self._mesh_triangles = triangles[keep]
            self._build_mesh_grid()
            self.available = self.available or bool(len(self._mesh_triangles))
        except ImportError:
            self._mesh_triangles = None
            self._mesh_grid = {}
        except Exception as exc:
            raise RuntimeError(f"could not build terrain mesh height query from {path}: {exc}") from exc

    def _build_mesh_grid(self) -> None:
        if self._mesh_triangles is None:
            return
        bounds_min = self._mesh_triangles[:, :, :2].min(axis=1)
        bounds_max = self._mesh_triangles[:, :, :2].max(axis=1)
        cell_size = self._mesh_cell_size
        for index, (lower, upper) in enumerate(zip(bounds_min, bounds_max)):
            for ix in range(math.floor(lower[0] / cell_size), math.floor(upper[0] / cell_size) + 1):
                for iy in range(math.floor(lower[1] / cell_size), math.floor(upper[1] / cell_size) + 1):
                    self._mesh_grid.setdefault((ix, iy), []).append(index)

    def _mesh_height_at(self, x: float, y: float) -> float | None:
        if self._mesh_triangles is None:
            return None
        cell = (math.floor(float(x) / self._mesh_cell_size), math.floor(float(y) / self._mesh_cell_size))
        point = np.asarray([float(x), float(y)], dtype=float)
        heights = []
        for index in self._mesh_grid.get(cell, []):
            triangle = self._mesh_triangles[index]
            a, b, c = triangle[:, :2]
            denominator = np.cross(b - a, c - a)
            if abs(denominator) <= 1e-12:
                continue
            weight_b = np.cross(point - a, c - a) / denominator
            weight_c = np.cross(b - a, point - a) / denominator
            weight_a = 1.0 - weight_b - weight_c
            if min(weight_a, weight_b, weight_c) < -1e-7:
                continue
            heights.append(
                weight_a * triangle[0, 2] + weight_b * triangle[1, 2] + weight_c * triangle[2, 2]
            )
        return float(max(heights)) if heights else None

    @classmethod
    def from_roots(cls, data_root: Path, city: str, preferred_source: str = "vc_full") -> "TerrainHeightQuery":
        for source in (preferred_source, "vc_full", "vc_full_raw"):
            query = cls(data_root / source / city)
            if query.available:
                return query
        return cls(None)

    def height_at(self, x: float, y: float, fallback: float = 0.0, *, k: int = 8, max_distance: float = 6.0) -> float:
        mesh_height = self._mesh_height_at(x, y)
        if mesh_height is not None:
            return mesh_height
        if not self.available or self._tree is None or self._heights is None:
            return float(fallback)
        count = min(k, len(self._heights))
        distances, indices = self._tree.query([float(x), float(y)], k=count)
        distances = np.atleast_1d(distances).astype(float)
        indices = np.atleast_1d(indices).astype(int)
        mask = np.isfinite(distances) & (distances <= max_distance)
        if not np.any(mask):
            return float(fallback)
        distances = distances[mask]
        indices = indices[mask]
        if float(distances[0]) <= 1e-6:
            return float(self._heights[int(indices[0])])
        weights = 1.0 / np.maximum(distances, 1e-3) ** 2
        return float(np.average(self._heights[indices], weights=weights))

    def height_near(
        self,
        samples: Iterable[tuple[float, float]],
        fallback: float = 0.0,
        *,
        mode: str = "max",
    ) -> float:
        heights = [self.height_at(x, y, fallback) for x, y in samples]
        if not heights:
            return float(fallback)
        if mode == "mean":
            return float(np.mean(heights))
        if mode == "min":
            return float(np.min(heights))
        return float(np.max(heights))
