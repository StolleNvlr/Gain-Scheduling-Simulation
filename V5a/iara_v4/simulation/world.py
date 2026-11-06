"""Scenario definitions in SI units.

The ``coppelia_default`` scenario mirrors the three cuboids in the supplied
CoppeliaSim scene.  The other scenarios are synthetic stress cases used only by
the Python plant; they are *not* claimed to be CoppeliaSim replicas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..core.geometry import (
    CircleObstacle,
    Obstacle,
    RectangleObstacle,
    point_clearance,
)
from ..core.types import Pose2D, Vec2


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    start: Pose2D
    goal: Vec2
    obstacles: tuple[Obstacle, ...]
    bounds: tuple[float, float, float, float]
    timeout: float = 60.0
    goal_tolerance: float = 0.08
    description: str = ""
    hidden_obstacles: tuple[Obstacle, ...] = ()
    metadata: dict[str, float | str | bool] = field(default_factory=dict)

    @property
    def all_obstacles(self) -> tuple[Obstacle, ...]:
        """Obstacles known to the planner plus unmapped obstacles.

        The planner receives only ``obstacles``.  The plant, clearance metric and
        range sensor receive ``all_obstacles``.  This is how the VFH+ ablation
        tests reaction to an unmapped obstacle without changing the global map.
        """

        return self.obstacles + self.hidden_obstacles

    def __post_init__(self) -> None:
        min_x, min_y, max_x, max_y = self.bounds
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("invalid scenario bounds")
        if not (min_x <= self.start.x <= max_x and min_y <= self.start.y <= max_y):
            raise ValueError("start is outside bounds")
        if not (min_x <= self.goal.x <= max_x and min_y <= self.goal.y <= max_y):
            raise ValueError("goal is outside bounds")
        if self.timeout <= 0.0:
            raise ValueError("timeout must be positive")
        if point_clearance(self.start.position, self.all_obstacles) < 0.12:
            raise ValueError("start is too close to an obstacle")
        if point_clearance(self.goal, self.all_obstacles) < 0.12:
            raise ValueError("goal is too close to an obstacle")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": list(self.start.as_tuple()),
            "goal": list(self.goal.as_tuple()),
            "obstacles": [ob.to_dict() for ob in self.obstacles],  # type: ignore[attr-defined]
            "hidden_obstacles": [ob.to_dict() for ob in self.hidden_obstacles],  # type: ignore[attr-defined]
            "bounds": list(self.bounds),
            "timeout": self.timeout,
            "goal_tolerance": self.goal_tolerance,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        def parse(items) -> list[Obstacle]:
            result: list[Obstacle] = []
            for item in items:
                kind = item.get("kind")
                if kind == "rectangle":
                    result.append(RectangleObstacle.from_dict(item))
                elif kind == "circle":
                    result.append(CircleObstacle.from_dict(item))
                else:
                    raise ValueError(f"unknown obstacle kind: {kind!r}")
            return result

        sx, sy, st = data["start"]
        return cls(
            name=str(data["name"]),
            start=Pose2D(float(sx), float(sy), float(st)),
            goal=Vec2(float(data["goal"][0]), float(data["goal"][1])),
            obstacles=tuple(parse(data.get("obstacles", []))),
            bounds=tuple(float(v) for v in data["bounds"]),  # type: ignore[arg-type]
            timeout=float(data.get("timeout", 60.0)),
            goal_tolerance=float(data.get("goal_tolerance", 0.08)),
            description=str(data.get("description", "")),
            hidden_obstacles=tuple(parse(data.get("hidden_obstacles", []))),
            metadata=dict(data.get("metadata", {})),
        )


def _coppelia_obstacles() -> tuple[Obstacle, ...]:
    # Coordinates and sizes were measured from the supplied scene (SI metres).
    return (
        RectangleObstacle(1.0, 0.5, 0.7, 0.5, "Cuboid_1"),
        RectangleObstacle(1.925, -0.025, 0.75, 0.45, "Cuboid_2"),
        RectangleObstacle(3.15, 0.9, 0.8, 0.5, "Cuboid_3"),
    )


def default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="coppelia_default",
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Vec2(4.0, 0.0),
            obstacles=_coppelia_obstacles(),
            bounds=(-0.5, -1.5, 4.5, 1.8),
            timeout=60.0,
            description="Three cuboids measured from the supplied CoppeliaSim scene.",
            metadata={"source": "coppelia", "visualized": True},
        ),
        Scenario(
            name="open_water",
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Vec2(3.5, 0.0),
            obstacles=(),
            bounds=(-0.5, -1.0, 4.0, 1.0),
            timeout=45.0,
            description="No obstacles; sanity check for tracking and speed profile.",
        ),
        Scenario(
            name="single_block",
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Vec2(4.0, 0.0),
            obstacles=(RectangleObstacle(2.0, 0.0, 0.8, 0.6, "block"),),
            bounds=(-0.5, -1.5, 4.5, 1.5),
            timeout=55.0,
            description="One centred block; both sides are feasible.",
        ),
        Scenario(
            name="slalom",
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Vec2(4.2, 0.0),
            obstacles=(
                RectangleObstacle(1.2, 0.4, 0.5, 0.5, "gate_1"),
                RectangleObstacle(2.2, -0.4, 0.5, 0.5, "gate_2"),
                RectangleObstacle(3.2, 0.4, 0.5, 0.5, "gate_3"),
            ),
            bounds=(-0.5, -1.5, 4.8, 1.5),
            timeout=65.0,
            description="Three staggered gates.",
        ),
        Scenario(
            name="narrow_passage",
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Vec2(4.0, 0.0),
            obstacles=(
                RectangleObstacle(2.0, 0.75, 1.2, 0.8, "wall_top"),
                RectangleObstacle(2.0, -0.75, 1.2, 0.8, "wall_bottom"),
            ),
            bounds=(-0.5, -1.6, 4.5, 1.6),
            timeout=70.0,
            description="0.7 m gap; tests the conservatism of the disc footprint.",
        ),
        Scenario(
            name="u_trap",
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Vec2(4.0, 0.0),
            obstacles=(
                RectangleObstacle(2.0, 0.5, 1.2, 0.2, "trap_top"),
                RectangleObstacle(2.0, -0.5, 1.2, 0.2, "trap_bottom"),
                RectangleObstacle(2.6, 0.0, 0.2, 1.0, "trap_back"),
            ),
            bounds=(-0.5, -1.5, 4.5, 1.5),
            timeout=70.0,
            description="U-shaped trap for the APF local-minimum baseline.",
        ),
        Scenario(
            name="hidden_block",
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Vec2(4.0, 0.0),
            obstacles=(),
            hidden_obstacles=(RectangleObstacle(2.0, 0.0, 0.6, 0.6, "unmapped_block"),),
            bounds=(-0.5, -1.5, 4.5, 1.5),
            timeout=60.0,
            description="Unmapped block on the direct route; VFH+ should react.",
            metadata={"unmapped": True},
        ),
        Scenario(
            name="coppelia_hidden",
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Vec2(4.0, 0.0),
            obstacles=_coppelia_obstacles(),
            hidden_obstacles=(RectangleObstacle(2.0, -0.7, 0.8, 0.5, "unmapped_lower_block"),),
            bounds=(-0.5, -1.6, 4.5, 1.8),
            timeout=70.0,
            description="Scene cuboids plus an unmapped block in the lower corridor.",
            metadata={"source": "coppelia", "unmapped": True, "visualized": False},
        ),
        Scenario(
            name="offset_goal",
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Vec2(3.8, 1.3),
            obstacles=_coppelia_obstacles(),
            bounds=(-0.5, -1.5, 4.5, 1.8),
            timeout=65.0,
            description="Same cuboids as the scene with an off-axis goal.",
        ),
    ]


def scenario_by_name(name: str) -> Scenario:
    for scenario in default_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(f"unknown scenario {name!r}")
