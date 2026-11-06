"""List every object in the loaded CoppeliaSim scene.

Run this script while CoppeliaSim has the V2 scene loaded. The output is used
to configure the boat and thruster paths without guessing their names.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print(
        "ERROR: pip install coppeliasim-zmqremoteapi-client",
        file=sys.stderr,
    )
    sys.exit(1)


def walk(sim, handle, depth, max_depth=6):
    if handle == sim.handle_scene:
        alias, path, obj_type = "Scene", "/", "scene"
    else:
        alias = sim.getObjectAlias(handle, 1)
        path = sim.getObjectAlias(handle, 4)
        obj_type = sim.getObjectType(handle)
    print("  " * depth + f"{alias}  (path={path}, type={obj_type})")
    if depth >= max_depth:
        return
    child_index = 0
    while True:
        child = sim.getObjectChild(handle, child_index)
        if child < 0:
            break
        walk(sim, child, depth + 1, max_depth)
        child_index += 1


def main() -> int:
    try:
        client = RemoteAPIClient()
        sim = client.require("sim")
    except Exception as exc:
        print(
            f"ERROR: could not connect to CoppeliaSim ZMQ server: {exc}",
            file=sys.stderr,
        )
        print(
            "Make sure CoppeliaSim is running and the ZMQ server is active.",
            file=sys.stderr,
        )
        return 1

    print("=== Scene hierarchy ===")
    walk(sim, sim.handle_scene, 0)
    print()
    print("=== All joints in scene ===")
    all_objects = sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)
    for handle in all_objects:
        print(
            f"  {sim.getObjectAlias(handle, 4)}  "
            f"(mode={sim.getJointMode(handle)})"
        )
    print()
    print("=== All shapes in scene ===")
    all_shapes = sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
    for handle in all_shapes:
        mass, _ = sim.getShapeMassAndInertia(handle)
        print(f"  {sim.getObjectAlias(handle, 4)}  mass={mass:.3f} kg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
