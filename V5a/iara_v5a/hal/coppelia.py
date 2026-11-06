"""CoppeliaSim ZMQ Remote API HAL.

The client dependency and network connection are both lazy.  Importing this
module therefore remains safe on machines used only for deterministic V4
tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iara_v4.core.types import ControlCommand, Measurement, Pose2D

from .base import HALError, HALConfig


@dataclass(frozen=True)
class CoppeliaHALConfig(HALConfig):
    """Coppelia-specific scene and differential-drive parameters."""

    scene_path: str | None = None
    propeller_radius: float = 0.02
    track_width: float = 0.132
    host: str = "localhost"
    port: int = 23000
    boat_path: str = "/Cuboid"
    thruster_left_path: str = "/Cuboid/Dummy_thrusterLeft/thrusterJoint_left"
    thruster_right_path: str = "/Cuboid/Dummy_thrusterRight/thrusterJoint_right"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.propeller_radius <= 0.0:
            raise ValueError("propeller_radius must be positive")
        if self.track_width <= 0.0:
            raise ValueError("track_width must be positive")
        if not self.host:
            raise ValueError("host must not be empty")
        if self.port <= 0:
            raise ValueError("port must be positive")


class CoppeliaHAL:
    """CoppeliaSim plant controlled in synchronous stepping mode."""

    name = "coppelia"

    def __init__(self, host: str = "localhost", port: int = 23000) -> None:
        self.host = host
        self.port = port
        self.client: Any | None = None
        self.sim: Any | None = None
        self._handles: dict[str, int] = {}
        self._initialized = False
        self._dt = 0.05
        self._timeout = 60.0
        self._propeller_radius = 0.02
        self._track_width = 0.132
        self._paths = {
            "boat": "/iara",
            "thruster_left": "/iara/thrusterLeft",
            "thruster_right": "/iara/thrusterRight",
        }

    def _connect(self) -> None:
        if self.sim is not None:
            return
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        except ImportError as exc:
            raise HALError(
                "coppeliasim-zmqremoteapi-client is not installed; "
                "install it before using CoppeliaHAL"
            ) from exc
        try:
            self.client = RemoteAPIClient(host=self.host, port=self.port)
            self.sim = self.client.require("sim")
        except Exception as exc:
            raise HALError(
                f"could not connect to CoppeliaSim at {self.host}:{self.port}"
            ) from exc

    def reset(self, config: HALConfig) -> Measurement:
        """Load the configured scene, discover handles, and start stepping."""

        if not isinstance(config, CoppeliaHALConfig):
            raise HALError("CoppeliaHAL.reset requires CoppeliaHALConfig")
        if config.scene_path is not None and not Path(config.scene_path).is_file():
            raise HALError(f"scene not found: {config.scene_path}")
        self._dt = config.dt
        self._timeout = config.timeout
        self._propeller_radius = config.propeller_radius
        self._track_width = config.track_width
        self._paths = {
            "boat": config.boat_path,
            "thruster_left": config.thruster_left_path,
            "thruster_right": config.thruster_right_path,
        }
        self._connect()
        assert self.sim is not None
        try:
            if config.scene_path is not None:
                self.sim.loadScene(str(Path(config.scene_path).resolve()))
            self._handles = self._discover_handles()
            self.sim.setStepping(True)
            self.sim.startSimulation()
            self._initialized = True
            self.sim.step()
            return self.read_measurement()
        except HALError:
            raise
        except Exception as exc:
            self._initialized = False
            raise HALError("failed to initialize the CoppeliaSim scene") from exc

    def _discover_handles(self) -> dict[str, int]:
        if self.sim is None:
            raise HALError("CoppeliaSim is not connected")
        handles: dict[str, int] = {}
        for name, path in self._paths.items():
            try:
                handles[name] = int(self.sim.getObject(path))
            except Exception as exc:
                raise HALError(
                    f"missing CoppeliaSim object {path!r}; create or configure "
                    "the boat and both velocity-mode thruster joints"
                ) from exc
        return handles

    def read_measurement(self) -> Measurement:
        if not self._initialized or self.sim is None:
            raise HALError("HAL is not initialized; call reset() first")
        try:
            pos = self.sim.getObjectPosition(self._handles["boat"], -1)
            ori = self.sim.getObjectOrientation(self._handles["boat"], -1)
            
            # Correção: A API retorna (lista_linear, lista_angular)
            lin_vel, ang_vel = self.sim.getObjectVelocity(self._handles["boat"])
            
            psi = float(ori[2])
            cpsi, spsi = math.cos(psi), math.sin(psi)
            
            vx, vy = float(lin_vel[0]), float(lin_vel[1])
            r = float(ang_vel[2])  # Yaw rate está no eixo Z da velocidade angular
            
            t = float(self.sim.getSimulationTime())
            return Measurement(
                time=t,
                source_time=t,
                pose=Pose2D(float(pos[0]), float(pos[1]), psi),
                u=cpsi * vx + spsi * vy,
                v=-spsi * vx + cpsi * vy,
                r=r,
            )
        except Exception as exc:
            # Se der erro, mostra a mensagem real no terminal em vez de esconder
            print(f"ERRO DETALHADO NO HAL: {exc}")
            raise HALError("failed to read the CoppeliaSim boat state") from exc

    def apply_command(self, command: ControlCommand) -> None:
        if not self._initialized or self.sim is None:
            raise HALError("HAL is not initialized; call reset() first")
        
        if command.wheels is None:
            left = command.v - command.omega * self._track_width / 2.0
            right = command.v + command.omega * self._track_width / 2.0
        else:
            left, right = command.wheels.left, command.wheels.right

        current_m = self.read_measurement()
        u_current = current_m.u
        v_current = current_m.v
        r_current = current_m.r

        try:
            self.sim.setJointTargetVelocity(self._handles["thruster_left"], left / self._propeller_radius)
            self.sim.setJointTargetVelocity(self._handles["thruster_right"], right / self._propeller_radius)

            # 1. Força de Avanço (Freio mais forte para não virar foguete)
            v_cmd = (left + right) / 2.0
            drag_ff = (4.0 * v_cmd) + (2.0 * v_cmd * abs(v_cmd))
            # Aumentamos o ganho de 20 para 60 e o limite de 20 para 60
            force_forward = drag_ff + 60.0 * (v_cmd - u_current)
            force_forward = max(-60.0, min(60.0, force_forward))

            # 2. Resistência Lateral (Aumentada para parar de derrapar)
            # Aumentamos os coeficientes para 40 e 20
            force_side = -(40.0 * v_current) - (20.0 * v_current * abs(v_current))
            force_side = max(-60.0, min(60.0, force_side))

            # 3. Torque de Guinada (Mantido o que funcionou)
            w_cmd = (right - left) / self._track_width
            torque_z = (w_cmd * 0.05) + 0.1 * (w_cmd - r_current) 
            torque_z = max(-0.5, min(0.5, torque_z))

            self.sim.addForceAndTorque(
                self._handles["boat"], 
                [force_forward, force_side, 0.0], 
                [0.0, 0.0, torque_z], 
                1
            )
        except Exception as exc:
            raise HALError("failed to apply thruster command") from exc

    def set_disturbance(
        self, force_world: np.ndarray, torque_world: np.ndarray
    ) -> None:
        """Apply one world-frame external force and torque to the boat."""

        if not self._initialized or self.sim is None:
            raise HALError("HAL is not initialized; call reset() first")
        force = np.asarray(force_world, dtype=float)
        torque = np.asarray(torque_world, dtype=float)
        if force.shape != (3,) or torque.shape != (3,):
            raise ValueError("force_world and torque_world must have shape (3,)")
        try:
            self.sim.addForceAndTorque(
                self._handles["boat"], force.tolist(), torque.tolist(), -1
            )
        except Exception as exc:
            raise HALError("failed to apply external disturbance") from exc

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if not self._initialized or self.sim is None:
            raise HALError("HAL is not initialized; call reset() first")
        try:
            self.sim.step()
        except Exception as exc:
            raise HALError("failed to advance CoppeliaSim") from exc

    def is_running(self) -> float:
        if self.sim is None:
            raise HALError("CoppeliaSim is not connected")
        try:
            return float(self.sim.getSimulationTime())
        except Exception as exc:
            raise HALError("failed to read CoppeliaSim time") from exc

    def close(self) -> None:
        if self.sim is not None and self._initialized:
            try:
                self.sim.stopSimulation()
            except Exception:
                pass
        self._initialized = False
