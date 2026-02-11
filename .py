from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final, NamedTuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
from pydantic import BaseModel, ConfigDict

try:
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit, transpile
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False


class Action(IntEnum):
    VIRAL = 0
    KNOWLEDGE = 1
    DECOMPRESSION = 2
    FRACTAL = 3
    SILENCE = 4


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 100
    burnout_limit: float = 0.85
    somatic_critical: float = 0.15
    base_recovery: float = 0.35
    shadow_threshold: float = 0.22


class State(NamedTuple):
    cognitive_load: float
    diversity: float
    entropy: float
    somatic: float
    shadow: float

    def as_array(self) -> np.ndarray:
        return np.array(self, dtype=np.float32)

    def clip(self) -> 'State':
        return State(*(np.clip(v, 0.0, 1.0) for v in self))


class SymbioticX15D(gym.Env):
    metadata = {"render_modes": ["human"], "version": "2.1.0"}

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        self._state: State | None = None
        self.steps_taken: int = 0

        # Removed logger configuration to avoid pickling issues with SubprocVecEnv
        # logger.remove()
        # logger.add(lambda msg: print(msg, end=""), format="<cyan>15D</cyan> | {level} | {message}")

    def _quantum_noise(self) -> float:
        if not HAS_QUANTUM:
            return np.random.uniform(0.0, 0.25)

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        sim = AerSimulator()
        job = sim.run(transpile(qc, sim), shots=1)
        counts = job.result().get_counts()
        return 0.30 if "1" in counts else 0.05

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.steps_taken = 0
        self._state = State(0.10, 0.50, 0.00, 1.00, 0.05)
        return self._state.as_array(), {"info": "FelipeCore initialized"}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")

        s = self._state
        reward = 0.0
        status = "Stable"

        quantum_spike = self._quantum_noise()

        match Action(action):
            case Action.VIRAL:
                reward = 1.2 * s.somatic - 0.3
                delta = State(0.30, -0.10, 0.00, -0.20, quantum_spike * 0.5)
            case Action.KNOWLEDGE:
                reward = 0.90 + s.diversity * 0.3
                delta = State(0.08, 0.20, 0.00, 0.00, 0.00)
            case Action.DECOMPRESSION:
                reward = 0.25 + (1 - s.somatic) * 0.4
                delta = State(-self.cfg.base_recovery, 0.00, 0.00, 0.28, -0.05)
            case Action.FRACTAL:
                reward = 0.55 + s.diversity * 0.50
                delta = State(0.05, 0.35, 0.00, -0.08, quantum_spike)
            case Action.SILENCE:
                reward = 2.8 if s.cognitive_load > 0.65 else -0.8
                delta = State(-s.cognitive_load, 0.00, 0.00, 0.40, -0.10)
                status = "Metabolic Brake"

        # Atualização atômica e imutável
        new_state = State(
            s.cognitive_load + delta.cognitive_load,
            s.diversity + delta.diversity,
            min(1.0, self.steps_taken / self.cfg.max_steps),
            s.somatic + delta.somatic,
            s.shadow + delta.shadow,
        ).clip()

        # Penalidades sistêmicas (depois do delta!)
        if new_state.cognitive_load >= self.cfg.burnout_limit:
            reward -= 3.2
            status = "BURNOUT ZONE"
        if new_state.somatic <= self.cfg.somatic_critical:
            reward -= 2.8
            status = "SOMATIC CRITICAL"

        if quantum_spike > self.cfg.shadow_threshold:
            reward -= 1.1

        self.steps_taken += 1
        self._state = new_state

        terminated = (
            new_state.cognitive_load >= 1.0 or
            new_state.somatic <= 0.0 or
            self.steps_taken >= self.cfg.max_steps
        )

        return (
            new_state.as_array(),
            reward,
            terminated,
            False,
            {"status": status, "quantum_spike": quantum_spike}
        )