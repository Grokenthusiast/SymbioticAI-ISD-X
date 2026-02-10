import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class SymbioticX15D(gym.Env):
    """
    SymbioticAI-15D-X: A Multidimensional Reinforcement Learning Framework.
    
    This environment transitions from "Attention Economy" (3D) to "Awareness Economy" (15D).
    It models the interplay between algorithmic engagement, cognitive health, and systemic 
    resource consumption (The Technological Shadow).
    """
    def __init__(self):
        super(SymbioticX15D, self).__init__()

        # --- ACTION SPACE ---
        # 0: High-Dopamine Viral (Short-term gain)
        # 1: Deep Knowledge/Educational (Medium-term growth)
        # 2: Somatic Decompression (Recovery/Health)
        # 3: Fractal Discovery (Breaking the filter bubble)
        # 4: Systemic Silence (Mandatory metabolic brake)
        self.action_space = spaces.Discrete(5)

        # --- OBSERVATION SPACE (The 15D State) ---
        # [0] Cognitive Load (Dim 5) - Mental fatigue
        # [1] Bubble Diversity (Dim 7) - Information variety
        # [2] Session Entropy (Dim 8) - Temporal decay
        # [3] Somatic Resonance (Dim 13) - User's physical well-being
        # [4] Shadow Noise (Dim 14) - Presence of bots/inauthentic data
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initializing the FelipeCore state
        self.state = np.array([0.2, 0.5, 0.0, 1.0, 0.1], dtype=np.float32)
        self.steps_taken = 0
        return self.state, {}

    def _calculate_shadow_interference(self):
        """Simulates Dimension 14: Technological Shadow (Bot noise)"""
        return random.uniform(0, 0.3) if random.random() > 0.8 else 0.0

    def step(self, action):
        cognitive_load, diversity, entropy, somatic, shadow = self.state
        
        reward = 0.0
        done = False
        info = {"status": "Active"}

        # --- 15D LOGIC ENGINE ---
        if action == 0:  # VIRAL CONTENT
            engagement = 1.2
            mental_cost = 0.25
            reward = engagement * somatic  # If user is tired, viral content hurts more.
            cognitive_load += mental_cost
            somatic -= 0.15
            diversity -= 0.05

        elif action == 1:  # DEEP KNOWLEDGE
            reward = 0.8
            cognitive_load += 0.05
            diversity += 0.15
            somatic -= 0.02

        elif action == 2:  # DECOMPRESSION
            reward = 0.1  # Low immediate engagement
            cognitive_load -= 0.4
            somatic += 0.3
            diversity += 0.02

        elif action == 3:  # FRACTAL DISCOVERY
            reward = 0.5 + (diversity * 0.5)
            cognitive_load += 0.1
            diversity += 0.3
            shadow += 0.05 # Exploration exposes more raw data (shadow)

        elif action == 4:  # SYSTEMIC SILENCE
            reward = -0.5 if cognitive_load < 0.5 else 1.5
            cognitive_load = 0.0
            somatic += 0.5
            info["status"] = "Metabolic Brake Activated"

        # --- DIMENSION 14: SHADOW FILTERING ---
        current_shadow = self._calculate_shadow_interference()
        if current_shadow > 0.2:
            reward -= 1.0 # Penalize the algorithm for promoting bot-heavy trends
            shadow = current_shadow

        # --- THE 15D REWARD PENALTY (CRASH-PROOFING) ---
        # If Cognitive Load exceeds 0.8, the system enters "Autophagy" (Self-Eating)
        if cognitive_load > 0.8:
            reward -= 3.0 
            info["status"] = "CRITICAL: COGNITIVE OVERLOAD"

        # If Somatic health drops too low
        if somatic < 0.2:
            reward -= 2.0
            info["status"] = "CRITICAL: SOMATIC COLLAPSE"

        # Update State
        self.steps_taken += 1
        entropy = min(1.0, self.steps_taken / 50.0)
        
        self.state = np.clip(
            [cognitive_load, diversity, entropy, somatic, shadow], 0, 1
        )

        # Terminate if user is exhausted or time is up
        if self.state[0] >= 1.0 or self.state[3] <= 0.0 or entropy >= 1.0:
            done = True
            info["status"] = "Session Terminated: System Equilibrium Reached"

        return self.state, reward, done, False, info

# --- SIMULATION RUNNER ---
if __name__ == "__main__":
    env = SymbioticX15D()
    obs, _ = env.reset()
    
    print(">>> Initializing SymbioticAI-15D-X Simulation")
    print("-" * 50)

    for i in range(20):
        # A real RL agent (PPO/DQN) would make decisions here
        # For simulation, we use a 'Smart-Random' heuristic
        if obs[0] > 0.7:
            action = 4 # Force break
        elif obs[1] < 0.3:
            action = 3 # Force diversity
        else:
            action = random.choice([0, 1, 2])

        obs, reward, done, _, info = env.step(action)
        
        print(f"Step {i+1} | Action: {action} | Load: {obs[0]:.2f} | Somatic: {obs[3]:.2f} | Reward: {reward:.2f}")
        if info["status"] != "Active":
            print(f"Log: {info['status']}")
            
        if done:
            print("-" * 50)
            print(">>> Equilibrium reached. Resetting FelipeCore.")
            break