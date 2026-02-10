import numpy as np
import random
from main import SymbioticX15D # Import the 15D environment

class SymbioticAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        # Q-Table: Where the AI stores its learned patterns (Discretized for state efficiency)
        self.q_table = {} 
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0 # Initial exploration rate (Randomness)
        self.epsilon_decay = 0.995

    def get_state_key(self, state):
        # Round values to prevent state explosion and help the AI generalize
        return tuple(np.round(state, 1))

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        # Epsilon-greedy: Balance between exploring new paths and exploiting known rewards
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        # Q-Learning logic (The Bellman Equation)
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        
        # Update the Q-value based on the reward and future potential
        self.q_table[state_key][action] = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value
        )
        
        # Gradually reduce exploration as the agent matures
        self.epsilon *= self.epsilon_decay

# --- TRAINING PHASE ---
if __name__ == "__main__":
    env = SymbioticX15D()
    agent = SymbioticAgent(env.action_space.n)
    
    print("🧠 AI is learning to balance your cognitive load... Please wait.")

    for episode in range(500): # Training for 500 simulated sessions
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

    print("🔥 Training Complete! The AI has achieved Symbiotic Equilibrium.")
    
    # Final Test Result
    state, _ = env.reset()
    print(f"Initial State (15D Metrics): {state}")
    print(f"Optimal Action suggested by AI: {agent.choose_action(state)}")
