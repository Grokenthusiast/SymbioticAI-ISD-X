import gymnasium as gym
from stable_baselines3 import PPO
from main import SymbioticX15D  # Importing your 15D environment

def train_symbiotic_ai():
    # 1. Initialize the custom environment
    env = SymbioticX15D()

    # 2. Create the PPO model
    # MlpPolicy: Multi-layer Perceptron Policy (Deep Neural Network)
    # verbose=1: Displays training progress in the console
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003, 
        n_steps=2048
    )

    print("🧠 Starting Neural Training for FelipeCore via PPO...")
    
    # 3. Training phase (PPO thrives with more data/timesteps)
    model.learn(total_timesteps=10000)

    # 4. Save the trained brain (weights and biases)
    model.save("ppo_symbiotic_brain")
    print("✅ Training complete! Model saved successfully.")

    return model, env

def test_agent(model, env):
    print("\n--- REAL-TIME EQUILIBRIUM TEST ---")
    obs, _ = env.reset()
    
    for i in range(10):
        # The model predicts the optimal action based on learned policy
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, _, info = env.step(action)
        
        print(f"Step {i+1:02} | AI Action: {action} | Load: {obs[0]:.2f} | Somatic: {obs[3]:.2f} | Status: {info['status']}")
        
        if done:
            print(">>> Session terminated by AI constraints.")
            break

if __name__ == "__main__":
    trained_model, environment = train_symbiotic_ai()
    test_agent(trained_model, environment)
