from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics  # bom pra logs

%pip install stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv  # Import DummyVecEnv

import torch as th
from typing import Callable

# Supondo que SymbioticX15D e EnvConfig estão no mesmo notebook ou importados
# from your_module import SymbioticX15D, EnvConfig

def linear_decay_lr(initial_lr: float = 3e-4) -> Callable[[float], float]:
    """Decay linear clássico, mas com mínimo pra não zerar."""
    def func(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return func


def cosine_decay_lr(initial_lr: float = 3e-4) -> Callable[[float], float]:
    """Alternativa mais moderna (usada em muitos papers recentes)."""
    import math
    def func(progress_remaining: float) -> float:
        return initial_lr * (1 + math.cos(math.pi * (1 - progress_remaining))) / 2
    return func


def train_symbiotic_ai(n_envs: int = 8, total_timesteps: int = 200_000):
    config = EnvConfig(max_steps=200)

    # Vectorized envs!!! Isso muda TUDO
    def make_train_env():
        env = SymbioticX15D(config=config)
        env = Monitor(env)
        env = RecordEpisodeStatistics(env)  # logs episode reward/length automáticos
        return env

    vec_env = make_vec_env(make_train_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)  # ou Dummy pra debug

    # Arquitetura mais leve pro seu obs pequeno
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=th.nn.ReLU,
        ortho_init=True,  # default bom
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cosine_decay_lr(3e-4),  # experimente trocar por linear_decay_lr
        n_steps=2048,          # mais samples por update → melhor pra PPO
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # ajuda exploração em ambientes punitivos
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./symbiotic_15d_logs_v3/",
        device="cuda" if th.cuda.is_available() else "cpu",
    )

    # Callbacks
    def make_eval_env():
        env = SymbioticX15D(config=config)
        env = Monitor(env)
        return env

    # Explicitly make eval_env a VecEnv for consistency
    eval_env = make_vec_env(make_eval_env, n_envs=1, vec_env_cls=DummyVecEnv) # Use DummyVecEnv for eval

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=80.0, verbose=1)  # Ajuste esse valor!!!
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=5000 // n_envs,  # ajustado pro n_envs
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    print("🧠 Iniciando propagação de gradientes quântico-simbióticos em paralelo...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,  # legal no colab
    )

    model.save("ppo_symbiotic_v3_15d")
    print("✅ Cérebro simbiótico salvo. Versão 3.0 – agora com vetorização.")

    vec_env.close()
    eval_env.close() # Close eval_env as well
    return model, eval_env.envs[0] # Return the unwrapped environment for testing, or handle VecEnv in test_equilibrium


def test_equilibrium(model, env: gym.Env, n_episodes: int = 3):
    print("\n" + "═"*50)
    print("🧪 TESTE DE ESTABILIDADE DO FELIPECORE v3")
    print("═"*50)

    for episode in range(1, n_episodes + 1):
        # env.reset() returns (obs, info) for a single environment
        # For VecEnv, it returns (obs_array, info_dict)
        # If env is a DummyVecEnv of 1, obs will be [obs], so we take obs[0]
        obs_array, _ = env.reset()
        obs = obs_array[0] if isinstance(env, DummyVecEnv) else obs_array # Adjust for VecEnv vs single env
        total_reward = 0.0
        done = False
        step = 0

        print(f"\nEpisódio {episode}")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # step returns (obs_array, rewards, dones, infos) for VecEnv
            obs_array, reward_array, terminated_array, truncated_array, info_array = env.step(action)
            obs = obs_array[0] # Take the observation from the first (and only) environment
            reward = reward_array[0]
            terminated = terminated_array[0]
            truncated = truncated_array[0]
            info = info_array[0]
            
            done = terminated or truncated
            total_reward += reward
            step += 1

            load, _, _, somatic, _ = obs
            load_emoji = "🔥" if load > 0.7 else "🧘" if load < 0.3 else "⚖️"
            print(f"  Step {step:03} | Ação: {action} | Load: {load:.2f} {load_emoji} | Somatic: {somatic:.2f} | Rew: {reward:+.2f}")

            if done:
                status = info.get("status", "—")
                print(f"  Fim do episódio → {status} | Total Reward: {total_reward:.2f}")

    print("\nTeste concluído.")


if __name__ == "__main__":
    try:
        # Treina com 8 envs paralelos (ajuste conforme sua máquina/colab)
        brain, test_env_unwrapped = train_symbiotic_ai(n_envs=4, total_timesteps=150_000)
        # For test_equilibrium, we need a single environment, so we pass the unwrapped one
        test_equilibrium(brain, test_env_unwrapped, n_episodes=5)
    except Exception as e:
        from loguru import logger
        logger.critical(f"Colapso neural total: {e}")