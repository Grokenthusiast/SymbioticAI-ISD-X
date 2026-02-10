import gymnasium as gym
from gymnasium import spaces
import numpy as np

class X_15D_Environment(gym.Env):
    """
    Ambiente de Reinforcement Learning para o Feed do X baseado no FelipeCore 15D.
    O objetivo é maximizar o Valor Sistêmico (Engajamento + Saúde Mental).
    """
    def __init__(self):
        super(X_15D_Environment, self).__init__()
        
        # Ações: 0 = Post Viral/Raso, 1 = Post Educativo/Thread, 2 = Post Descompressão
        self.action_space = spaces.Discrete(3)
        
        # Estado: [Carga Cognitiva (0-1), Diversidade da Bolha (0-1), Tempo de Sessão (min)]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        self.state = np.array([0.2, 0.5, 0.0], dtype=np.float32)

    def step(self, action):
        # Dim 5: Carga Cognitiva | Dim 13: Bem-estar
        carga, diversidade, tempo = self.state
        
        reward = 0
        done = False
        
        if action == 0: # Viral/Raso
            reward = 1.0  # Alto engajamento imediato
            carga += 0.2  # Aumenta o estresse/cansaço
        elif action == 1: # Educativo
            reward = 0.7  # Engajamento moderado
            carga -= 0.1  # Valor agregado reduz o "vazio" cognitivo
            diversidade += 0.1
        elif action == 2: # Descompressão
            reward = -0.2 # Baixo engajamento (curto prazo)
            carga -= 0.4  # Reseta o sistema (benefício longo prazo)
            
        # --- A MÁGICA 15D: Penalização por Saturação ---
        if carga > 0.8:
            reward -= 2.0 # Punição pesada por causar "vício/estresse"
            
        # Atualiza o estado
        self.state = np.clip([carga, diversidade, tempo + 0.1], 0, 1)
        
        # Condição de parada (usuário cansou ou atingiu limite)
        if tempo > 0.9 or carga > 0.95:
            done = True
            
        return self.state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.2, 0.5, 0.0], dtype=np.float32)
        return self.state, {}

# Exemplo de loop de execução
env = X_15D_Environment()
obs, _ = env.reset()

print("Iniciando Simulação de Feed 15D...")
for _ in range(10):
    # Aqui um modelo de IA (como PPO ou DQN) escolheria a ação.
    # Vamos simular uma escolha aleatória para ver a lógica:
    action = env.action_space.sample() 
    obs, reward, done, _, _ = env.step(action)
    
    print(f"Ação: {action} | Carga Cognitiva: {obs[0]:.2f} | Recompensa: {reward:.2f}")
    if done:
        print("--- Portal de Reset Ativado: Usuário precisa de pausa! ---")
        break
