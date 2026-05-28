# SAC.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from RL.ReplayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Actor ------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)  # salida: media de la acción (μ)
        self.log_std = nn.Linear(256, action_dim)  # salida: logaritmo de la desviación estándar (log σ)
        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=device) 

    def forward(self, x):
        """
        Pasa la observación (estado) por la red para obtener
        los parámetros de la distribución de la acción.
        Retorna:
          mean → media de la distribución Normal
          std  → desviación estándar (positiva)
        """
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean(x)      # salida media (μ)
        log_std = self.log_std(x).clamp(-20, 2)   # salida logaritmo de la desviación estándar (log σ)
                                                  # Limitamos el rango con clamp de log_std para evitar que σ sea muy grande o muy pequeña
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        """
        Muestra una acción estocástica a partir del estado actual.
        Devuelve:
          action   → acción muestreada y limitada (por tanh)
          log_prob → probabilidad logarítmica corregida por el cambio de variable tanh
        """
        state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)  # Evitar errores numéricos si hay NaN o infinitos en el estado
        mean, std = self.forward(state) # Obtener media y desviación estándar de la política
        
        # Limpiar posibles valores problemáticos y asegurar límites razonables
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.nan_to_num(std, nan=0.1, posinf=1.0, neginf=1e-3)        
        std = torch.clamp(std, 1e-6, 1.0)   # σ mínima 1e-6, máxima 1.0
        normal = torch.distributions.Normal(mean, std)  # Crear la distribución normal de acciones:  N(mean, std)
        # Muestrear usando reparametrización (permite gradientes)
        # z = μ + σ * ε   , donde ε ~ N(0,1)
        z = normal.rsample()

        # Pasar la acción por tanh para limitarla a [-1, 1]
        # y escalarla a su rango físico real multiplicando por max_action
        action = torch.tanh(z) * self.max_action

        # Calcular la probabilidad logarítmica de la acción
        # Corrige el cambio de variable por la función tanh:
        # log π(a) = log N(z | μ, σ) - log(1 - tanh(z)^2)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)

        # Sumar las probabilidades de cada dimensión de acción
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        
        return action, log_prob  # Acción muestreada y su probabilidad logarítmica


# ------------------ Critic ------------------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.q(x)

# ------------------ SAC Agent ------------------
# Implementa el algoritmo Soft Actor-Critic (SAC).
# Es un método de aprendizaje por refuerzo continuo, fuera de política (off-policy),
# que maximiza tanto la recompensa esperada como la entropía (exploración).

class SAC:
    def __init__(self, state_dim, action_dim, max_action,
                 gamma=0.99, tau=0.005, lr=3e-4, alpha=0.2,
                 buffer_size=int(1e6), batch_size=256):
        self.gamma = gamma          # Factor de descuento de recompensas futuras
        self.tau = tau              # Factor de interpolación para actualización suave (soft update)
        self.alpha = alpha          # Coeficiente de entropía (balancea exploración y explotación)
        self.batch_size = batch_size
        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=device)

        # Actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critics
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) +
                                           list(self.critic2.parameters()), lr=lr)

        # Replay buffer (importado)
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    # ------------------ Acción para inferencia ------------------
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # Reemplazar NaN por 0
        state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    # ------------------ Recompensa ------------------
    def compute_reward(self,
                   state,
                   next_state,
                   goal,
                   obstacles,
                   current_action,
                   previous_action):


        pos = np.array([state[0], state[1]])
        next_pos = np.array([next_state[0], next_state[1]])
        goal_pos = np.array(goal)
    
        # r_d  progreso
        dist_prev = np.linalg.norm(pos - goal_pos)
        dist_curr = np.linalg.norm(next_pos - goal_pos)
    
        if dist_curr <= 0.5:
            r_d = 10
    
        elif dist_curr < dist_prev:
            r_d = 2
    
        else:
            r_d = -1
    
        # r_c  colisión
        
        collision = False
    
        for ox, oy in obstacles:
    
            d = np.linalg.norm(next_pos - np.array([ox, oy]))
    
            if d < 0.5:
                collision = True
                break
    
        if collision:
            r_c = -10
        else:
            r_c = 5

        # r_s -> smoothness
        lambda_s = 0.1
        v_i, w_i = current_action
        v_prev, w_prev = previous_action
    
        r_s = lambda_s * (
            (v_i - v_prev) ** 2 +
            (w_i - w_prev) ** 2
        )
    
        # Recompensa total
        reward = r_d + r_c - r_s
        return reward

    # ------------------ Entrenamiento ------------------
    def train_step(self):
        if self.buffer.size < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.float32, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        done = torch.tensor(done, dtype=torch.float32, device=device)

        # -------- Critic update --------
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob    # Tomar el mínimo de ambos (Double Q-learning)
            target_q = reward + (1 - done) * self.gamma * target_q   # Retorno objetivo de Bellman

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # -------- Actor update --------
        action_new, log_prob_new = self.actor.sample(state)
        actor_loss = (self.alpha * log_prob_new - torch.min(self.critic1(state, action_new),
                                                            self.critic2(state, action_new))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # -------- Target update --------
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # ------------------ Guardar / Cargar ------------------
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic1.state_dict(), filename + "_critic1.pth")
        torch.save(self.critic2.state_dict(), filename + "_critic2.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.critic1.load_state_dict(torch.load(filename + "_critic1.pth", map_location=device))
        self.critic2.load_state_dict(torch.load(filename + "_critic2.pth", map_location=device))

