# TD3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from RL.ReplayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Actor ------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=device)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x

# ------------------ Critic ------------------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.q1_layer = nn.Linear(256, 1)   
        # Q2
        self.l3 = nn.Linear(state_dim + action_dim, 256)
        self.l4 = nn.Linear(256, 256)
        self.q2_layer = nn.Linear(256, 1)   

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        # Q1
        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        q1 = self.q1_layer(x1)
        # Q2
        x2 = F.relu(self.l3(sa))
        x2 = F.relu(self.l4(x2))
        q2 = self.q2_layer(x2)
        return q1, q2

    def q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        return self.q1_layer(x1)   


# ------------------ TD3 Agent ------------------
class TD3:
    def __init__(self, state_dim, action_dim, max_action,
                 gamma=0.99, tau=0.005, lr=3e-4, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2, buffer_size=int(1e6), batch_size=256):

        self.gamma = gamma                  # Factor de descuento
        self.tau = tau                      # Parámetro de actualización suave 
        self.policy_noise = policy_noise    # Ruido agregado a acciones 
        self.noise_clip = noise_clip        # Límite para el ruido de política
        self.policy_freq = policy_freq      # Frecuencia de actualización del actor
        self.batch_size = batch_size
        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=device)
        self.total_it = 0

        # Actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    # Acción para inferencia
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]
    
    # ------------------ Recompensa -------------------------
    # Cálculo de recompensa personalizada
    # En este caso: basada en distancia al objetivo y penalización
    # por cercanía a obstáculos.
    # -------------------------------------------------------
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

        self.total_it += 1
        # Extraer un batch aleatorio del buffer
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.float32, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        done = torch.tensor(done, dtype=torch.float32, device=device)

        # ------------------ Critic update ------------------
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)  # Extraer un batch aleatorio del buffer
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ------------------ Actor update ------------------
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # Guardar y cargar pesos
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
