# Entrono.py


import numpy as np
import math

class Config:
    def __init__(self):
        self.dt = 0.1
        self.robot_radius = 2.0     # Radio del robot
        self.ob = np.random.uniform(-1, 15, size=(10, 2))  # obstáculos iniciales

class Entorno:
    def __init__(self, config=None):
        self.config = config if config is not None else Config()
        self.reset()

    def reset(self, start=[0.0, 0.0], yaw=math.pi/8.0):
        self.x = np.array([start[0], start[1], yaw, 0.0, 0.0])  # [x, y, yaw, v, w]
        self.obst_tray = [self.config.ob.copy()]
        return self.get_state()

    def motion(self, action):
        """Actualiza el estado del robot según la acción [v, w]"""
        v, w = action
        self.x[2] += w * self.config.dt
        self.x[0] += v * math.cos(self.x[2]) * self.config.dt
        self.x[1] += v * math.sin(self.x[2]) * self.config.dt
        self.x[3] = v
        self.x[4] = w
        return self.get_state()

    def update_obstacles(self, step):
        """Mueve los obstáculos según una trayectoria tipo onda"""
        ob = self.config.ob.copy()
        for i in range(len(ob)):
            ob[i][1] += 0.05 * np.sin(0.1 * step + i)
            ob[i][0] += 0.05 * np.cos(0.05 * step + i)
        self.config.ob = ob
        self.obst_tray.append(ob.copy())
        return ob

    def get_state(self, goal=[20, 15]):
        """Devuelve el estado extendido para los algoritmos"""
        # Obstáculos rellenados hasta 10 pares
        ob_2 = [tuple(par) for par in self.config.ob]
        while len(ob_2) < 10:
            ob_2.append((0.0, 0.0))
        ob_flat = [coord for obs in ob_2 for coord in obs]
        return list(self.x) + goal + ob_flat

    def distance_to_goal(self, goal=[20, 15]):
        return math.hypot(self.x[0] - goal[0], self.x[1] - goal[1])

    def get_trajectory(self):
        return np.array(self.obst_tray)
