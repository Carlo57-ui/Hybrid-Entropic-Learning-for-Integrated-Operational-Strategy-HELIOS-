# main_inference.py

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import imageio

from RL.SAC import SAC
from RL.TD3 import TD3
from controllers.controller_dwa import DWAController
from controllers.controller_apf import APFController
from controllers.controller_teb import TEBController
from entropy_combiner import EntropyCombiner
from Entorno import Entorno

# Nombre de pesos a cargar
load_name = "TD3+DWA"

# Inicializar agente (misma config que en entrenamiento)
#agent = SAC(state_dim=2+1+2+2+20, action_dim=2, max_action=[1.0, 3.14])
agent = TD3(state_dim=2+1+2+2+20, action_dim=2, max_action=[1.0, 3.14])

# Cargar pesos
load_path = os.path.join("pesos", load_name)
agent.load(load_path)
print(f"Pesos cargados desde {load_path}")

# Inicializar controlador
control = DWAController
#control = APFController()
#control = TEBController()

# Inicializar fusión híbrida
combiner = EntropyCombiner(H_umbral=0.3, k=0.5)

# Inicializar entorno
env = Entorno()

show_animation = True

def plot_robot(x, y, yaw, robot_radius):
    circle = plt.Circle((x, y), robot_radius, color="blue", alpha=0.5, label="Robot")
    plt.gcf().gca().add_artist(circle)
    out_x = x + robot_radius * np.cos(yaw)
    out_y = y + robot_radius * np.sin(yaw)
    plt.arrow(x, y, out_x - x, out_y - y, head_width=0.1, head_length=0.2, fc='black', ec='black')
    plt.axis("equal")
    plt.grid(True)

def main(goal=[20, 15]):
    print("Iniciando simulación de inferencia...")
    env.reset()
    step = 0
    trajectory = np.array(env.x)

    # --- NUEVO: medición de tiempo y almacenamiento de frames ---
    start_time = time.time()
    frames = []

    while True:
        # Estado actual
        estado_ext = env.get_state(goal)

        # Normalizar estado para SAC/TD3
        estado_ext_norm = np.array([
            estado_ext[0]/20.0,   # x
            estado_ext[1]/20.0,   # y
            estado_ext[2]/np.pi,  # theta
            estado_ext[3]/1.0,    # v_prev
            estado_ext[4]/3.14,   # w_prev
            estado_ext[5]/20.0,   # goal x
            estado_ext[6]/20.0,   # goal y
        ] + estado_ext[7:])

        # Acción RL
        a_p = agent.get_action(estado_ext_norm)

        # Acción del controlador
        a_r = control(estado_ext)      # DWA
        #a_r = control.plan(estado_ext)  # APF/TEB

        # Acción híbrida
        a_h, alfa, H = combiner.combine(np.array(a_p), np.array(a_r))

        # Mover robot
        #env.motion([a_p[0], a_p[1]])
        env.motion([a_h[0], a_h[1]])
        trajectory = np.vstack((trajectory, env.x))

        # Actualizar obstáculos
        env.update_obstacles(step)

        # Animación
        if show_animation:
            plt.cla()
            plt.plot(env.x[0], env.x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(env.config.ob[:, 0], env.config.ob[:, 1], "ok")
            plot_robot(env.x[0], env.x[1], env.x[2], env.config.robot_radius)

            # Dibujar y pausar
            plt.draw()
            plt.pause(0.0001)

            # --- NUEVO: guardar frame ---
            fig = plt.gcf()
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)

        step += 1
        if step == 6000 or env.distance_to_goal(goal) <= env.config.robot_radius:
            break

    # --- NUEVO: cálculo de tiempo ---
    end_time = time.time()
    elapsed_time = end_time - start_time

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"Tiempo de inferencia: {minutes:02d}:{seconds:02d}")

    # --- NUEVO: guardar GIF ---
    if len(frames) > 0:
        imageio.mimsave("simulacion.gif", frames, fps=30)
        print("Simulación guardada como simulacion.gif")

    print("Simulación de inferencia terminada.")

if __name__ == '__main__':
    main()