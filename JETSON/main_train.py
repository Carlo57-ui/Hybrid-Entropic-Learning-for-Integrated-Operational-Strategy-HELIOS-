# main_train.py

import math
import matplotlib.pyplot as plt
from openpyxl import Workbook
import numpy as np
import os

from RL.SAC import SAC
from RL.TD3 import TD3
from controllers.controller_dwa import DWAController
from controllers.controller_apf import APFController
from controllers.controller_teb import TEBController
from entropy_combiner import EntropyCombiner
from Entorno import Entorno

<<<<<<< HEAD
save_name = "TEB"
=======
save_name = "TD3+TEB"
>>>>>>> 9a97feb2250f730d780612d4826361557ed292d3

# Inicializar agentes
#agent = SAC(state_dim=2+1+2+2+20, action_dim=2, max_action=[1.0, 3.14])
agent = TD3(state_dim=2+1+2+2+20, action_dim=2, max_action=[1.0, 3.14])

# Control clásico
control = TEBController()
#control = APFController()
#control = DWAController

# Fusión híbrida
combiner = EntropyCombiner(H_umbral=0.3, k=0.5)

# Entorno
env = Entorno()

show_animation = False   # pon True si quieres ver animación
EPISODES = 100
MAX_STEPS = 6000


def plot_robot(x, y, yaw, robot_radius):
    circle = plt.Circle((x, y), robot_radius, color="blue", alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    out_x = x + robot_radius * np.cos(yaw)
    out_y = y + robot_radius * np.sin(yaw)
    plt.arrow(x, y, out_x - x, out_y - y,
              head_width=0.1, head_length=0.2)
    plt.axis("equal")
    plt.grid(True)


def run_episode(goal=[20, 15]):

    env.reset()

    step = 0
    trajectory = np.array(env.x)
    min_dist_obs = float("inf")
    vel_history = []
    reached_goal = False

    while True:

        estado_ext = env.get_state(goal)

        estado_ext_norm = np.array([
            estado_ext[0] / 20.0,
            estado_ext[1] / 20.0,
            estado_ext[2] / np.pi,
            estado_ext[3] / 1.0,
            estado_ext[4] / 3.14,
            estado_ext[5] / 20.0,
            estado_ext[6] / 20.0,
        ] + estado_ext[7:])

        a_p = agent.get_action(estado_ext_norm)
        #a_r = control(estado_ext)  # DWA
        a_r = control.plan(estado_ext)  # TEB/APF
        a_h, alfa, H = combiner.combine(np.array(a_p), np.array(a_r))

<<<<<<< HEAD
        #next_state = env.motion([a_r[0], a_r[1]])  # Para entrenamiento individual
=======
>>>>>>> 9a97feb2250f730d780612d4826361557ed292d3
        next_state = env.motion([a_h[0], a_h[1]])
        trajectory = np.vstack((trajectory, env.x))

        # Distancia a obstáculos
        dists = np.linalg.norm(env.config.ob - env.x[:2], axis=1)
        dists = dists - env.config.robot_radius
        min_dist_obs = min(min_dist_obs, np.min(dists))

        vel_history.append([a_h[0], a_h[1]])

        env.update_obstacles(step)

        reward = agent.compute_reward(
            estado_ext, next_state, goal, env.config.ob
        )

        done = float(env.distance_to_goal(goal)
                     <= env.config.robot_radius)

        agent.buffer.add(estado_ext, a_p,
                         reward, next_state, done)
        agent.train_step()

        if show_animation:
            plt.cla()
            plt.plot(env.x[0], env.x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(env.config.ob[:, 0],
                     env.config.ob[:, 1], "ok")
            plot_robot(env.x[0], env.x[1],
                       env.x[2],
                       env.config.robot_radius)
            plt.pause(0.0001)

        step += 1

        if done:
            reached_goal = True
            break

        if step >= MAX_STEPS:
            break

    # --------- Métricas ---------

    diffs = np.diff(trajectory[:, :2], axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=1))

    vel_history = np.array(vel_history)

    if len(vel_history) > 1:
        delta_v = np.diff(vel_history, axis=0)
        smooth_v = np.var(delta_v[:, 0])
        smooth_w = np.var(delta_v[:, 1])
    else:
        smooth_v = 0
        smooth_w = 0

    return step, path_length, min_dist_obs, smooth_v, smooth_w, reached_goal


def main():

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Resumen"

    sheet.append([
        "Epoca",
        "Pasos",
        "Longitud trayectoria (m)",
        "Distancia mínima obstáculos (m)",
        "Suavidad lineal (var)",
        "Suavidad angular (var)",
        "Llegó a la meta"
    ])

    for epoca in range(1, EPISODES + 1):

        print(f"Epoca {epoca}/{EPISODES}")

        pasos, L, dmin, sv, sw, goal = run_episode()

        sheet.append([
            epoca,
            pasos,
            L,
            dmin,
            sv,
            sw,
            int(goal)
        ])

    workbook.save("datos_robot.xlsx")

    save_path = os.path.join("pesos", save_name)
    agent.save(save_path)

    print("Entrenamiento completo. Datos guardados.")


if __name__ == '__main__':
    main()
