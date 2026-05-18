# main_inference.py

import numpy as np
import matplotlib.pyplot as plt
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


MODEL_NAME = "SAC+DWA"
MODEL_PATH = os.path.join("pesos", MODEL_NAME)

SHOW_ANIMATION = True

MAX_STEPS = 6000

GOAL = [20, 15]

# Cada cuántos pasos marcar posiciones
MARK_EVERY = 2000


agent = SAC(state_dim=2 + 1 + 2 + 2 + 20, action_dim=2, max_action=[1.0, 3.14])
# agent = TD3(state_dim=2 + 1 + 2 + 2 + 20, action_dim=2, max_action=[1.0, 3.14])

agent.load(MODEL_PATH)

print(f"Pesos cargados desde {MODEL_PATH}")

control = TEBController()
#control = APFController()
#control = DWAController


combiner = EntropyCombiner(
    H_umbral=0.3,
    k=0.5
)


env = Entorno()


def plot_robot(x, y, yaw, robot_radius):

    circle = plt.Circle(
        (x, y),
        robot_radius,
        color="blue",
        alpha=0.5
    )

    plt.gcf().gca().add_artist(circle)

    out_x = x + robot_radius * np.cos(yaw)
    out_y = y + robot_radius * np.sin(yaw)

    plt.arrow(
        x,
        y,
        out_x - x,
        out_y - y,
        head_width=0.1,
        head_length=0.2,
        fc='black',
        ec='black'
    )

    plt.axis("equal")
    plt.grid(True)


def plot_final_trajectories(
        robot_history,
        obstacles_history,
        goal,
        mark_every=50):

    plt.figure(figsize=(12, 10))

    robot_history = np.array(robot_history)


    plt.plot(
        robot_history[:, 0],
        robot_history[:, 1],
        color='blue',
        linewidth=2.5,
        label='Agent Trajectory'
    )

    # Marcas cada n pasos
    for i in range(0, len(robot_history), mark_every):

        plt.scatter(
            robot_history[i, 0],
            robot_history[i, 1],
            color='blue',
            s=40
        )

        # Mostrar número de paso
        plt.text(
            robot_history[i, 0],
            robot_history[i, 1],
            str(i),
            fontsize=8,
            color='blue'
        )

        # Flecha de dirección
        if i > 0:

            dx = robot_history[i, 0] - robot_history[i - 1, 0]
            dy = robot_history[i, 1] - robot_history[i - 1, 1]

            plt.arrow(
                robot_history[i - 1, 0],
                robot_history[i - 1, 1],
                dx,
                dy,
                color='blue',
                head_width=0.12,
                length_includes_head=True
            )


    for j, obs_hist in enumerate(obstacles_history):

        obs_hist = np.array(obs_hist)

        plt.plot(
            obs_hist[:, 0],
            obs_hist[:, 1],
            '--',
            color='black',
            linewidth=1.5,
            label=f'Obstacle {j}'
        )

        # Posiciones cada n pasos
        for i in range(0, len(obs_hist), mark_every):

            plt.scatter(
                obs_hist[i, 0],
                obs_hist[i, 1],
                color='black',
                s=25
            )

            plt.text(
                obs_hist[i, 0],
                obs_hist[i, 1],
                str(i),
                fontsize=8,
                color='black'
            )


    plt.scatter(
        goal[0],
        goal[1],
        color='gold',
        marker='*',
        s=300,
        label='Goal'
    )


    plt.scatter(
        robot_history[0, 0],
        robot_history[0, 1],
        color='cyan',
        s=180,
        marker='o',
        label='Start'
    )


    plt.title(
        "Agent's Path and Obstacles",
        fontsize=16
    )

    plt.xlabel("Position X")
    plt.ylabel("Position Y")

    plt.axis("equal")
    plt.grid(True)

    plt.legend()

    # Guardar imagen
    plt.savefig(
        "trayectorias_finales.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()



def normalize_state(estado_ext):

    estado_ext_norm = np.array([

        estado_ext[0] / 20.0,
        estado_ext[1] / 20.0,
        estado_ext[2] / np.pi,

        estado_ext[3] / 1.0,
        estado_ext[4] / 3.14,

        estado_ext[5] / 20.0,
        estado_ext[6] / 20.0,

    ] + estado_ext[7:])

    return estado_ext_norm


def run_inference(goal=GOAL):

    env.reset()

    trajectory = np.array(env.x)

    step = 0

    reached_goal = False

    print("Iniciando inferencia...\n")


    robot_history = []

    obstacles_history = [
        [] for _ in range(len(env.config.ob))
    ]

    start_time = time.time()
    frames = []

    while True:


        estado_ext = env.get_state(goal)
        estado_norm = normalize_state(estado_ext)
        a_p = agent.get_action(estado_norm)

      
        a_r = control.plan(estado_ext)  # Para APF / TEB
        #a_r = control(estado_ext)  # Para DWA

        a_h, alfa, H = combiner.combine(
            np.array(a_p),
            np.array(a_r)
        )


        env.motion([a_h[0], a_h[1]])
        trajectory = np.vstack((trajectory, env.x))
        robot_history.append(
            (env.x[0], env.x[1])
        )

        for i, obs in enumerate(env.config.ob):

            obstacles_history[i].append(
                (obs[0], obs[1])
            )

        env.update_obstacles(step)

        dist_goal = env.distance_to_goal(goal)

        print(f"Step: {step} ")


        if SHOW_ANIMATION:

            plt.cla()

            # Robot
            plt.plot(
                env.x[0],
                env.x[1],
                "ob"
            )

            # Meta
            plt.plot(
                goal[0],
                goal[1],
                "*y",
                markersize=15
            )

            # Obstáculos
            plt.plot(
                env.config.ob[:, 0],
                env.config.ob[:, 1],
                "ok"
            )

            # Dibujar robot
            plot_robot(
                env.x[0],
                env.x[1],
                env.x[2],
                env.config.robot_radius
            )

            # Mostrar información
            plt.title(
                f"Step: {step} | "
                f"Alpha: {alfa:.3f} | "
                f"H: {H:.3f}"
            )

            plt.draw()
            plt.pause(0.0001)
            fig = plt.gcf()
            fig.canvas.draw()
            image = np.frombuffer(
                fig.canvas.tostring_rgb(),
                dtype='uint8'
            )

            image = image.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )

            frames.append(image)

        if dist_goal <= env.config.robot_radius:

            reached_goal = True

            print("\nMeta alcanzada.")

            break

        if step >= MAX_STEPS:

            print("\nMáximo número de pasos alcanzado.")

            break

        step += 1

    total_time = time.time() - start_time

    print(f"\nTiempo total: {total_time:.2f} segundos")

    if len(frames) > 0:

        imageio.mimsave(
            "simulacion.gif",
            frames,
            fps=30
        )

        print("Simulación guardada como simulacion.gif")

    plot_final_trajectories(
        robot_history,
        obstacles_history,
        goal,
        mark_every=MARK_EVERY
    )

    print(
        "Imagen final guardada como "
        "trayectorias_finales.png"
    )

    return trajectory, reached_goal


def main():

    trajectory, reached = run_inference()

    if reached:
        print("Resultado: ÉXITO")
    else:
        print("Resultado: FALLÓ")



if __name__ == "__main__":

    main()
