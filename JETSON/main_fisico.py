# main_fisico.py

import numpy as np
import os
import time

from RL.SAC import SAC
from RL.TD3 import TD3
from controllers.controller_dwa import DWAController
from controllers.controller_apf import APFController
from controllers.controller_teb import TEBController
from entropy_combiner import EntropyCombiner
from serial_comm import SerialComm
from obstacle_detector import ObstacleDetector
from utils import local_to_global_latlon


# Comunicación serial
serial = SerialComm()

# Nombre de pesos a cargar
load_name = "TD3+APF"

# Inicializar agente (misma config que en entrenamiento)
# agent = SAC(state_dim=2+1+2+2+20, action_dim=2, max_action=[1.0, 3.14])
agent = TD3(state_dim=2+1+2+2+20, action_dim=2, max_action=[1.0, 3.14])

# Cargar pesos
load_path = os.path.join("pesos", load_name)
agent.load(load_path)
print(f"Pesos cargados desde {load_path}")

# Inicializar controlador
#control = DWAController
control = APFController()
# control = TEBController()

# Inicializar fusión híbrida
combiner = EntropyCombiner(H_umbral=0.3, k=0.5)

# Inicializar detector de obstáculos
detector = ObstacleDetector()


def main():
    # ---------------- Valores iniciales de estado ----------------
    # Espera hasta recibir datos del TTGO_J
    entrada = None
    while entrada is None:
        entrada = serial.read_json()
        time.sleep(0.1)
    print(entrada)
    # Inicializar variables
    q = entrada["q"]             # [lat, lon]
    teta = [entrada["teta"]] # [yaw]
    a_h = entrada["a_h"]         # [v, w]
    q_meta = entrada["q_meta"]  # [lat_meta, lon_meta]


    # #entrada = [20, 15, 35, 0, 0, 35, 56]  # [x, y, theta, v_prev, w_prev, goal_x, goal_y]

    # print(entrada)
    # q = [entrada[0], entrada[1]]          # posición actual
    # teta = [entrada[2]]       # orientación en radianes
    # a_h = [entrada[3],entrada[4]]        # acción previa (v, w)
    # q_meta = [entrada[5], entrada[6]]     # meta (x, y)

    while True:
       
        # Detectar obstáculos
        obstacles = detector.get_obstacles()  # lista [(X, Z), (X, Z), ...]
        obstacles = local_to_global_latlon(q[0], q[1], teta[0], obstacles) #devuelve una lista de (lat, lon) de cada obstáculo

        # Convertir a lista plana [x1, y1, x2, y2, ...]
        q_obst = []
        for ox, oy in obstacles:
            q_obst += [ox, oy]

        # Rellenar hasta 20 valores (10 obstáculos máx)
        if len(q_obst) < 20:
            q_obst += [0.0] * (20 - len(q_obst))
        else:
            q_obst = q_obst[:20]
        
        # Estado completo
        estado_ext = q + teta + [a_h[0], a_h[1]] + q_meta + q_obst

        # Normalizar estado
        estado_ext_norm = np.array([
            estado_ext[0] / 20.0,   # x
            estado_ext[1] / 20.0,   # y
            estado_ext[2] / np.pi,  # theta
            estado_ext[3] / 1.0,    # v_prev
            estado_ext[4] / 3.14,   # w_prev
            estado_ext[5] / 20.0,   # goal x
            estado_ext[6] / 20.0,   # goal y
        ] + estado_ext[7:])

        # Acción de RL
        a_p = agent.get_action(estado_ext_norm)

        # Acción del controlador
        #a_r = control(estado_ext)      # DWA
        a_r = control.plan(estado_ext)  # APF/TEB

        # Fusión híbrida
        a_h, alfa, H = combiner.combine(np.array(a_p), np.array(a_r))
        
        # Enviar al robot
        serial.send_json([a_h[0], a_h[1]])
        print(f"Acción híbrida: {a_h}, alfa={alfa:.2f}, H={H:.2f}")


if __name__ == '__main__':
    main()
