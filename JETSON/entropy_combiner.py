# entropy_combiner.py

import numpy as np

class EntropyCombiner:
    def __init__(self, H_umbral=0.3, k=0.5):
        self.H_umbral = H_umbral
        self.k = k

    def softmax(self, x): # Convierte acciones continuas en probabilidades válidas.
        exp_x = np.exp(x - np.max(x))  # evitar overflow numérico
        return exp_x / (np.sum(exp_x) + 1e-8)

    def compute_entropy(self, ap):  # Calcula la entropía a partir de las acciones predichas.
        probs = self.softmax(ap)
        return -np.sum(probs * np.log(probs + 1e-8))

    def compute_alpha(self, H): # Calcula el factor de transición α.
        return 1.0 / (1.0 + np.exp(-self.k * (self.H_umbral - H)))

    def combine(self, ap, ar):  # Combina las acciones predichas y reales en híbridas.
        H = self.compute_entropy(ap)
        alpha = self.compute_alpha(H)

        vh = alpha * ap[0] + (1 - alpha) * ar[0]
        wh = alpha * ap[1] + (1 - alpha) * ar[1]

        return np.array([vh, wh]), alpha, H
