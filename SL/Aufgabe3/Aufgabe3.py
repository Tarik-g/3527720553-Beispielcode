#Implementieren Sie eine Restricted Bolzman Machine als einfachste
#Form eines Autoencoders. Am Eingang sollen die MNIST Ziffern
#angelegt werden. Durch Aktivierung der Hidden Schicht und
#Rückaktivierung sollten die MNIST Ziffern wieder rekonstruiert
#werden können. Nun reduzieren Sie die Anzahl der Hidden
#Neuronen auf 100. Können die Ziffern trotzdem so rekonstruiert
#werden, dass man sie noch erkennen kann?

import numpy as np
import pygame

# --------------------------------------------------------------------------------------------------
# Screen Setup
# --------------------------------------------------------------------------------------------------
xmax = 600
ymax = 200
pygame.init()
screen = pygame.display.set_mode((xmax, ymax))

# --------------------------------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sample(p):
    return (np.random.rand(*p.shape) < p).astype(float)

# --------------------------------------------------------------------------------------------------
# RBM KLASSE
# --------------------------------------------------------------------------------------------------
class RBM:
    
    def __init__(self, visible=28*28, hidden=100):
        self.visible = visible
        self.hidden = hidden

        self.W = np.random.normal(0, 0.01, size=(hidden, visible))
        self.v_bias = np.zeros(visible)
        self.h_bias = np.zeros(hidden)

    def hidden_prob(self, v):
        return sigmoid(self.W @ v + self.h_bias)

    def visible_prob(self, h):
        return sigmoid(self.W.T @ h + self.v_bias)

    def contrastive_divergence(self, v0, lr=0.1):
        # positive phase
        h0_p = self.hidden_prob(v0)
        h0 = sample(h0_p)

        # reconstruction
        v1_p = self.visible_prob(h0)
        v1 = sample(v1_p)

        # negative phase
        h1_p = self.hidden_prob(v1)

        # update weights
        self.W += lr * (np.outer(h0_p, v0) - np.outer(h1_p, v1))
        self.v_bias += lr * (v0 - v1)
        self.h_bias += lr * (h0_p - h1_p)

        return h0_p, v1_p
        
    def train(self, v0, lr=0.1):
        return self.contrastive_divergence(v0, lr)

def main():
    test_data = np.loadtxt("SL/Aufgabe3/mnist.csv", delimiter=",")
    
    # RBM
    RBMaschine = RBM()

    pattern = 0
    endlos = True

    while endlos:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                endlos = False

        # -----------------------------
        # INPUT + INPUT VISUALISIERUNG
        # -----------------------------
        screen.fill((0, 0, 0))

        ii = 0
        in_vec = np.zeros(28*28)

        # MNIST pixels
        for y in range(28):
            for x in range(28):

                val = test_data[pattern, ii] / 255.0
                in_vec[ii] = val
                ii += 1

                pygame.draw.rect(
                    screen,
                    (int(255 * val), 0, 0),
                    pygame.Rect(x * 5, y * 5, 5, 5)
                )

        # -----------------------------
        # TRAIN / FORWARD PASS
        # -----------------------------
        hid_vec, out_vec = RBMaschine.train(in_vec,0.1)

        # -----------------------------
        # HIDDEN VISUALISIERUNG (10x10)
        # -----------------------------
        jj = 0
        for y in range(10):
            for x in range(10):

                val = hid_vec[jj]
                jj += 1

                pygame.draw.rect(
                    screen,
                    (0, int(255 * val), 0),
                    pygame.Rect(200 + x * 10, y * 10, 10, 10)
                )

        # -----------------------------
        # OUTPUT VISUALISIERUNG
        # -----------------------------
        kk = 0

        for y in range(28):
            for x in range(28):

                val = out_vec[kk]
                kk += 1

                pygame.draw.rect(
                    screen,
                    (0, 0, int(255 * val)),
                    pygame.Rect(400 + x * 5, y * 5, 5, 5)
                )

        # -----------------------------
        # LOOP CONTROL
        # -----------------------------
        if pattern % 100 == 0:
            pygame.time.delay(500)

        pattern = (pattern + 1) % len(test_data)
        pygame.display.flip()

main()