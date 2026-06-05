# Aufgabe 4
# Implementieren Sie den zweidimensionalen Pooling Layer eines
# Convolutional Networks. Laden Sie dazu ein Schwarz-Weiß Bild
# (z.B. im PGM Format) und erzeugen Sie ein kleineres Bild indem
# Sie die 2x2 Pooling Maske mit Stride 2 (heißt in Schrittweite 2)
# über das Bild schieben.

import numpy as np
import pygame

xmax = 400
ymax = 200
pygame.init()
screen = pygame.display.set_mode((xmax, ymax))

def main():

    test_data = np.loadtxt("SL/Aufgabe4/mnist.csv", delimiter=",") 
    test_data[:10]
    test_data[test_data==255]
    test_data.shape

    pattern = 0
    endlos = True
    while endlos:
            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                            endlos = False

            in_vec = np.zeros(28*28)

            # --- draw input activation ---
            ii=0
            screen.fill((0, 0, 0))
            for y in range(0,28):
                for x in range(0,28):
                    in_vec[ii] = val = test_data[pattern,ii]/255.0              
                    ii=ii+1
                    pygame.draw.rect(screen, (255*val, 0, 0), pygame.Rect(x*5, y*5, 5, 5))
 
           # --- calculate maxPooling 2x2 -> 1x1
            ii = 0
            maxPool = np.zeros(14*14)

            for y in range(0, 28, 2):
                for x in range(0, 28, 2):
                    maxPool[ii] = np.max([
                    in_vec[y*28 + x],
                    in_vec[y*28 + x + 1],
                    in_vec[(y+1)*28 + x],
                    in_vec[(y+1)*28 + x + 1]
                    ])
                    ii += 1

            # --- draw maxPoolingLayer ---
            jj=0
            for y in range(0,14):
                for x in range(0,14):
                    val = maxPool[jj]             
                    jj=jj+1
                    pygame.draw.rect(screen, (0,255*val, 0), pygame.Rect(200+x*5, (y+10)*5, 5, 5))

            pygame.time.delay(1000)    # may be reduced in training time     
            pattern=pattern+1
            pygame.display.flip()

# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
main()