import pygame
import numpy as np

pygame.init()
screen = pygame.display.set_mode((200, 200))
pygame.display.set_caption("Reinforcement Learning")
clock = pygame.time.Clock()

maze=["##########",
      "#        #",
      "#        #",
      "####    ##",
      "#        #",
      "#        #",
      "#    #####",
      "#        #",
      "#       G#",
      "##########"]

x_dir = [-1,1,0,0] # x-Richtung für Aktion 0,1,2,3
y_dir = [0,0,-1,1] # y-Richtung für Aktion 0,1,2,3
q = np.random.rand(100, 4)*0.1 # q[s][a]=0..0.1, q[100][4]
alpha = 0.5 # Lernrate
gamma = 0.9 # Discount Faktor
epsilon = 50 # für Epsilon-Greedy Aktionsauswahl
for episode in range(1000):
    x_agent  = 1 # x-Agent auf Start
    y_agent  = 1 # y-Agent auf Start
    goal_reached = False
    while(not goal_reached):
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                exit()
        # zeichne Labyrinth und Q-Bewertung
        screen.fill((0,0,0))
        for y in range(10):
            for x in range(10):
                if maze[y][x]=='#':
                    pygame.draw.rect(screen, (0, 128, 255), 
                      pygame.Rect(x*20, y*20, 15, 15))
                else:
                    pygame.draw.rect(screen, 
                      (0, 200*np.max(q[y*10+x]), 0), 
                      pygame.Rect(x*20, y*20, 15, 15))
        reward = 0
        s = y_agent*10+x_agent # eindimensionaler state

        if np.random.randint(100)<epsilon: # Epsilon Greedy
            a = np.random.randint(4) # action
        else:
            # argmax ergibt den Index und damit die Aktion, 
            # bei dem der q am größten ist
            a = np.argmax(q[s]) 
        # wenn keine Wand, bewege Agent
        if maze[y_agent+y_dir[a]][x_agent+x_dir[a]]!='#':
            x_agent += x_dir[a]
            y_agent += y_dir[a]
        # wenn Ziel erreicht, dann Belohnung
        if maze[y_agent][x_agent]=='G':
            goal_reached = True
            reward = 1
        # neuer eindimensionaler Zustand
        new_s = y_agent*10+x_agent
        # Q-Update Formel
        q[s][a] += alpha*(reward + gamma*np.max(q[new_s]) - q[s][a])

        pygame.draw.rect(screen, (255, 0, 0), 
          pygame.Rect(x_agent*20, y_agent*20, 15, 15))  
        pygame.display.flip()
        clock.tick(60) # 60 Frames pro Sekunde
