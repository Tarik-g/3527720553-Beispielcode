import pygame as pyg
import numpy as np
import random

#   die Q-Updates machen
def updateQ(reward, state, action, nextState):
    global er_re, er_s, er_a, er_ns, tick, Q, alpha, gamma
    # Replay-Buffer füllen
    er_re[tick%400]= reward   # experience replay Belohnung
    er_s[tick%400] = state    # experience replay Zustand
    er_a[tick%400] = action   # experience replay Aktion
    er_ns[tick%400]= nextState# experience replay nächster Zustand

    for i in range(batch_size):
        r = random.randint(0,399) 
        # Q[s][a]+=r+alpha*(gamma * max_a' Q(s',a')-Q(s,a)) 
        Q[int(er_s[r])][int(er_a[r])] += er_re[r] + alpha*(gamma * np.max(Q[int(er_ns[r])]) - Q[int(er_s[r])][int(er_a[r])])

#   Nächste Aktion anfragen
def getAction(state): # gibt -1 für Schläger links oder +1 für rechts zurück
    global epsilon, Q
    if np.random.rand() <= epsilon:
        return np.random.choice([-1, 1])
    return (np.argmax(Q[int(state)]) * 2) - 1

#   Macht aus 5 Koordinaten -> 1 Koordinate
#   damit bekommt jeder Zustand eine eindeutige Nummer
def getState(x_ball, y_ball, vx_ball, vy_ball, x_racket):
    return (((x_ball*13 +y_ball)*2 +(vx_ball+1)/2)*2 +(vy_ball+1)/2)*12 +x_racket 

# Parameter für Q-Learning und Experience Replay
epsilon = 1
alpha = 0.1
gamma = 0.95
batch_size = 32
er_re = np.zeros(400) 
er_s  = np.zeros(400) 
er_a  = np.zeros(400) 
er_ns = np.zeros(400) 

tick = 0    # zählt bei jeder Zustandsändrung hoch
episode = 0 # zählt die Episoden

num_of_states = 13*12*2*2*12
num_of_actions = 2
Q = np.random.rand(num_of_states, num_of_actions)/1000.0

pyg.init()
screen = pyg.display.set_mode((240, 260))
pyg.display.set_caption("Q-Learning Experience-Replay")
file = open('reward_experience_replay.txt','w')
x_racket, x_ball, y_ball, vx_ball, vy_ball, score = 5, 1, 1, 1, 1, 0

cont = True
#clock = pyg.time.Clock()
while cont:
    for event in pyg.event.get(): 
        if event.type == pyg.QUIT:
              cont = False   

    epsilon -= 1/400
    if (epsilon<0):
        epsilon=0
    screen.fill((0,0,0))
    font = pyg.font.SysFont("arial", 15)
    t = font.render("Score:"+str(score)+" Episode:"+str(episode), True, (255,255,255))
    screen.blit(t, t.get_rect(centerx = screen.get_rect().centerx))
    pyg.draw.rect(screen, (0, 128, 255), pyg.Rect(x_racket*20, 250, 80, 10))
    pyg.draw.rect(screen, (255, 100, 0), pyg.Rect(x_ball*20, y_ball*20, 20, 20))

    state = getState(x_ball, y_ball, vx_ball, vy_ball, x_racket)
    action = getAction(state)

    x_racket = x_racket + action # Aktion ausführen
    if x_racket>11: x_racket = 11
    if x_racket<0:  x_racket = 0

    x_ball, y_ball = x_ball + vx_ball, y_ball + vy_ball
    if x_ball > 10 or x_ball < 1: vx_ball *= -1
    if y_ball > 11 or y_ball < 1: vy_ball *= -1

    reward = 0
    if y_ball == 12:
        reward = -1 # Annahme: Ball daneben
        if x_ball >= x_racket and x_ball <= x_racket + 4:
            reward = +1 # Ball doch getroffen
        episode += 1
        score = score + reward

    nextState = getState(x_ball, y_ball, vx_ball, vy_ball, x_racket)
    updateQ(reward, state, (action+1)//2, nextState)

    tick += 1
    if reward!=0:
        file.write(str(reward)+",")
        file.flush()
    #clock.tick(60) # Refresh-Zeiten festlegen 60 FPS
    pyg.display.flip()