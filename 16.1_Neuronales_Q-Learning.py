import pygame as pyg
import numpy as np
import random

def one_hot_state(input):
    # gibt den Zustand als one Hot Vektor zurück
    in_vector = np.zeros(state_dim)
    in_vector[int(input)] = 1 # one hot vector  
    return in_vector  

def model_predict(in_vec):
    # gibt die Aktivität beider Neuronen zurück
    return np.matmul(weights.T, in_vec) 

def model_fit(in_vec, target_vec):
    global weights
    out_vec = model_predict(in_vec)
    # Gewichtsmatrix mit Delta-Lernregel anpassen
    weights += np.outer(in_vec.T,(target_vec-out_vec)) 

# die Q-Updates machen
def updateQ(reward, state, action, nextState):
    global replay_re, replay_s, replay_a, replay_ns

    # Experience Replay Ringbuffer füllen
    replay_re[tick % 400] = reward
    replay_s[ tick % 400] = state
    replay_a[ tick % 400] = action
    replay_ns[tick % 400] = nextState
    if tick>400:
        for i in range(batch_size):
            r = random.randint(0,399) 
            Qval = model_predict(one_hot_state(replay_s[r]))
            target = np.zeros(2) # target mit Q-updateformel definieren
            target[int(replay_a[r])] = replay_re[r] + alpha*(gamma * np.max(model_predict(one_hot_state(replay_ns[r]))) - Qval[int(replay_a[r])])
            model_fit(one_hot_state(replay_s[r]), np.array(target))

# Nächste Aktion mit epsilon-Greedy
def getAction(state):
    if np.random.rand() <= epsilon:
        return np.random.choice([-1, 1])
    act_values = model_predict(one_hot_state(state))
    return (np.argmax(act_values) * 2) - 1

# Reduziert den Zustand auf eine Zahl
def getState(x_ball, y_ball, vx_ball, vy_ball, x_racket):
    return (((x_ball*13 +y_ball)*2 +(vx_ball+1)/2)*2 +(vy_ball+1)/2)*12 +x_racket

# Q-Network Parameter
state_dim  = 12*13*2*2*12
action_dim = 2
epsilon = 1
alpha = 0.1
gamma = 0.95
batch_size = 32
weights = np.random.rand(state_dim, action_dim)/1000.0
episode = 0
tick = 0
replay_re  = np.zeros(400, dtype=int)
replay_s   = np.zeros(400, dtype=int)
replay_a   = np.zeros(400, dtype=int)
replay_ns  = np.zeros(400, dtype=int)

pyg.init()
screen = pyg.display.set_mode((240, 260))
pyg.display.set_caption("Neural-Pong")
file = open('reward_neural.txt','w')
x_racket, x_ball, y_ball, vx_ball, vy_ball, score = 5, 1, 1, 1, 1, 0
#clock = pyg.time.Clock()
cont = True
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

    # berechne neuen State und neue Schlägerposition
    x_racket = x_racket + action
    if x_racket>11: x_racket = 11
    if x_racket<0:  x_racket = 0

    x_ball, y_ball = x_ball + vx_ball, y_ball + vy_ball
    if x_ball > 10 or x_ball < 1: vx_ball *= -1
    if y_ball > 11 or y_ball < 1: vy_ball *= -1

    reward = 0
    if y_ball == 12:
        reward = -1
        if x_ball >= x_racket and x_ball <= x_racket + 4:
            reward = +1
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