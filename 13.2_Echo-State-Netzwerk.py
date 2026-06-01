import numpy as np
import matplotlib.pyplot as plt

m_z=12345 # Zufallsinitialisierung (seed)
m_w=34645 # Zufallsinitialisierung (seed)

def rndNum(): # Eigener Pseudozufallsgenerator
    global m_z, m_w
    m_z = 36969 * (m_z & 65535) + (m_z >> 16)
    m_w = 18000 * (m_w & 65535) + (m_w >> 16)
    return (((int)(m_z << 16) + m_w)%100000)/100000  

def mackey_glass(tau=17, n=10, beta=0.2, gamma=0.1, t_max=3000, dt=1):
    # Generiere Mackey-Glass-Zeitreihen mit Euler
    t = np.arange(0, t_max+dt, dt)
    x = np.zeros(len(t))
    x[0:tau] = 0.5  # Anfangsbedingung
    for i in range(tau, len(t)):
        x[i] = x[i-1] + dt * (beta*x[i-tau]/(1 + x[i-tau]**n) - gamma*x[i-1])
    return x

# Daten generieren
trainLen, testLen, initLen = 1500, 1500, 100
data = mackey_glass(tau=17, t_max=trainLen+testLen)

# ESN-Reservoir generieren
res_size = 300
W_in = np.zeros((res_size,2)) # Eingangsgewichte
W = np.zeros((res_size,res_size)) # Reservoirgewichte
for i in range(res_size):
    W_in[i][0] = rndNum()-0.5
    W_in[i][1] = rndNum()-0.5
    for j in range(res_size):
        if rndNum()<0.05: # 5% Verbindungen
            W[i][j] = rndNum()-0.5

# Gewichtsmatrix auf Spektralradius von 1.2 skalieren
W *= 1.2 / max(abs(np.linalg.eig(W)[0])) 

X = np.zeros((1+res_size,trainLen-initLen)) # Reservoir-Werte
Yt = data[initLen+1:trainLen+1] # Target-Werte

# ESN trainieren
x = np.zeros(res_size)
for t in range(trainLen):
    u = data[t]
    x = np.tanh(np.dot(W_in, [1,u])+np.dot(W, x))
    if t >= initLen:
        X[:,t-initLen] = np.concatenate(([1],x))

reg = 1e-7  # Regularisierungskoeffizient
W_out = np.linalg.solve(np.dot(X,X.T) + reg*np.eye(1+res_size), np.dot(X,Yt)).T

# Mackey-Glass-Attraktor vorhersagen
Y = np.zeros(testLen)
u = data[trainLen]
for t in range(testLen):
    x = np.tanh(np.dot(W_in, [1,u])+np.dot(W, x))
    y = np.dot( W_out, np.concatenate(([1],x)) )
    Y[t] = y
    u = y

mse = np.mean((data[trainLen+1:trainLen+501]-Y[:500])**2)

print('MSE = ' + str( mse ))

# Attraktoren plotten
plt.figure(figsize=(12,6))
plt.title("Mackey-Glass-Attraktor")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.plot(data[trainLen+1:trainLen+1001], 'g', label="Originale Daten")
plt.plot(Y[:1000], '--r', label="Generierte Daten")
plt.legend(loc='upper right')
plt.show()