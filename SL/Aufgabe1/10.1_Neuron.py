import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#                 BIAS,x,y
train  = np.array( [[1,0,0],
                    [1,1,0],
                    [1,0,1],
                    [1,1,1]])
target = np.array([0,0,0,1]) # AND Operation
out    = np.array([0,0,0,0])
weight = np.random.rand(3)*(0.5)
learnrate  = 1.0
grad   = np.zeros(3)

def sigmoid(summe): # Transferfunktion
    return 1.0/(1.0+np.exp(-1.0*summe)) 

def learn():
    global train, weight, out, target, learnrate
    # Neuronenausgabe für alle 4 Trainingsmuster berechnen
    out = sigmoid(np.matmul(train, weight))
    # Gradienten berechnen
    grad = np.matmul(train.T,(out-target)) * (out.T.dot(np.subtract(np.ones(4),out)))
    weight -= learnrate*grad # Gewichte anpassen

def outp(N=100): # Daten für die Ausgabefunktion generieren
    global weight
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)
    oo = sigmoid(weight[0] + weight[1]*xx + weight[2]*yy)
    return xx, yy, oo

def on_close(event): # Fenster schließen
    exit(0)

plt.ion()
fig = plt.figure()
fig.canvas.mpl_connect('close_event', on_close)
while True:   # Endlosschleife
    #for i in range(1000):
    learn()   # lerne einen Schritt
    plt.clf() # Bildschirm löschen
    X, Y, Z = outp() # generiere Plotdaten
    ax = fig.add_subplot(111, projection='3d')
    # 3D plot von den Daten
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', 
        lw=0.5, rstride=8, cstride=8, alpha=0.3)
    ax.set_title('Neuron lernt AND-Funktion')
    ax.set_xlabel('In[1]')
    ax.set_ylabel('In[2]')
    ax.set_zlabel('Ausgabe\ndes Neurons')
    ax.set_zlim(0, 1)
    plt.draw()
    plt.pause(0.00001)
