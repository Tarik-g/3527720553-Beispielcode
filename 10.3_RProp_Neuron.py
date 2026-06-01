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
weight   = np.random.rand(3)*(0.5)
delta    = np.linspace(0.125,0.125,3)
grad_old = np.zeros(3)
grad_new = np.zeros(3) 
eta_plus = 1.2  # Faktor zur Vergrößerung der Lernrate
eta_minus = 0.5 # Faktor zur Verkleinerung der Lernrate
delta_max = 50 # Maximale Gewichtsänderung
delta_min = 0  # Minimale Gewichtsänderung

def sigmoid(summe): # Transferfunktion
    return 1.0/(1.0+np.exp(-1.0*summe)) 

def learn():
	global train, weight, out, target, grad_old, grad_new, delta
	# Neuronenausgabe berechnen
	out = sigmoid(np.matmul(train, weight))
	# Gradienten berechnen
	grad_old = np.copy(grad_new)
	grad_new = np.matmul(train.T,(out-target))
	########### iRprop- #############
	for i in range(0,3):
		if grad_old[i]*grad_new[i]>0: # Lernrate vergrößern
			delta[i] = min(delta[i]*eta_plus, delta_max)
		if grad_old[i]*grad_new[i]<0: # Lernrate verkleinern
			delta[i] = max(delta[i]*eta_minus, delta_min)
			grad_new[i] = 0 # Einziger Unterschied zu Rprop
	weight -= delta*np.sign(grad_new) # Gewichte anpassen

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
	learn()   # lerne einen Schritt iRprop-
	plt.clf() # Bildschirm löschen
	X, Y, Z = outp() # generiere Plotdaten
	ax = fig.add_subplot(111, projection='3d')
	# 3D plot von den Daten
	ax.plot_surface(X, Y, Z, edgecolor='royalblue', 
		lw=0.5, rstride=8, cstride=8, alpha=0.3)
	ax.set_title('Neuron lernt AND-Funktion mit iRProp-')
	ax.set_xlabel('In[1]')
	ax.set_ylabel('In[2]')
	ax.set_zlabel('Ausgabe\ndes Neurons')
	ax.set_zlim(0, 1)
	plt.draw()
	plt.pause(0.00001)
