import matplotlib.pyplot as plt
import numpy as np
import math

k = 1
# Spiralen erzeugen
spiral1 = np.zeros((97,2))
spiral2 = np.zeros((97,2))
proto   = np.zeros((2*97,2))
for i in range(97):
    phi = i/16 * math.pi
    r = (104 - i)/208
    # Klasse A (Array Index gerade)
    proto[2*i][0]=spiral1[i][0]=(r*math.cos(phi))+0.5
    proto[2*i][1]=spiral1[i][1]=(r*math.sin(phi))+0.5
    # Klasse B (Array Index ungerade)
    proto[2*i+1][0]=spiral2[i][0]=(-r*math.cos(phi))+0.5
    proto[2*i+1][1]=spiral2[i][1]=(-r*math.sin(phi))+0.5

test_pts = np.zeros((10000,2))
i=0
for y in np.arange(0.0 ,1.0 ,0.01):
    for x in np.arange(0.0 ,1.0 ,0.01):
        test_pts[i][0] = x
        test_pts[i][1] = y
        i+=1

knnA_x = np.array([])
knnA_y = np.array([])
knnB_x = np.array([])
knnB_y = np.array([])

for i, point in enumerate(test_pts):
    count = 0
    distances=np.linalg.norm(proto[:,0:2]-point, axis=1)
    for j in range(k):
        bestIndex = np.argmin(distances)
        distances[bestIndex] = np.max(distances)
        if (bestIndex%2)==0: # Klasse A, Index gerade
            count+=1

    if count>k/2: # Mehrheitsentscheidung
        knnA_x = np.append(knnA_x, test_pts[i][0])
        knnA_y = np.append(knnA_y, test_pts[i][1])
    else:
        knnB_x = np.append(knnB_x, test_pts[i][0])
        knnB_y = np.append(knnB_y, test_pts[i][1])

plt.scatter(knnA_x, knnA_y, c=[0.7,0.7,0.7], marker='.')
plt.scatter(knnB_x, knnB_y, c=[0.3,0.3,0.3], marker='.')
plt.scatter(spiral1[:,0], spiral1[:,1], c=[[1.,1.,1.]], marker='x')
plt.scatter(spiral2[:,0], spiral2[:,1], c=[[0.,0.,0.]], marker='x')
plt.title('K-Nächste-Nachbarn-Algorithmus mit k='+str(k))
plt.xlabel('x-Achse')
plt.ylabel('y-Achse')
plt.show()

