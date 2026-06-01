import numpy as np
import matplotlib.pyplot as plt

cities = 100

def get_length(travel, distance):
	# Entfernung vom Ende der Reise zum Anfang
	length=distance[int(travel[-1]),int(travel[0])] 
	for i in range(0, cities-1):
		# Einzelentfernungen aufadieren
		length+=distance[int(travel[i]),int(travel[i+1])]
	return length

def main():
	# Entfernungstabelle mit Zufallszahlen
	distance = np.random.rand(cities, cities)
	# Rundreise als 0,1,2,3,4...99 als Startrundreise setzen
	travel = np.linspace(0, cities, cities, endpoint=False)
	graph = np.array([])
	t = 1
	# 100000 Versuche besser zu werden
	for step in range(0,100000):
		t=t*0.9999 # die Temperatur wird verkleinert
		length = get_length(travel, distance)
		i = np.random.randint(0,cities-1)
		j = np.random.randint(0,cities-1)
		# vertausche Stadt i mit Stadt j
		travel[i], travel[j] = travel[j], travel[i]
		
		new_length = get_length(travel, distance)
		# Wahrscheinlichkeit für einen Rückschritt
		# Achtung: Statt Fitness wird die Rundreiselänge 
		# genommen. Deshalb das negative Vorzeichen...
		p = np.exp(-(new_length-length)/t)
		# Wenn die neue Rundreise kürzer ist oder
		# die Wahrscheinlichkeit für Rückschritte größer
		# als ein Zufallswert zwischen 0 und 1, dann...
		if (new_length<length or np.random.random()<p):
			# ...übernehme den Wert
			graph = np.append(graph,new_length)
		else:			
			# sonst...tausche die Städte zurück
			travel[i], travel[j] = travel[j], travel[i]


	plt.title('Simulated Annealing')
	plt.xlabel('Vertauschungsschritt')
	plt.ylabel('Rundreiselänge')
	plt.plot(graph)	
	plt.show()

if __name__ == "__main__":
    main()
