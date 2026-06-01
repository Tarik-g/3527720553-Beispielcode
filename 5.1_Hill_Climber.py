import numpy as np
import matplotlib.pyplot as plt

cities = 100

def get_length(travel, distance):
	# Entfernung vom Ende der Reise zum Anfang
	length=distance[int(travel[-1]),int(travel[0])] 
	for i in range(0, cities-1):
		# Einzelentfernungen aufaddieren
		length+=distance[int(travel[i]),int(travel[i+1])]
	return length

def main():
	# Entfernungstabelle mit Zufallszahlen
	distance = np.random.rand(cities, cities)
	# Rundreise als 0,1,2 ... 99 als Startrundreise setzen
	travel = np.linspace(0, cities, cities, endpoint=False)
	graph = np.array([])
	# 10000 Versuche, besser zu werden
	for step in range(0,10000):
		length = get_length(travel, distance)
		i = np.random.randint(0,cities-1)
		j = np.random.randint(0,cities-1)
		# vertausche Stadt i mit Stadt j
		travel[i], travel[j] = travel[j], travel[i]
		new_length = get_length(travel, distance)
		# Wenn die neue Rundreise länger ist, dann ...
		if (new_length>length):
			# ... tausche die Städte zurück
			travel[i], travel[j] = travel[j], travel[i]
		else:
			graph = np.append(graph,new_length)

	plt.title('Hill Climber')
	plt.xlabel('Verbesserungsschritt')
	plt.ylabel('Rundreiselänge')
	plt.plot(graph)
	plt.show()

if __name__ == "__main__":
    main()