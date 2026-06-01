import numpy as np
import matplotlib.pyplot as plt

pop_size = 1000 # Populationsgröße
gen_len  = 100  # Länge eines Gens
gene     = np.zeros((pop_size, gen_len),dtype=int)
new_gene = np.zeros((pop_size, gen_len),dtype=int)
fitns    = np.ones(pop_size)

rek_rate = 0.8  # Rekombinationsrate
mut_rate = 0.2  # Mutationsrate

job_time = np.random.rand(gen_len) # Array von Jobzeiten

def rank_select():
	# Vereinfachte Rank Selection, nur die besten 500 
	# Individuen mit gleicher Wahrscheinlichkeit wählen
    return np.random.randint(0,pop_size/2)

def crossover(): # Rekombination
	global gene
	split = np.random.randint(gen_len)
	sel1  = rank_select() # selektiere Individuum 1
	sel2  = rank_select() # selektiere Individuum 2
	return np.append(gene[sel1][:split],gene[sel2][split:]), np.append(gene[sel2][:split],gene[sel1][split:])

def mutate(): # Mutation
	global gene, pop_size
	for i in range((int)(pop_size*mut_rate)):
		indi = np.random.randint(0,pop_size-1)
		bit  = np.random.randint(0,gen_len-1)# irgendein
		gene[indi][bit] = 1-gene[indi][bit]  # Bit kippen

def eval_fitness(): # Auswertung der Fitness
	global job_time, fitns, pop_size, gene
	sum = 0
	for i in range(pop_size):
		# Zeit berechnen, die jede CPU braucht 
		time_for_comp = np.zeros((2,), dtype=float)
		for j in range(gen_len):
			bit = gene[i][j] # Job auf CPU 0 oder 1
			time_for_comp[bit]+=job_time[j]
		# Fitness = negative Gesamtzeit
		# wobei Gesamtzeit = Maximum der CPU-Zeiten
		fitns[i] = -max(time_for_comp[0],time_for_comp[1])
	# sortiere gene[] nach fitns[] absteigend
	sorted_indices = np.flip(np.argsort(fitns))
	gene = gene[sorted_indices]
	fitns = fitns[sorted_indices]

if __name__ == "__main__":
	graph = np.array([])
	for i in range(10):
		mutate() # initialisiere das Array zufällig
	for gen in range(0,100): # Generation
		eval_fitness() # Fitness auswerten
		print("Gen:"+str(gen)+" Bestzeit:"+str(-fitns[0]))
		graph = np.append(graph,-fitns[0])
		for i in range(pop_size):
			if i<pop_size*rek_rate:
				# erzeuge zwei neue Individuen
				new_gene[i],new_gene[i+1] = crossover();
				i=i+1 # weil zwei Individuen erzeugt wurden
			else:
				# selektiere ein Individuum
				new_gene[i] = np.copy(gene[rank_select()])

		gene = np.copy(new_gene)
		mutate()

	plt.title('Genetischer Algorithmus')
	plt.xlabel('Generation')
	plt.ylabel('Beste Gesamtzeit')
	plt.plot(graph)	
	plt.show()