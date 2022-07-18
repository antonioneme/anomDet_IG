import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from scipy.stats import wasserstein_distance
from scipy import spatial, stats
import random
from tqdm import tqdm
import argparse
from sklearn import cluster

class DistanceType(Enum):
    """
    Tipos de distancia a utilizar
    1.- Manhattan
    2.- Euclidiana
    3.- Wasserstein
    4.- Coseno
    5.- Aitchison (para datos composicionales)
    """
    MANHATTAN = 0
    EUCLIDEAN = 1
    WASSERSTEIN = 2
    COSINE = 3
    AITCHISON = 4
    HELLINGER = 5
    HILBERT = 6
    FHR = 7
    JS = 8

class DistanceCalculator:
    """
    Clase utilizada para calcular las distancias
    """
    def manhattanDistance(self, vector1, vector2):
        n1 = len(vector1)
        n2 = len(vector2)
        #Si los vectores no son de igual dimensión vamos a mandar un 
        #mensaje al usuario
        if(n1 != n2):
            print("Los vectores deben ser de la misma dimensión")
            return -1
        d = 0
        for i in range(0,n1):
            d += np.abs(vector1[i] - vector2[i])
    
        return d


    def euclideanDistance(self, vector1, vector2):
        n1 = len(vector1)
        n2 = len(vector2)
    
        if(n1 != n2):
            print("Los vectores deben ser de la misma dimensión")
            return -1
    
        d = 0
        for i in range(0,n1):
            d += (vector1[i] - vector2[i])**2
    
        d = np.sqrt(d)
        return d


    def cosineDistance(self, vector1, vector2):
        n1 = len(vector1)
        n2 = len(vector2)
    
        if(n1 != n2):
            print("Los vectores deben ser de la misma dimensión")
            return -1
    
        #Obtenemos el producto punto y las dos normas
        dotProduct = 0
        norm1 = 0
        norm2 = 0
        for i in range(0,n1):
            dotProduct += vector1[i]*vector2[i]
            norm1 += vector1[i]**2
            norm2 += vector2[i]**2
    
        norm1 = np.sqrt(norm1)
        norm2 = np.sqrt(norm2)
    
        cosineDistance = 1 - (dotProduct/(norm1*norm2))
        return cosineDistance
    
    #Metodos necesarios para calcular la distancia Wasserstein
    def generateDataWithDistribution(self,distribution, nData):
        distribution = np.array(distribution)/sum(distribution)
        cumulativeDistribution = np.zeros(len(distribution) + 1)
        #Calculamos la distribución acumulada
        for i in range(1, len(cumulativeDistribution)):
            cumulativeDistribution[i] = cumulativeDistribution[i - 1] + distribution[i-1]

        generatedData = []
        for i in range(0,nData):
            diceRoll = random.random()
            index = 0
            for j in range(1, len(cumulativeDistribution)):
                if(diceRoll >= cumulativeDistribution[j-1] and diceRoll<= cumulativeDistribution[j]):
                    index = j-1
                    generatedData.append(index)
                    break
    
        return generatedData
    
    def wassersteinDistance(self, vector1, vector2):

        """
        Substitute it by what is implemented on DB-Ng-Knorr

        """
        ds = wasserstein_distance(vector1, vector2)

        return ds

        """
        data1 = self.generateDataWithDistribution(vector1, 1000)
        data2 = self.generateDataWithDistribution(vector2, 1000)
        return wasserstein_distance(data1, data2)
        """

    def aitchisonDistance(self, vector1, vector2):
        #Debemos volver composicionales los datos p
        vector1 = np.array(vector1)/sum(vector1)
        vector2 = np.array(vector2)/sum(vector2)
        D = len(vector1)
        distance = 0
        for i in range(0,D):
            x1i = vector1[i]
            x2i = vector2[i]
            for j in range(0,D):
                x1j = vector1[j]
                x2j = vector2[j]
                if(x1i != 0 and x2i != 0 and x1j != 0 and x2j != 0):
                    distance = distance + (1/(2*D))*(np.log(x1i/x1j) - np.log(x2i/x2j))**2
        
        distance = np.sqrt(distance)
        
        return distance

    def hellingerDistance(self, vector1, vector2):
        #Debemos volver composicionales los datos p
        vector1 = np.array(vector1)/sum(vector1)
        vector2 = np.array(vector2)/sum(vector2)

        d = spatial.distance.euclidean(np.sqrt(vector1), np.sqrt(vector2))/np.sqrt(2.0)

        return d

    def hilbertDistance(self, vector1, vector2):
        #Debemos volver composicionales los datos p
        vector1 = np.array(vector1)/sum(vector1)
        vector2 = np.array(vector2)/sum(vector2)
        mx = -1000000
        mn = 1000000
        #print ("Hilbert")
        for i,p in enumerate(vector1):
                #print ("hl = ", i, p)
                if vector2[i] > 0:
                        r = p/vector2[i]
                else:
                        r = 0.0
                if r > mx:
                        mx = r
                if r < mn:
                        mn = r
        if mn> 0:
                d = np.log(mx/mn)
        else:
                d = 1.0

        return d
    
    def fhrDistance(self, vector1, vector2):
        #Debemos volver composicionales los datos p
        vector1 = np.array(vector1)/sum(vector1)
        vector2 = np.array(vector2)/sum(vector2)
        s = 0.0 
        for i,p in enumerate(vector1):
                s = s + np.sqrt(p * vector2[i])
        d = 2.0 * np.arccos(s)
        return d

    def jsDistance(self, vector1, vector2):
        #Debemos volver composicionales los datos p
        d = spatial.distance.jensenshannon(vector1, vector2)
        #d = spatial.distance.euclidean(np.sqrt(vector1), np.sqrt(vector2))/np.sqrt(2.0)
        return d
    
    
    
    def computeDistance(self, distanceType, vector1, vector2):
        if(distanceType == DistanceType.MANHATTAN):
            return self.manhattanDistance(vector1, vector2)
        elif(distanceType == DistanceType.EUCLIDEAN):
            return self.euclideanDistance(vector1, vector2)
        elif(distanceType == DistanceType.WASSERSTEIN):
            return self.wassersteinDistance(vector1, vector2)
        elif(distanceType == DistanceType.COSINE):
            return self.cosineDistance(vector1, vector2)
        elif(distanceType == DistanceType.AITCHISON):
            return self.aitchisonDistance(vector1, vector2)
        elif(distanceType == DistanceType.HELLINGER):
            return self.hellingerDistance(vector1, vector2)
        elif(distanceType == DistanceType.HILBERT):
            return self.hilbertDistance(vector1, vector2)
        elif(distanceType == DistanceType.FHR):
            return self.fhrDistance(vector1, vector2)
        elif(distanceType == DistanceType.JS):
            return self.jsDistance(vector1, vector2)


#Pequeña prueba del calculador de distancias

#vector1 = np.array([5,2,4])
#vector2 = np.array([1,2,4])

#distanceCalculator = DistanceCalculator()
#print("Euclidean distance "+str(distanceCalculator.computeDistance(DistanceType.EUCLIDEAN, vector1, vector2)))
#print("Manhattan distance "+str(distanceCalculator.computeDistance(DistanceType.MANHATTAN, vector1, vector2)))
#print("Wasserstein distance "+str(distanceCalculator.computeDistance(DistanceType.WASSERSTEIN, vector1, vector2)))
#print("Cosine distance " + str(distanceCalculator.computeDistance(DistanceType.COSINE, vector1, vector2)))


class LOFArbitraryDistance:
    
    def __init__(self, data, distanceType, minPoints):
        """
        data: una tabla de datos donde cada fila es un vector
        """
        self.data = data
        self.distanceType = distanceType
        #Para no tener que crear demasiados calculadores de distancia
        #Creare uno que usaré en los distintos métodos del objeto.
        self.distanceCalculator = DistanceCalculator()
        self.minPoints = minPoints
    
    def getKDistance(self, point):
        nData = np.size(self.data, 0)
        kCounter = 0
        foundVectors = []
        distanceSoFar = 0
        for i in range(0,self.minPoints):
            minDist = float('inf')
            minVector = None
            for j in range(0,nData):
                vector = self.data[j,:]
                d = self.distanceCalculator.computeDistance(self.distanceType, vector, point) 
                if(not list(vector) in foundVectors and d <= minDist):
                    minVector = vector
                    minDist = d
                
            foundVectors.append(list(minVector))
            distanceSoFar = minDist
        
        return distanceSoFar
    #Para usar este metodo hay que obtener primero la distancia k
    def getNeighborhood(self, point, kDistance):
        neighbors = []
        nData = np.size(self.data, 0)
        for i in range(0,nData):
            vector = self.data[i,:]
            if(self.distanceCalculator.computeDistance(self.distanceType, point,vector) <= kDistance):
                neighbors.append(vector)
        
        return neighbors
    
    def getReachabilityDistance(self, point, neighbor):
        d = self.distanceCalculator.computeDistance(self.distanceType, point, neighbor)
        kDistance = self.getKDistance(neighbor)
        return np.max([d, kDistance])
    
    def getLocalReachabilityDistance(self, point):
        pointKDistance = self.getKDistance(point)
        #Obtenemos los puntos del k-vecindario
        neighbors = self.getNeighborhood(point, pointKDistance)
        nNeighbors = len(neighbors)
        lrd = 0
        for i in range(0,nNeighbors):
            neighbor = neighbors[i]
            reachDist = self.getReachabilityDistance(point, neighbor)
            lrd = lrd + reachDist/nNeighbors
        lrd = 1.0/lrd
        return lrd
    
    
    def getLOFScore(self, point):
        lrdPoint = self.getLocalReachabilityDistance(point)
        pointKDistance = self.getKDistance(point)
        neighbors = self.getNeighborhood(point, pointKDistance)
        #print ("i = ", lrdPoint, pointKDistance, neighbors)
        nNeighbors = len(neighbors)
        lofScore = 0
        for i in range(0,nNeighbors):
            neighbor = neighbors[i]
            lrdNeighbor = self.getLocalReachabilityDistance(neighbor)
            lofScore = lofScore + lrdNeighbor/lrdPoint
        
        return lofScore/nNeighbors
    
    def computeAllLOFScores(self):
        lofScores = []
        for i in tqdm(range(0,np.size(self.data, 0))):
            lofScores.append(self.getLOFScore(self.data[i,:]))
        
        return lofScores    



def createLabyrinthPoints(nPoints, nAnnulus, maxRadius):
    delta = maxRadius/(2*nAnnulus)
    annulusBoundaries = []
    distanceCalculator = DistanceCalculator()
    for i in range(0,nAnnulus):
        if(i%2 == 0):
            annulusBoundaries.append([i*delta, (i+1)*delta])
    points = []
    while(len(points) < nPoints):
        newPoint = [-maxRadius + 2*random.random()*maxRadius, -maxRadius + 2*random.random()*maxRadius]
        normPoint = distanceCalculator.euclideanDistance(newPoint, [0,0])
        for j in range(0,len(annulusBoundaries)):
            boundaries = annulusBoundaries[j]
            if(normPoint >= boundaries[0] and normPoint <= boundaries[1]):
                points.append(newPoint)
    return np.array(points)


#points = createLabyrinthPoints(2000, 5, 10)

#plt.figure(figsize = (7,7))
#plt.plot(points[:,0], points[:,1], color = "#39b3fa", marker = "o", linestyle = "none")
#plt.axis('equal')

#Agreguemos un par de puntos anomalos anomalo
#points = list(points)
#points.append([1.5,0])
#points.append([3.5,0])
#points.append([6,0])
#points.append([8,0])
#points = np.array(points)

#plt.figure(figsize = (7,7))
#plt.plot(points[:,0], points[:,1], color = "#39b3fa", marker = "o", linestyle = "none")
#plt.axis('equal')



class ColorGenerator:
    #Regresa un esquema de colores aleatorio    
    def buildRandomScheme(self, numberOfColors):
        colorStrings = []
        for i in range(0,numberOfColors):
            colorStrings.append(self.getRandomColor())
            
        return colorStrings
        
    def getRandomColor(self):
        s = "#"
        r = str(hex(randrange(16)))[-1:] + str(hex(randrange(16)))[-1:]
        g = str(hex(randrange(16)))[-1:] + str(hex(randrange(16)))[-1:]
        b = str(hex(randrange(16)))[-1:] + str(hex(randrange(16)))[-1:]
        s = s+r+g+b
        
        return s
    
    def hex2Vector(self, value):
        r = int(value[1:3],16)
        g = int(value[3:5],16)
        b = int(value[5:7],16)
        
        return np.array([r,g,b])
    
    def getGradientColor(self, startColor, endColor, value, minValue, maxValue):
        color = startColor + ((value - minValue)/(maxValue - minValue))*(endColor - startColor)
        
        s = '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))
        return s

class ColorPointPlotter:
    
    def plotPointsWithColors(self, points, values, minColor, maxColor, title, ax):
        nPoints = np.size(points, 0)
        maxValue = np.max(values)
        minValue = np.min(values)
        colorGenerator = ColorGenerator()
        maxRegistered = False
        minRegistered = False
        
        for i in range(0,nPoints):
            point = points[i,:]
            pointColor = colorGenerator.getGradientColor(minColor, maxColor, values[i], minValue, maxValue)
            if(not maxRegistered and values[i] == maxValue):
                ax.plot(point[0], point[1], color = pointColor, marker = "o", linestyle = "none", label = "Max val. "+str(round(maxValue,2)))
                maxRegistered = True
            elif(not minRegistered and values[i] == minValue):
                ax.plot(point[0], point[1], color = pointColor, marker = "o", linestyle = "none", label = "Min val. "+str(round(minValue,2)))
                minRegistered = True
            else:
                ax.plot(point[0], point[1], color = pointColor, marker = "o", linestyle = "none")
        
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        ax.legend()
        
    def plotPointsWithColorsMinMax(self, points, values, minColor, maxColor,minValue, maxValue, title, ax):
        nPoints = np.size(points, 0)
        colorGenerator = ColorGenerator()
        maxRegistered = False
        minRegistered = False
        firstPoint = points[0,:]
        ax.plot(firstPoint[0],firstPoint[1],color = maxColor/255, linestyle = "none", marker = "o", label = "Max val. "+str(round(maxValue, 2)))
        ax.plot(firstPoint[0],firstPoint[1], color = minColor/255, linestyle = "none", marker = "o", label = "Min val. "+str(round(minValue, 2)))
        for i in range(0,nPoints):
            point = points[i,:]
            pointColor = colorGenerator.getGradientColor(minColor, maxColor, values[i], minValue, maxValue)
            ax.plot(point[0], point[1], color = pointColor, marker = "o", linestyle = "none")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        ax.legend()

def	read_cda(FF):

	f = open(FF, "r")
	x = f.readlines()
	f.close()

	Pts = []
	for line in x:
		xx = line.split('\t')
		tmp = []
		for j in xx:
			tmp.append(float(j))
		Pts.append(tmp)
	return [Pts, len(Pts[0]), len(Pts)]

def	save_lof(FF, scoreLOF, Distances, numP):
	f = open(FF, "w")
	f.write("#")
	for ds in Distances:
		f.write(ds + "\t")
	f.write("dist\n")
	for i in range(numP):
		for ds in Distances:
			f.write(str(scoreLOF[ds][i]) + "\t")
		f.write(str(i) + "\n")
	f.close()

def	save_stats(FF, Distances, InfoRecall, InfoSpecificity, InfoAccuracy, InfoTP):
	f = open(FF, "w")

	f.write("#Distance\tRecall\tSpec.\tAccu.\tTP\n")

	for ds in Distances:
		f.write(ds + "\t")
		f.write( str(InfoRecall[ds]) + "\t" + str(InfoSpecificity[ds]) + "\t" + str(InfoAccuracy[ds]) + "\t" + str(InfoTP[ds]) + "\n" )
	f.close()


def	save_class(FF, Distances, Class, numP):
	f = open(FF, "w")

	f.write("#")
	for ds in Distances:
		f.write(ds + "\t")
	f.write("\n")

	for i in range(numP):
		for ds in Distances:
			f.write( str( Class[ds][i] ) + "\t")
		f.write( str(i) + "\n" )

	f.close()

"""
python3 lof_cda_several.py  -i 
"""
parser = argparse.ArgumentParser()
parser.add_argument('-i', action = "store", dest = "i", help = "The input file containing the compositional data to be checked for anomalies")
parser.add_argument('-o', action = "store", dest = "o", help = "The output file containing the LOF score for each input vector")
parser.add_argument('-o2', action = "store", dest = "o2", help = "The output file containing classification metrics for LOF score for each input vector and kmeans")
parser.add_argument('-o3', action = "store", dest = "o3", help = "The output file containing the class based on LOF and KM")
#parser.add_argument('-d', action = "store", dest = "d", help = "The distance function")
parser.add_argument('-k', action = "store", dest = "k", help = "The minPoints parameter for LOF")
parser.add_argument('-u', action = "store", dest = "u", help = "The first u rows are expected vectors")

args = parser.parse_args()
"""
    MANHATTAN = 0
    EUCLIDEAN = 1
    WASSERSTEIN = 2
    COSINE = 3
    AITCHISON = 4
"""
#distanceType = DistanceType.EUCLIDEAN



#points = [[0.3, 0.4, 0.3], [0.5, 0.1, 0.4], [0.2, 0.7, 0.1], [0.5, 0.0, 0.5], [0.2, 0.2, 0.6], [0.7, 0.0, 0.3], [0.0, 0.3, 0.7], [0.1, 0.2, 0.7], [0.0, 0.05, 0.95], [0.8, 0.0, 0.2], [0.666, 0.333, 0.001]]


[points, dim, numP] = read_cda(args.i)

points = np.array(points)

minPoints = int(args.k)

nu = int(args.u)
numanom = numP - nu

#Distances = ['euclidean', 'manhattan', 'wasserstein', 'cosine', 'hellinger', 'fhr', 'js']

#Distances = ['euclidean', 'manhattan', 'wasserstein', 'cosine', 'aitchison', 'hellinger', 'fhr', 'js', 'hilbert']
#Distances = ['euclidean', 'manhattan', 'wasserstein', 'cosine', 'aitchison', 'hellinger', 'fhr', 'js']
Distances = ['euclidean', 'manhattan', 'wasserstein', 'cosine', 'aitchison', 'hellinger', 'hilbert', 'fhr', 'js']

InfoRecall = {}
InfoSpecificity = {}
InfoAccuracy = {}
InfoTP = {}

Class = {}

scoreLOF = {}
for ds in Distances:
	if ds == 'euclidean':
		distanceType = DistanceType.EUCLIDEAN
	else:
		if ds == 'manhattan':
			distanceType = DistanceType.MANHATTAN
		else:
			if ds == 'wasserstein':
				distanceType = DistanceType.WASSERSTEIN
			else:
				if ds == 'cosine':
					distanceType = DistanceType.COSINE
				else:
					if ds == 'aitchison':
						distanceType = DistanceType.AITCHISON
					else:
						if ds == 'hellinger':
							distanceType = DistanceType.HELLINGER
						else:
							if ds == 'hilbert':
								distanceType = DistanceType.HILBERT
							else:
								if ds == 'fhr':
									distanceType = DistanceType.FHR
								else:
									if ds == 'js':
										distanceType = DistanceType.JS

	lofArbitraryDistance = LOFArbitraryDistance(points, distanceType, minPoints)

	anomalyScores = lofArbitraryDistance.computeAllLOFScores()

	scoreLOF[ds] = anomalyScores

	# Since LOF does not classify, let's add a classifer.
	# Apply kmeans with k = 2. The class for the first

	#print ("ds = ", scoreLOF[ds])
	"""
	km = cluster.KMeans(n_clusters = 2, n_init = 10, max_iter = 1000)
	X = list(map(lambda el:[el], scoreLOF[ds]))
	#print("ds = ", ds)
	#print ("X = ", X)
	if ds == 'wasserstein':
		Y = list(X)
		nX = []
		for yy in Y:
			if np.isnan(yy):
				nX.append([10.0])
			else:
				nX.append(yy)
		X = nX
		#print ("X = ", X)
	km.fit(X)
	#km.fit(scoreLOF[ds])
	#print ("X = ", X)

	TP = 0.0
	TN = 0.0
	FP = 0.0
	FN = 0.0
	# LU is the class assigned to each of the usual vectors
	LU = [0.0] * 2
	# LA is the class assigned to each of the anomalies
	LA = [0.0] * 2
	for v in range(numP):
		if v < nu:
			# consider the usual or expected vectors
			LU[km.labels_[v]] = LU[km.labels_[v]] + 1
		else:
			# Anomalies
			LA[km.labels_[v]] = LA[km.labels_[v]] + 1

	Class[ds] = km.labels_

	# The assumption is that the most numerous class is the true class
	if LU[0] >= LU[1]:
		if LU[1] > 0:
			TN = LU[0]/LU[1]
		else:
			TN = 1.0
		if LU[0] > 0:
			FP = LU[1]/LU[0]
		else:
			FP = 1.0
		# uc is the usual class
		uc = 0
		# ac is the anomaly class
		ac = 1
	else:
		if LU[0] > 0:
			TN = LU[1]/LU[0]
		else:
			TN = 1.0
		if LU[1] > 0:
			FP = LU[0]/LU[1]
		else:
			FP = 1.0
		uc = 1
		ac = 0

	TP = LA[ac]/numanom
	FN = LA[uc]/numanom
	"""

	TP = 0.0
	TN = 0.0
	FP = 0.0
	FN = 0.0
	# LU is the class assigned to each of the usual vectors
	LU = [0.0] * 2
	# LA is the class assigned to each of the anomalies
	LA = [0.0] * 2

	avgScoreU = 0.0
	avgScoreA = 0.0
	for i in range(numP):
		if i < nu:
			avgScoreU = avgScoreU + scoreLOF[ds][i]
		else:
			avgScoreA = avgScoreA + scoreLOF[ds][i]

	avgScoreU = avgScoreU/nu
	avgScoreA = avgScoreA/numanom

	#

	Cl = []
	for i in range(numP):
		du = abs(scoreLOF[ds][i] - avgScoreU)
		da = abs(scoreLOF[ds][i] - avgScoreA)
		if i < nu:
			if du < da:
				TN = TN + 1
				cl = 0
			else:
				FP = FP + 1
				cl = 1
		else:
			if da < du:
				TP = TP + 1
				cl = 1
			else:
				FN = FN + 1
				cl = 0
		Cl.append(cl)
	
	Class[ds] = Cl


	InfoRecall[ds] = TP/(TP + FN)
	InfoSpecificity[ds] = TN/(TN + FP)
	InfoAccuracy[ds] = (TP + TN)/(TP + TN + FP + FN)
	InfoTP[ds] = TP

#print ("pts = ", points)

#print ("dst = ", LOFArbitraryDistance)

#print ("LOF = ", anomalyScores)

save_lof(args.o, scoreLOF, Distances, numP)
save_stats(args.o2, Distances, InfoRecall, InfoSpecificity, InfoAccuracy, InfoTP)
save_class(args.o3, Distances, Class, numP)
#save_lof(args.o, anomalyScores)

