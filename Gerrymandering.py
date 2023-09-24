import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, distance
from Representation import Pixel
from Representation import District
from random import shuffle
import time

MAX_X, MAX_Y = 50, 50
NUM_CITIES = 5
MAX_POPULATION = 1000
DISTRICT_CENTERS = [(np.random.randint(MAX_Y), np.random.randint(MAX_X)) for col in range(NUM_CITIES)]
DISTRICTS = []
COLORS = [[np.random.randint(255), np.random.randint(255), np.random.randint(255)] for col in range(NUM_CITIES)]


# Generates intial map with random pop, dem voters, rep voters, and black pixels
def generateBlankMap():
    id = 0
    newMap = [[col for col in range(MAX_X)] for col in range(MAX_Y)]
    for r, row in enumerate(newMap):
        for c, col in enumerate(row):
            population = np.random.randint(10, MAX_POPULATION)
            voterPopulation = int(population * np.random.uniform(0.4, 0.8))
            partySplit = np.random.uniform(0.2, 0.8)
            dem = int(voterPopulation * partySplit)
            rep = int(voterPopulation - dem)
            # print(f'pop: {population} voterpop: {voterPopulation} partysplit: {partySplit} dem: {dem} rep: {rep}')
            newMap[r][c] = Pixel(id, r, c, [255, 255, 255], population, dem, rep)   
            id += 1
    return newMap

def generateDistricts():
    for i, district in enumerate(DISTRICT_CENTERS):
        DISTRICTS.append(District(district[0], district[1], COLORS[i], [], 0, 0, 0))


# Generates voronoi diagram by coloring each pixel in a district a different color
def generateVoronoiDiagram(map):
    # vor = Voronoi(DISTRICTS)
    for row in range(MAX_Y):
        for col in range(MAX_X):
            currPixel = voterMap[row][col]
            min_distance = float('inf')
            nearest_district = None
            for i, district in enumerate(DISTRICT_CENTERS):
                d = distance.euclidean([row, col], district)
                if d < min_distance:
                    min_distance = d
                    nearest_district = i
            
            DISTRICTS[nearest_district].addPixel(currPixel)
            voterMap[row][col].color = COLORS[nearest_district]
    # Coloring centers black
    # for district in DISTRICT_CENTERS:
    #     voterMap[district[0]][district[1]].color = [0, 0, 0]
    return voterMap


def getBoundaryPixels(map):
    boundaryMap = [[col for col in range(MAX_X)] for col in range(MAX_Y)]
    boundaryPixels = []
    # Add border buffer
    map.insert(0, [[0, 0, 0] for x in range(len(map))])
    map.append([[0, 0, 0] for x in range(len(map)-1)])
    for r, row in enumerate(map):
        map[r].insert(0, ([0, 0, 0]))
        map[r].append(([0, 0, 0]))
    
    # Coloring boundary pixels
    r, c = 1, 1
    while r < len(map)-1:
        while c < len(map[0])-1:
            currPixel = map[r][c]
            isBlack = False
            if currPixel == [0, 0, 0]:
                isBlack = True
            neighborPixels = [map[r+1][c], map[r-1][c], map[r][c-1], map[r][c+1]]
            for neighbor in neighborPixels:
                if currPixel != neighbor and isBlack == False and neighbor != [0, 0, 0]:
                    # voterMap[r-1][c-1].color = [0, 0, 0]
                    boundaryPixels.append(voterMap[r-1][c-1])
                    break
                else:
                    boundaryMap[r-1][c-1] = map[r][c]
            c +=1
        c = 1
        r +=1
    return boundaryPixels

def randFlipPixels(pixels):
    for pixel in pixels:
        pixel.color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        neighbors = []
        neighbors.append(voterMap[pixel.row+1][pixel.col]) if pixel.row != MAX_Y-1 else None
        neighbors.append(voterMap[pixel.row-1][pixel.col]) if pixel.row != 0 else None
        neighbors.append(voterMap[pixel.row][pixel.col+1]) if pixel.col != MAX_X-1 else None
        neighbors.append(voterMap[pixel.row][pixel.col-1]) if pixel.col != 0 else None
        # print(neighbors)
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor.color != pixel.color:
                pixel.color = neighbor.color
                for boundaryPixel in boundaryPixels:
                    if boundaryPixel.id == pixel.id:
                        boundaryPixels.remove(boundaryPixel)
                boundaryPixels.append(neighbor)



# Takes in voterMap and converts to visual RGB map
def getVisualMap(map):
    visualMap = [[col for col in range(MAX_X)] for col in range(MAX_Y)]
    for r, row in enumerate(visualMap):
        for c, col in enumerate(row):
            visualMap[r][c] = map[r][c].color
    return visualMap



# fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ax2 = plt.subplots()
voterMap = generateBlankMap()
generateDistricts()
voterMap = generateVoronoiDiagram(voterMap)
for district in DISTRICTS:
    district.printInfo()
# Cast back to visual rgb map to find boundaries
visualMap = getVisualMap(voterMap)
# ax1.imshow(voterMap)
boundaryPixels = getBoundaryPixels(visualMap)


visualMap = getVisualMap(voterMap)
        

ax2.imshow(visualMap)
plt.suptitle("Voronoi Diagram for Cities")


def update(frame):
    randFlipPixels(boundaryPixels)
    visualMap = getVisualMap(voterMap)
    visualMap_np = np.array(visualMap, dtype=np.uint8)
    visualMap = visualMap_np.tolist()
    ax2.imshow(visualMap)
    return ax2,

ani = FuncAnimation(fig, update, interval=100, blit=False)
plt.show()
