import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, distance
from skimage.draw import line

MAX_X, MAX_Y = 50, 50
fig, ax = plt.subplots()
voterData = np.zeros((MAX_Y, MAX_X, 3), dtype=np.uint8)  # 3 channels for RGB
edgeData = np.zeros((MAX_Y, MAX_X), dtype=bool)  # Boolean matrix to store edge pixels
numCities = 5
cities = [[np.random.randint(MAX_Y), np.random.randint(MAX_X)] for x in range(numCities)]
COLORS = [[np.random.randint(255), np.random.randint(255), np.random.randint(255)] for x in range(numCities)]

vor = Voronoi(cities)

for y in range(MAX_Y):
    for x in range(MAX_X):
        min_distance = float('inf')
        nearest_city = None
        for i, city in enumerate(cities):
            d = distance.euclidean([y, x], city)
            if d < min_distance:
                min_distance = d
                nearest_city = i
        voterData[y, x] = COLORS[nearest_city]

# Rasterize the Voronoi edges
for simplex in vor.ridge_vertices:
    if -1 not in simplex:  # Check if ridge is finite
        x0, y0 = vor.vertices[simplex[0]]
        x1, y1 = vor.vertices[simplex[1]]
        rr, cc = line(int(y0), int(x0), int(y1), int(x1))
        edgeData[rr, cc] = True

# Highlight the edges in a specific color, e.g., black
voterData[edgeData] = [0, 0, 0]

ax.imshow(voterData)
plt.title("Voronoi Diagram with Edges Highlighted")
plt.axis('off')

def update(frame):
    # If you want to update the Voronoi diagram in the animation, you can add the logic here.
    return

ani = FuncAnimation(fig, update, interval=100, blit=False)
plt.show()
