class Pixel:
    def __init__(self, id, row, col, color, population, dem, rep):
        self.id = id
        self.row = row
        self.col = col
        self.color = color
        self.population = population
        self.dem = dem
        self.rep = rep

    def printInfo(self):
        print(f'ID: {self.id}\trow: {self.row}\tcol: {self.col}\tcolor: {self.color}')
    

class District:
    majority = 0
    def __init__(self, startRow, startCol, color, pixels, totalPopulation, totalDem, totalRep):
        self.row = startRow
        self.col = startCol
        self.color = color
        self.pixels = pixels
        self.totalPopulation = totalPopulation
        self.totalDem = totalDem
        self.totalRep = totalRep
    def addPixel(self, Pixel):
        self.pixels.append(Pixel)
        self.totalDem += Pixel.dem*1.3
        self.totalRep += Pixel.rep
        self.totalPopulation += Pixel.population
        self.checkMajority()

    def removePixel(self, id):
        for pixel in self.pixels:
            if pixel.id == id:
                self.pixels.remove(pixel)
                self.totalDem -= pixel.dem
                self.totalRep -= pixel.rep
                self.totalPopulation -= pixel.population
        self.checkMajority()

    def checkMajority(self):
        self.majority = 1 if self.totalDem > self.totalRep else 0
    
    def printInfo(self):
        print(f'Color: {self.color}\tSize: {len(self.pixels)}\tTotalPop: {self.totalPopulation} \tTotalDem: {self.totalDem} \tTotalRep: {self.totalRep} \tSplit: {self.totalDem/(self.totalDem+self.totalRep)}')
        
         