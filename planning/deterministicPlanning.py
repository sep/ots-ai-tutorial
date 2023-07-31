from pathfinding.core.diagonal_movement import DiagonalMovement
import os

MAP_ROOT= os.path.join("data", "maps")
TEST_PATH= os.path.join(MAP_ROOT, "AR0011SR.map")
OCTILE_MOTION = 'type octile'
HEIGHT = 'height'
WIDTH = 'width'
MAP = 'map'
TRAVERSABLE = ['.', 'G', 'S', 'W']
BLOCKED = ['@', 'O']

class Map():
    
    def __init__(self):
        self.motionModel = None
        self.occlusion = None

    @staticmethod
    def loadMap(path):
        with open(path, 'r') as stream:
            lines = stream.readlines()
            lines = [line[:-1] for line in lines]
            newMap = Map()
            if lines[0] == 'type octile':
                newMap.motionModel = DiagonalMovement
            else:
                print("Saw unexpected motion model:", lines[0])
                raise ValueError
            height = -1
            width = -1
            dimLine = lines[1].split()
            if dimLine[0] == HEIGHT:
                height = int(dimLine[1])
            elif dimLine[0] == WIDTH:
                width = int(dimLine[1])
            else:
                print("Expected to find height or width, found:", lines[1])
                raise ValueError
            dimLine = lines[2].split()
            if dimLine[0] == HEIGHT:
                height = int(dimLine[1])
            elif dimLine[0] == WIDTH:
                width = int(dimLine[1])
            else:
                print("Expected to find height or width, found:", lines[1])
                raise ValueError
            if not (lines[3] == MAP):
                print("Not parsing a map, instead reading a:", lines[3])
                raise ValueError
            grid = []
            for mapLine in lines[4:]:
                lineAr = []
                for char in mapLine:
                    if char in TRAVERSABLE:
                      lineAr.append(False)
                    elif char in BLOCKED:
                      lineAr.append(True)
                    else:
                      print("Saw an unexpected map glyph while parsing", char)
                      raise ValueError
                if not (len(lineAr) == width):
                      print ("Map line had a bad size {}, expected {}".format(len(lineAr), width))
                      raise ValueError
                grid.append(lineAr)
            if not (len(grid) == height):
                      print ("Map has wrong height. Got {} but expected {}".format(len(grid), height))
                      raise ValueError
            newMap.occlusion = grid
            return newMap


def checkAllMapsLoadable(rootDir):
    for fname in os.listdir(rootDir):
        mapPath = os.path.join(rootDir, fname)
        Map.loadMap(mapPath)
    return True
            

if __name__ == "__main__":
    checkAllMapsLoadable(MAP_ROOT)
