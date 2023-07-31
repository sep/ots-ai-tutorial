from pathfinding.core.diagonal_movement import DiagonalMovement
import os

MAP_ROOT= os.path.join("data", "maps")
SCENARIO_ROOT = os.path.join("data", "scenarios")
TEST_SCEN_PATH= os.path.join(SCENARIO_ROOT, "AR0011SR.map.scen")
TEST_MAP_PATH= os.path.join(MAP_ROOT, "AR0011SR.map")
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
        self.name = None

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
            newMap.name = os.path.basename(path)
            return newMap

        
class InstanceGenerator():

    def __init__(self):
        self.name = None
        self.map = None
        self.traversalDicts = []


    @staticmethod
    def mapPathOfInstancePath(instancePath):
        mapName = os.path.basename(instancePath)
        mapName = mapName[:-5]
        mapName = os.path.join("data", "maps", mapName)
        return mapName


    @staticmethod
    def load(instancePath, mapPath=None):
        if mapPath is None:
            mapPath = InstanceGenerator.mapPathOfInstancePath(instancePath)
        result = InstanceGenerator()
        result.name = os.path.basename(instancePath)
        if not os.path.basename(mapPath) in result.name:
            print("It appears the map and instances are not aligned:", instancePath, mapPath)
            raise ValueError
        else:
            result.map = Map.loadMap(mapPath)
            with open(instancePath, 'r') as stream:
                lines = stream.readlines()
                lines = [line[:-1] for line in lines]
                for targetPath in lines[1:]:
                    pieces = targetPath.split()
                    if not len(pieces) == 9:
                        print("Poorly formatted scenario file. Each line should have 9 entries")
                        raise ValueError
                    result.traversalDicts.append({
                        'width' : int(pieces[2]),
                        'height' : int(pieces[3]),
                        'start' : {
                            'x' : int(pieces[4]),
                            'y' : int(pieces[5]),
                        },
                        'goal' : {
                            'x' : int(pieces[6]),
                            'y' : int(pieces[7]),
                        },
                        'cost' : float(pieces[8])
                    })
                    
        return result
        

def checkAllMapsLoadable(rootDir):
    for fname in os.listdir(rootDir):
        mapPath = os.path.join(rootDir, fname)
        Map.loadMap(mapPath)
    return True


def checkAllScenariosLoadable(rootDir):
    for fname in os.listdir(rootDir):
        instancePath = os.path.join(rootDir, fname)
        InstanceGenerator.load(instancePath)
    return True
            

if __name__ == "__main__":
    checkAllScenariosLoadable(SCENARIO_ROOT)
