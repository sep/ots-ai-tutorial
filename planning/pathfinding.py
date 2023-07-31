from pathfinding.core.diagonal_movement import DiagonalMovement
import os

TEST_PATH= os.path.join("data", "maps", "AR0011SR.map")
OCTILE_MOTION = 'type octile'
HEIGHT = 'height'
WIDTH = 'width'
MAP = 'map'

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
            print(lines[0:4])
            

if __name__ == "__main__":
    Map.loadMap(TEST_PATH)
