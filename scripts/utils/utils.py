import math
from typing import List, Tuple


class Vertex:
    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos
        self.edges_and_costs = {}

    def add_edge_with_cost(self, succ: Tuple[int, int], cost: float):
        if succ != self.pos:
            self.edges_and_costs[succ] = cost

    @property
    def edges_and_c_old(self):
        return self.edges_and_costs

class Vertices:
    def __init__(self):
        self.list = []

    def add_vertex(self, v: Vertex):
        self.list.append(v)

    @property
    def vertices(self):
        return self.list
    
def heuristic(p: Tuple[int, int], q: Tuple[int, int]) -> float:
    """
    Helper function to compute distance between two points.
    :param p: (x,y)
    :param q: (x,y)
    :return: manhattan distance
    """
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

def get_movements_8n(y: int, x: int) -> List:
    """
    get all possible 8-connectivity movements.
    :return: list of movements with cost [(dy, dx, movement_cost)]
    """
    return [(y + 1, x + 0),
            (y + 0, x + 1),
            (y - 1, x + 0),
            (y + 0, x - 1),
            (y + 1, x + 1),
            (y - 1, x + 1),
            (y - 1, x - 1),
            (y + 1, x - 1)]