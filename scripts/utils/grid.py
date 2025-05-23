import numpy as np
from .utils import get_movements_8n
from typing import Dict, List, Tuple

OBSTACLE = 100

class OccupancyGridMap:
    def __init__(self, y_dim, x_dim):
        """
        set initial values for the map occupancy grid
        |----------> y, column
        |           (x=0,y=2)
        |
        V (x=2, y=0)
        x, row
        :param y_dim: dimension in the y direction
        :param x_dim: dimension in the x direction
        """
        self.y_dim = y_dim
        self.x_dim = x_dim

        # the map extents in units [m]
        self.map_extents = (y_dim, x_dim)

        # the obstacle map
        self.occupancy_grid_map = np.zeros(self.map_extents, dtype=np.uint8)

        # obstacles
        self.visited = {}

    def is_unoccupied(self, pos: Tuple[int, int]) -> bool:
        """
        :param pos: cell position we wish to check
        :return: True if cell is occupied with obstacle, False else
        """
        (row, col) = (round(pos[0]), round(pos[1]))
        return not self.occupancy_grid_map[row][col] == OBSTACLE

    def in_bounds(self, cell: Tuple[int, int]) -> bool:
        """
        Checks if the provided coordinates are within
        the bounds of the grid map
        :param cell: cell position (x,y)
        :return: True if within bounds, False else
        """
        return 0 <= cell[0] < self.y_dim and 0 <= cell[1] < self.x_dim

    def filter(self, neighbors: List, avoid_obstacles: bool):
        """
        :param neighbors: list of potential neighbors before filtering
        :param avoid_obstacles: if True, filter out obstacle cells in the list
        :return:
        """
        if avoid_obstacles:
            return [node for node in neighbors if self.in_bounds(node) and self.is_unoccupied(node)]
        return [node for node in neighbors if self.in_bounds(node)]

    def succ(self, vertex: Tuple[int, int], avoid_obstacles: bool = False) -> list:
        """
        :param avoid_obstacles:
        :param vertex: vertex you want to find direct successors from
        :return:
        """
        (y, x) = vertex

        movements = get_movements_8n(y=y, x=x)

        # not needed. Just makes aesthetics to the path
        if (y + x) % 2 == 0: movements.reverse()

        filtered_movements = self.filter(neighbors=movements, avoid_obstacles=avoid_obstacles)
        return list(filtered_movements)
    
    def update_cell(self, pos: Tuple[int, int], value: float) -> None:
        """
        :param pos: cell position we wish to update
        :param value: value we wish to set the cell to
        :return: None
        """
        (y, x) = (round(pos[0]), round(pos[1]))
        (row, col) = (y, x)
        self.occupancy_grid_map[row, col] = value

    def set_obstacle(self, pos: Tuple[int, int]):
        """
        :param pos: cell position we wish to set obstacle
        :return: None
        """
        (y, x) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (y, x)
        self.occupancy_grid_map[row, col] = OBSTACLE

    def remove_obstacle(self, pos: Tuple[int, int], value: float):
        """
        :param pos: position of obstacle
        :return: None
        """
        (y, x) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (y, x)
        self.occupancy_grid_map[row, col] = value