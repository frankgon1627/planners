import numpy as np
import tf2_geometry_msgs
import rclpy
import math

from utils.priority_queue import Priority, PriorityQueue
from utils.grid import OccupancyGridMap
from utils.utils import Vertex, Vertices, heuristic
from typing import Dict, List, Tuple
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

class DStarLite(Node):
    def __init__(self) -> None:
        super().__init__("dstar_lite")
        self.create_subscription(Odometry, '/dlio/odom_node/odom', self.odometry_callback, 10)
        self.create_subscription(OccupancyGrid, '/obstacle_detection/combined_map', self.occupancy_grid_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.path_publisher = self.create_publisher(Path, '/planners/d_star_lite_path', 10)
        
        self.create_timer(0.1, self.generate_trajectory)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.odometry: Odometry | None = None
        self.goal: PoseStamped | None = None

        # local map information
        self.combined_grid: OccupancyGrid | None = None
        self.local_origin: Tuple[float, float] | None = None
        self.resolution: float | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.latest_data: np.ndarray[float] | None = None

        # global map that accumulates all the observations
        self.global_map: OccupancyGridMap | None = None
        self.global_map_origin: Tuple[float, float] | None = None
        self.global_map_height: int | None = None
        self.global_map_width: int | None = None

        # D-Star Lite Algorithm variables
        self.U: PriorityQueue = PriorityQueue()
        self.rhs: np.ndarray[float] | None = None
        self.g: np.ndarray[float] | None = None
        self.g_distance: np.ndarray[float] | None = None
        self.k_m: float = 0.0
        self.s_start: Tuple[int, int] | None = None
        self.s_goal: Tuple[int, int] | None = None
        self.last_position: Tuple[int, int] | None = None

        self.get_logger().info("D*-Lite Node Initialized")

    def odometry_callback(self, msg: Odometry) -> None:
        """Processes Odometry messages from ROS 2"""
        self.odometry = msg        

    def occupancy_grid_callback(self, msg: OccupancyGrid) -> None:
        """Processes OccupancyGrid messages from ROS 2"""
        self.combined_grid: OccupancyGrid = msg
        self.width: int = msg.info.width
        self.height: int = msg.info.height
        self.resolution: float = msg.info.resolution
        self.local_origin: Tuple[float, float] = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.data: np.ndarray[float] = np.array(msg.data).reshape((self.height, self.width))

    def goal_callback(self, msg: PoseStamped) -> None:
        """Receives goal position and determines start and goal positions in grid coordinates"""
        self.goal = msg

        # convert goal to odom frame if not in odom frame
        if self.goal.header.frame_id != "odom":
            in_odom_frame = False
            while not in_odom_frame:
                try:
                    transform = self.tf_buffer.lookup_transform('odom', self.goal.header.frame_id, rclpy.time.Time())
                    self.goal = tf2_geometry_msgs.do_transform_pose_stamped(self.goal, transform)
                    self.get_logger().info(f"Converted Goal to: {self.goal.pose.position}")
                    in_odom_frame = True
                except:
                    self.get_logger().warn("Failed to get Transform from base_link to map")

        self.initialize_global_map()

        # set start position and convert start and goal to grid coordinates
        global_start_i: int = int((self.odometry.pose.pose.position.x - self.global_map_origin[0])/self.resolution)
        global_start_j: int = int((self.odometry.pose.pose.position.y - self.global_map_origin[1])/self.resolution)
        self.last_position: Tuple[int, int] = (global_start_i, global_start_j)

        global_goal_i: int = int((self.goal.pose.position.x - self.global_map_origin[0])/self.resolution)
        global_goal_j: int = int((self.goal.pose.position.y - self.global_map_origin[1])/self.resolution)
        self.s_goal: Tuple[int, int] = (global_goal_i, global_goal_j)

        self.initialize_d_star_lite()
        path, g, rhs = self.move_and_replan(self.last_position)

        # TODO: Add something to publish the path

    def generate_trajectory(self) -> None:
        if self.combined_grid is None:
            self.get_logger().warn("No Occupancy Grid Received yet.")
            return

        if self.odometry is None:
            self.get_logger().warn("No Odometry Received yet.")
            return

        if self.goal is None:
            self.get_logger().warn("No Goal Received yet.")
            return
        
        global_start_i: int = int((self.odometry.pose.pose.position.x - self.global_map_origin[0])/self.resolution)
        global_start_j: int = int((self.odometry.pose.pose.position.y - self.global_map_origin[1])/self.resolution)
        new_position: Tuple[int, int] = (global_start_i, global_start_j)

        if new_position != self.last_position:
            # extract the information from the local map
            local_data: List[float] = self.combined_grid.data
            nodes: Dict[Tuple[int, int], float] = {}
            for s_i in range(self.width):
                for s_j in range(self.height):
                    value: float = local_data[s_j * self.width + s_i]
                    x_position: float = s_i * self.resolution + self.local_origin[0]
                    y_position: float = s_j * self.resolution + self.local_origin[1]
                    nodes[(x_position, y_position)] = value

            vertices: Vertices = Vertices()
            # TODO: GENERAL UPDATE CELL RATHER THAN JUST OBSTACLE AND FREE SPACE TO CONSIDER RISK
            for node, value in node.items():
                # if the node is perceived to be an obstacle
                if value == 100:
                    if self.global_map.is_unoccupied(node):
                        v: Vertex = Vertex(pos=node)
                        succ: List[Tuple[int, int]] = self.global_map.succ(node)
                        for u in succ:
                            v.add_edge_with_cost(succ, self.c(u, v.pos))
                            vertices.add_vertex(v)
                            self.global_map.set_obstacle(node)
                # if the node is perceived to be free space
                else:
                    if not self.global_map.is_unoccupied(node):
                        v: Vertex = Vertex(pos=node)
                        succ: List[Tuple[int, int]] = self.global_map.succ(node)
                        for u in succ:
                            v.add_edge_with_cost(succ, self.c(u, v.pos))
                            vertices.add_vertex(v)
                            self.global_map.remove_obstacle(node)
            self.new_edges_and_old_costs = vertices
            path, g, rhs = self.move_and_replan(new_position)
            # TODO: ADD SOMETHING TO PUBLISH THE PATH
            # TODO: FIGURE OUT HOW TO FIX COST TO INCORPORATE RISK

    def initialize_global_map(self) -> None:
        # initialize all variables relating to the global map
        original_bottom_right_x = self.combined_grid.info.origin.position.x
        original_bottom_right_y = self.combined_grid.info.origin.position.y
        original_top_left_x = original_bottom_right_x + self.width * self.resolution
        original_top_left_y = original_bottom_right_y + self.height * self.resolution

        # assume the goal location needs to call in the center of a map the same size as combined_risk_map_
        # determine the bottom right and top left of a map centered at goal
        goal_bottom_right_x = self.goal.pose.position.x - self.width * self.resolution / 2.0
        goal_bottom_right_y = self.goal.pose.position.y - self.height * self.resolution / 2.0
        goal_top_left_x = self.goal.pose.position.x + self.width * self.resolution / 2.0
        goal_top_left_y = self.goal.pose.position.y + self.height * self.resolution / 2.0

        # determine the boundaries of the global map, such that the global map minimally encompases the 
        # desired thresholds around the goal position and the original combined_risk_map_
        new_bottom_right_x = min(original_bottom_right_x, goal_bottom_right_x)
        new_bottom_right_y = min(original_bottom_right_y, goal_bottom_right_y)
        new_top_left_x = max(original_top_left_x, goal_top_left_x)
        new_top_left_y = max(original_top_left_y, goal_top_left_y)

        # construct an empty global planning grid
        global_height = new_top_left_y - new_bottom_right_y
        global_width = new_top_left_x - new_bottom_right_x
        self.global_map_height = int(global_height / self.resolution)
        self.global_map_width = int(global_width / self.resolution)
        self.global_map: OccupancyGridMap = OccupancyGridMap(self.global_map_width, self.global_map_height, exploration_setting="8N")
        self.global_map_origin: Tuple[float, float] = (new_bottom_right_x, new_bottom_right_y)

    def initialize_d_star_lite(self) -> None:
        # initialize rhs and g arrays
        self.rhs: np.ndarray[float] = np.inf * np.ones((self.global_map_width, self.global_map_height))
        self.g: np.ndarray[float] = self.rhs.copy()
        self.g_distance: np.ndarray[float] = self.g.copy()
        self.k_m = 0.0

        self.rhs[self.s_goal] = 0.0
        self.U.insert(self.s_goal, Priority(heuristic(self.s_start, self.s_goal), 0.0))

    def calculate_key(self, s: Tuple[int, int]) -> Tuple[float, float]:
        # TODO: FIX THE COST TO MATCH A* COST
        """Calculates the key for a vertex"""
        k1 = min(self.g[s], self.rhs[s]) + heuristic(self.s_start, s) + self.k_m
        k2 = min(self.g[s], self.rhs[s])
        return (k1, k2)
    
    def c(self, u: Tuple[int, int], v: Tuple[int, int]) -> float:
        """
        calcuclate the cost between nodes
        :param u: from vertex
        :param v: to vertex
        :return: euclidean distance to traverse. inf if obstacle in path
        """
        # TODO: FIX THE COST TO MATCH A* COST
        if not self.global_map.is_unoccupied(u) or not self.global_map.is_unoccupied(v):
            return float('inf')
        else:
            return heuristic(u, v)
        
    def contain(self, u: Tuple[int, int]) -> Tuple[int, int]:
        return u in self.U.vertices_in_heap
    
    def update_vertex(self, u: Tuple[int, int]):
        # TODO: FIX THE COST TO MATCH A* COST
        if self.g[u] != self.rhs[u] and self.contain(u):
            self.U.update(u, self.calculate_key(u))
        elif self.g[u] != self.rhs[u] and not self.contain(u):
            self.U.insert(u, self.calculate_key(u))
        elif self.g[u] == self.rhs[u] and self.contain(u):
            self.U.remove(u)

    def compute_shortest_path(self):
        # TODO: FIX THE COST TO MATCH A* COST
        while self.U.top_key() < self.calculate_key(self.s_start) or self.rhs[self.s_start] > self.g[self.s_start]:
            u = self.U.top()
            k_old = self.U.top_key()
            k_new = self.calculate_key(u)

            if k_old < k_new:
                self.U.update(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.U.remove(u)
                pred = self.global_map.succ(vertex=u)
                for s in pred:
                    if s != self.s_goal:
                        self.rhs[s] = min(self.rhs[s], self.c(s, u) + self.g[u])
                    self.update_vertex(s)
            else:
                self.g_old = self.g[u]
                self.g[u] = float('inf')
                pred = self.global_map.succ(vertex=u)
                pred.append(u)
                for s in pred:
                    if self.rhs[s] == self.c(s, u) + self.g_old:
                        if s != self.s_goal:
                            min_s = float('inf')
                            succ = self.global_map.succ(vertex=s)
                            for s_ in succ:
                                temp = self.c(s, s_) + self.g[s_]
                                if min_s > temp:
                                    min_s = temp
                            self.rhs[s] = min_s
                    self.update_vertex(u)

    def rescan(self) -> Vertices:
        new_edges_and_old_costs = self.new_edges_and_old_costs
        self.new_edges_and_old_costs = None
        return new_edges_and_old_costs

    def move_and_replan(self, robot_position: Tuple[int, int]):
        # TODO: FIX THE COST TO MATCH A* COST
        path: List[Tuple[int, int]] = [robot_position]
        self.s_start: Tuple[int, int] = robot_position
        self.s_last: Tuple[int, int] = robot_position
        self.compute_shortest_path()

        while self.s_start != self.s_goal:
            assert (self.rhs[self.s_start] != float('inf')), "There is no known path!"

            succ = self.global_map.succ(self.s_start, avoid_obstacles=False)
            min_s = float('inf')
            arg_min = None
            for s_ in succ:
                temp = self.c(self.s_start, s_) + self.g[s_]
                if temp < min_s:
                    min_s = temp
                    arg_min = s_

            ### algorithm sometimes gets stuck here for some reason !!! FIX
            self.s_start = arg_min
            path.append(self.s_start)
            # scan graph for changed costs
            changed_edges_with_old_cost: Vertices = self.rescan()
            # if any edge costs changed
            if changed_edges_with_old_cost:
                self.k_m += heuristic(self.s_last, self.s_start)
                self.s_last = self.s_start

                # for all directed edges (u,v) with changed edge costs
                vertices = changed_edges_with_old_cost.vertices

                vertex: Vertex
                for vertex in vertices:
                    v: Tuple[int, int] = vertex.pos
                    succ_v: Dict[Tuple[int, int], float] = vertex.edges_and_c_old
                    for u, c_old in succ_v.items():
                        c_new = self.c(u, v)
                        if c_old > c_new:
                            if u != self.s_goal:
                                self.rhs[u] = min(self.rhs[u], self.c(u, v) + self.g[v])
                        elif self.rhs[u] == c_old + self.g[v]:
                            if u != self.s_goal:
                                min_s = float('inf')
                                succ_u = self.global_map.succ(vertex=u)
                                for s_ in succ_u:
                                    temp = self.c(u, s_) + self.g[s_]
                                    if min_s > temp:
                                        min_s = temp
                                self.rhs[u] = min_s
                            self.update_vertex(u)
            self.compute_shortest_path()
        print("path found!")
        return path, self.g, self.rhs