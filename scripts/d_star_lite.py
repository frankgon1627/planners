#!/usr/bin/env python3
import numpy as np
import tf2_geometry_msgs
import rclpy
import math
import time

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
        self.create_subscription(Odometry, '/dlio/odom_node/odom', self.odometry_callback, 1)
        self.create_subscription(OccupancyGrid, '/obstacle_detection/combined_map', self.occupancy_grid_callback, 1)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 1)
        self.global_map_publisher = self.create_publisher(OccupancyGrid, '/planners/d_star_lite_map', 1)
        self.path_publisher = self.create_publisher(Path, '/planners/d_star_lite_path', 1)
        
        self.create_timer(0.2, self.generate_trajectory)

        self.tf_buffer: Buffer = Buffer()
        self.tf_listener: TransformListener = TransformListener(self.tf_buffer, self)

        self.odometry: Odometry | None = None
        self.goal: PoseStamped | None = None
        self.new_goal_received: bool = False

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
        self.new_edges_and_old_costs: Vertices | None = None

        # D-Star Lite Algorithm variables
        self.U: PriorityQueue | None = None
        self.rhs: np.ndarray[float] | None = None
        self.g: np.ndarray[float] | None = None
        self.k_m: float = 0.0
        self.last_iteration_global_index: Tuple[int, int] | None = None
        self.s_start: Tuple[int, int] | None = None
        self.s_goal: Tuple[int, int] | None = None
        self.s_last: Tuple[int, int] | None = None

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
        self.data: np.ndarray[float] = np.array(msg.data).reshape((self.width, self.height))

    def goal_callback(self, msg: PoseStamped) -> None:
        """Receives goal position and determines start and goal positions in grid coordinates"""
        self.goal = msg

        # convert goal to odom frame if not in odom frame
        if self.goal.header.frame_id != "odom":
            in_odom_frame: bool = False
            while not in_odom_frame:
                try:
                    transform = self.tf_buffer.lookup_transform('odom', self.goal.header.frame_id, rclpy.time.Time())
                    self.goal = tf2_geometry_msgs.do_transform_pose_stamped(self.goal, transform)
                    self.get_logger().info(f"Converted Goal to: {self.goal.pose.position}")
                    in_odom_frame = True
                except:
                    self.get_logger().warn("Failed to get Transform from base_link to map")

        self.new_goal_received = True

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
        
        if self.new_goal_received:
            self.initialize_global_map()

            # set start position and convert start and goal to grid coordinates
            global_start_i: int = int((self.odometry.pose.pose.position.x - self.global_map_origin[0])/self.resolution)
            global_start_j: int = int((self.odometry.pose.pose.position.y - self.global_map_origin[1])/self.resolution)
            self.last_iteration_global_index: Tuple[int, int] = (global_start_j, global_start_i)

            global_goal_i: int = int((self.goal.pose.position.x - self.global_map_origin[0])/self.resolution)
            global_goal_j: int = int((self.goal.pose.position.y - self.global_map_origin[1])/self.resolution)
            self.s_start: Tuple[int, int] = (global_start_j, global_start_i)
            self.s_goal: Tuple[int, int] = (global_goal_j, global_goal_i)

            self.initialize_d_star_lite()
            path, _, _ = self.move_and_replan(self.last_iteration_global_index)

            self.publish_path(path)
            self.new_goal_received = False
            return
        
        self.initialize_d_star_lite()
        global_start_i: int = int((self.odometry.pose.pose.position.x - self.global_map_origin[0])/self.resolution)
        global_start_j: int = int((self.odometry.pose.pose.position.y - self.global_map_origin[1])/self.resolution)
        new_position: Tuple[int, int] = (global_start_j, global_start_i)

        # TODO: CHECK FOR POSITION OR ORIENTATION CHANGE??? Maybe???
        if new_position != self.last_iteration_global_index:
            start_time: float = time.time()
            # extract the information from the local map
            local_data: List[float] = self.combined_grid.data
            nodes: Dict[Tuple[int, int], float] = {}
            for s_i in range(self.width):
                for s_j in range(self.height):
                    value: float = local_data[s_j * self.width + s_i]
                    x_position: float = s_i * self.resolution + self.local_origin[0]
                    y_position: float = s_j * self.resolution + self.local_origin[1]
                    # convert to global map index coordinates
                    global_cell_i: int = int((x_position - self.global_map_origin[0]) / self.resolution)
                    global_cell_j: int = int((y_position - self.global_map_origin[1]) / self.resolution)

                    # only update readings that lie within the global map
                    if self.global_map.in_bounds((global_cell_j, global_cell_i)):
                        nodes[(global_cell_j, global_cell_i)] = value

            # update the global map with the new readings
            vertices: Vertices = Vertices()
            for node, value in nodes.items():
                if self.global_map.occupancy_grid_map[node] != value:
                    v: Vertex = Vertex(pos=node)
                    succ: List[Tuple[int, int]] = self.global_map.succ(node)
                    for u in succ:
                        v.add_edge_with_cost(u, self.c(u, v.pos))
                    vertices.add_vertex(v)
                    # self.global_map.update_cell(node, value)
                    self.global_map.occupancy_grid_map[node] = value
            self.new_edges_and_old_costs = vertices
            self.get_logger().info(f"Updated Global Map in {time.time() - start_time:.2f} seconds")
            path, _, _ = self.move_and_replan(new_position)
            self.publish_path(path)

        # publish the global map
        global_map_msg: OccupancyGrid = OccupancyGrid()
        global_map_msg.header.frame_id = "odom"
        global_map_msg.header.stamp = self.get_clock().now().to_msg()
        global_map_msg.info.resolution = self.resolution
        global_map_msg.info.width = self.global_map_width
        global_map_msg.info.height = self.global_map_height
        global_map_msg.info.origin.position.x = self.global_map_origin[0]
        global_map_msg.info.origin.position.y = self.global_map_origin[1]
        global_map_msg.info.origin.position.z = self.odometry.pose.pose.position.z
        global_map_msg.data = self.global_map.occupancy_grid_map.flatten().tolist()
        self.global_map_publisher.publish(global_map_msg)
            
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
        self.global_map_height = math.ceil(global_height / self.resolution)
        self.global_map_width = math.ceil(global_width / self.resolution)
        self.global_map: OccupancyGridMap = OccupancyGridMap(self.global_map_height, self.global_map_width)
        self.global_map_origin: Tuple[float, float] = (new_bottom_right_x, new_bottom_right_y)

    def initialize_d_star_lite(self) -> None:
        # initialize rhs and g arrays
        self.rhs: np.ndarray[float] = np.inf * np.ones((self.global_map_height, self.global_map_width))
        self.g: np.ndarray[float] = self.rhs.copy()
        self.k_m = 0.0

        self.rhs[self.s_goal] = 0.0
        self.U: PriorityQueue = PriorityQueue()
        self.U.insert(self.s_goal, Priority(heuristic(self.s_start, self.s_goal), 0.0))

    def calculate_key(self, s: Tuple[int, int]) -> Tuple[float, float]:
        # TODO: FIX THE COST TO MATCH A* COST
        """Calculates the key for a vertex"""
        risk_cost: float = 1 + self.global_map.occupancy_grid_map[s]
        k1 = min(self.g[s], self.rhs[s]) + heuristic(self.s_start, s) + self.k_m
        k2 = min(self.g[s], self.rhs[s])
        return Priority(k1, k2)
    
    def c(self, u: Tuple[int, int], v: Tuple[int, int]) -> float:
        """
        calcuclate the cost between nodes
        :param u: from vertex
        :param v: to vertex
        :return: euclidean distance to traverse. inf if obstacle in path
        """
        # TODO: FIX THE COST TO MATCH A* COST
        u_occupied: bool = self.global_map.occupancy_grid_map[u] == 100
        v_occupied: bool = self.global_map.occupancy_grid_map[v] == 100
        if u_occupied or v_occupied:
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
        start_time: float = time.time()
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
                    self.update_vertex(s)
        self.get_logger().info(f"Shortest Path Computed in {time.time() - start_time:.2f} seconds")

    def move_and_replan(self, robot_position: Tuple[int, int]):
        start_time: float = time.time()
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

            # extract the changed costs
            if self.new_edges_and_old_costs is None:
                changed_edges_with_old_cost = Vertices()
            else:
                changed_edges_with_old_cost = self.new_edges_and_old_costs
            self.new_edges_and_old_costs = None

            # if any edge costs changed
            start_time_changed: float = time.time()
            if changed_edges_with_old_cost.vertices:
                self.k_m += heuristic(self.s_last, self.s_start)
                self.s_last = self.s_start

                # for all directed edges (u,v) with changed edge costs
                vertices: List[Vertex] = changed_edges_with_old_cost.vertices

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
                self.get_logger().info(f"Updated changed edges in {time.time() - start_time_changed:.2f} seconds")
                self.compute_shortest_path()
        self.get_logger().info(f"Path found in {time.time() - start_time:.2f} seconds")
        print("path found!")
        return path, self.g, self.rhs
    
    def publish_path(self, path: List[Tuple[int, int]]) -> None:
        """Publishes the computed path as a ROS 2 Path message"""
        path_msg: Path = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for (gy, gx) in path:
            pose: PoseStamped = PoseStamped()
            pose.header.frame_id = "odom"
            pose.pose.position.x = gx * self.resolution + self.global_map_origin[0]
            pose.pose.position.y = gy * self.resolution + self.global_map_origin[1]
            pose.pose.position.z = self.odometry.pose.pose.position.z
            path_msg.poses.append(pose)
        self.path_publisher.publish(path_msg)

def main():
    rclpy.init()
    node: DStarLite = DStarLite()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()