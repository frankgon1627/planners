#!/usr/bin/env python3
import rclpy
import numpy as np
from heapq import heappush, heappop
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from obstacle_detection_msgs.msg import RiskMap
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from typing import Dict, Tuple, List
import math
import time

class AStarPlanner(Node):
    def __init__(self) -> None:
        super().__init__('a_star_planner')
        self.create_subscription(Odometry, '/dlio/odom_node/odom', self.odometry_callback, 10)
        self.create_subscription(RiskMap, '/obstacle_detection/combined_map', self.occupancy_grid_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.global_map_publisher = self.create_publisher(RiskMap, '/planners/a_star_map', 1)
        self.global_map_rviz_publisher = self.create_publisher(OccupancyGrid, '/planners/a_star_map_rviz', 1)
        self.path_publisher = self.create_publisher(Path, '/planners/a_star_path', 10)
        self.sparse_path_publisher = self.create_publisher(Path, '/planners/sparse_a_star_path', 10)

        self.timer = self.create_timer(1,0, self.generate_trajectory)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.odometry: Odometry | None = None
        self.local_map: RiskMap | None = None
        self.local_map_width: int | None = None
        self.local_map_height: int | None = None
        self.local_map_origin: Tuple[float] | None = None
        self.local_map_data: np.ndarray[float] | None = None
        self.resolution: float | None = None
        self.risk_factor: float = 1.0

        self.goal: PoseStamped | None = None
        self.new_goal_received: bool = False

        # global map that accumulates all the observations
        self.global_map: RiskMap | None = None
        self.global_map_origin: Tuple[float, float] | None = None
        self.global_map_height: int | None = None
        self.global_map_width: int | None = None
        self.global_map_data: np.ndarray[float] | None = None

        self.directions: List[Tuple[int]] = [
            (1, 0), (-1, 0), (0, 1), (0, -1), 
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        self.get_logger().info("A* Planner Node Initialized")

    def odometry_callback(self, msg: Odometry) -> None:
        """Processes Odometry messages from ROS 2"""
        self.odometry = msg
        
    def occupancy_grid_callback(self, msg: RiskMap) -> None:
        """Processes CombinedRiskMap messages from ROS 2"""
        self.local_map: RiskMap = msg
        self.local_map_width: int = msg.info.width
        self.local_map_height: int = msg.info.height
        self.local_map_origin: Tuple[float] = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.local_map_data: np.ndarray[float] = np.array(msg.data).reshape((self.local_map_height, self.local_map_width))
        self.resolution: float = msg.info.resolution

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
        if self.local_map is None:
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
            self.new_goal_received = False

        # set start position and convert start and goal to grid coordinates
        global_start_i: int = int((self.odometry.pose.pose.position.x - self.global_map_origin[0])/self.resolution)
        global_start_j: int = int((self.odometry.pose.pose.position.y - self.global_map_origin[1])/self.resolution)

        global_goal_i: int = int((self.goal.pose.position.x - self.global_map_origin[0])/self.resolution)
        global_goal_j: int = int((self.goal.pose.position.y - self.global_map_origin[1])/self.resolution)
        start: Tuple[int, int] = (global_start_j, global_start_i)
        goal: Tuple[int, int] = (global_goal_j, global_goal_i)

        # extract the information from the local map
        local_data: np.ndarray[float] = self.local_map_data
        nodes: Dict[Tuple[int, int], float] = {}
        for s_i in range(self.local_map_width):
            for s_j in range(self.local_map_height):
                value: float = local_data[s_j, s_i]
                x_position: float = s_i * self.resolution + self.local_map_origin[0]
                y_position: float = s_j * self.resolution + self.local_map_origin[1]
                # convert to global map index coordinates
                global_cell_i: int = int((x_position - self.global_map_origin[0]) / self.resolution)
                global_cell_j: int = int((y_position - self.global_map_origin[1]) / self.resolution)

                # only update readings that lie within the global map
                if self.is_valid(global_cell_j, global_cell_i):
                    nodes[(global_cell_j, global_cell_i)] = value

        # update the global map with the new readings
        for node, value in nodes.items():
            if self.global_map_data[node] != value:
                self.global_map_data[node] = value

        dense_path: List[Tuple[int, int]] = self.a_star(start, goal)
        sparse_path: np.ndarray[float] = self.douglas_peucker(dense_path, 0.35)
        self.publish_path(dense_path, sparse=False)
        self.publish_path(sparse_path, sparse=True)

        if dense_path:
            self.get_logger().info("Published Path")
        else:
            self.get_logger().warn("No Path Found")

        # publish the global map
        self.global_map.header.frame_id = "odom"
        self.global_map.header.stamp = self.get_clock().now().to_msg()
        self.global_map.info.origin.position.z = self.odometry.pose.pose.position.z
        self.global_map.data = self.global_map_data.flatten().tolist()
        self.global_map_publisher.publish(self.global_map)
        # publish the global map for RViz visualization
        occupancy_grid_msg: OccupancyGrid = OccupancyGrid()
        occupancy_grid_msg.header.frame_id = "odom"
        occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid_msg.info.resolution = self.global_map.info.resolution
        occupancy_grid_msg.info.width = self.global_map.info.width
        occupancy_grid_msg.info.height = self.global_map.info.height
        occupancy_grid_msg.info.origin.position.x = self.global_map.info.origin.position.x
        occupancy_grid_msg.info.origin.position.y = self.global_map.info.origin.position.y
        occupancy_grid_msg.info.origin.position.z = self.global_map.info.origin.position.z
        occupancy_grid_msg.data = self.global_map_data.astype(np.int8).flatten().tolist()
        self.global_map_rviz_publisher.publish(occupancy_grid_msg)

    def initialize_global_map(self) -> None:
        # initialize all variables relating to the global map
        original_bottom_right_x = self.local_map.info.origin.position.x
        original_bottom_right_y = self.local_map.info.origin.position.y
        original_top_left_x = original_bottom_right_x + self.local_map_width * self.resolution
        original_top_left_y = original_bottom_right_y + self.local_map_height * self.resolution

        # assume the goal location needs to call in the center of a map the same size as combined_risk_map_
        # determine the bottom right and top left of a map centered at goal
        goal_bottom_right_x = self.goal.pose.position.x - self.local_map_width * self.resolution / 2.0
        goal_bottom_right_y = self.goal.pose.position.y - self.local_map_height * self.resolution / 2.0
        goal_top_left_x = self.goal.pose.position.x + self.local_map_width * self.resolution / 2.0
        goal_top_left_y = self.goal.pose.position.y + self.local_map_height * self.resolution / 2.0

        # determine the boundaries of the global map, such that the global map minimally encompases the 
        # desired thresholds around the goal position and the original combined_risk_map_
        new_bottom_right_x = min(original_bottom_right_x, goal_bottom_right_x)
        new_bottom_right_y = min(original_bottom_right_y, goal_bottom_right_y)
        new_top_left_x = max(original_top_left_x, goal_top_left_x)
        new_top_left_y = max(original_top_left_y, goal_top_left_y)
        self.global_map_origin: Tuple[float, float] = (new_bottom_right_x, new_bottom_right_y)

        # construct an empty global planning grid
        global_height = new_top_left_y - new_bottom_right_y
        global_width = new_top_left_x - new_bottom_right_x
        self.global_map_height = math.ceil(global_height / self.resolution)
        self.global_map_width = math.ceil(global_width / self.resolution)
        self.global_map: RiskMap = RiskMap()
        self.global_map.info.resolution = self.resolution
        self.global_map.info.width = self.global_map_width
        self.global_map.info.height = self.global_map_height
        self.global_map.info.origin.position.x = new_bottom_right_x
        self.global_map.info.origin.position.y = new_bottom_right_y
        self.global_map.info.origin.position.z = self.odometry.pose.pose.position.z
        self.global_map_data: np.ndarray[float] = np.zeros((self.global_map_height, self.global_map_width), dtype=np.float32)

    def is_valid(self, y: int, x: int) -> bool:
        """Checks if a position is within bounds and not an obstacle"""
        if 0 <= y < self.global_map_height and 0 <= x < self.global_map_width:
            return self.global_map_data[y, x] != 100
        return False

    def get_neighbors(self, current_node: Tuple[int]) -> List[Tuple[int, int]]:
        """Get the neighbors of the current cell"""
        y, x = current_node
        neighbors = []
        for direction in self.directions:
            new_y: int = y + direction[0]
            new_x: int = x + direction[1]

            if self.is_valid(new_y, new_x):
                neighbors.append((new_y, new_x))
        return neighbors

    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """Executes A Star Search Algorithm"""
        priority_queue: List[Tuple[int, Tuple[int, int]]] = [(0, start)]
        parents: Dict[Tuple[int, int]] = {}
        seen = set()
        distance_cost = {start: 0}
        cost = {start: 0}

        while priority_queue:
            _, location = heappop(priority_queue)

            if location == goal:
                path = self.reconstruct_path(parents, location)
                return path
            
            for neighbor in self.get_neighbors(location):
                if neighbor not in seen:
                    # update cost to include distance to neighbor
                    new_distance_cost: float = distance_cost[location] + ((location[0] - neighbor[0])**2 + (location[1] - neighbor[1])**2)**0.5
                    new_cost: float = new_distance_cost * (1 + self.risk_factor*self.global_map_data[neighbor[0], neighbor[1]])
                    # perform update if we have not seen this node or if we found a shorter path to this node
                    if neighbor not in cost or new_cost < cost[neighbor]:
                        cost[neighbor] = new_cost
                        distance_cost[neighbor] = new_distance_cost
                        # add distance to goal if we want to run astar
                        heuristic = new_cost + ((goal[0] - neighbor[0])**2 + (goal[1] - neighbor[1])**2)**0.5
                        # add the neighbor into the queue based on the heuristic value
                        heappush(priority_queue, (heuristic, neighbor))
                        # assign the parent pointer
                        parents[neighbor] = location
        self.get_logger().warn("No path found.")
        return []

    def reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstructs the path from the goal to the start"""
        path: List[Tuple[int, int]] = []
        current = goal
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]
    
    def douglas_peucker(self, path: List[Tuple[int, int]], epsilon: float) -> np.ndarray:
        """
        Creates a sparse representation of a path by keeping the middle intermediate
        point when looking at a overarching line segment
        """
        if path is None or len(path) < 3:
            return path
        
        # convert to numpy array
        path = np.array(path)
        
        d_max = 0
        max_index = 0

        # initial line vector
        line_vector = path[-1] - path[0]
        # determine maximum distance
        for i in range(1, path.shape[0]):
            # distance from point to line
            dist = np.linalg.norm(np.cross(line_vector, path[-1] - path[i])) / np.linalg.norm(line_vector)
            if dist > d_max:
                max_index = i
                d_max = dist

        # found far off point, recurse on left and right subpaths
        if d_max > epsilon:
            left_branch = self.douglas_peucker(path[:max_index+1], epsilon)
            right_branch = self.douglas_peucker(path[max_index:], epsilon)
            sparse_path = np.vstack([left_branch[:-1], right_branch])
        # no need to include intermediate points
        else:
            sparse_path = np.vstack([path[0], path[-1]])
        return sparse_path

    def publish_path(self, path: List[Tuple[int, int]] | None, sparse: bool = False) -> None:
        """Publishes the computed path as a ROS 2 Path message"""
        path_msg: Path = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        if path is not None:
            for (gy, gx) in path:
                pose = PoseStamped()
                pose.header.frame_id = "odom"
                pose.pose.position.x = gx * self.resolution + self.global_map_origin[0]
                pose.pose.position.y = gy * self.resolution + self.global_map_origin[1]
                pose.pose.position.z = self.odometry.pose.pose.position.z
                path_msg.poses.append(pose)
        
        if sparse:
            self.sparse_path_publisher.publish(path_msg)
        else:
            self.path_publisher.publish(path_msg)

    def heuristic(self, node: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Euclidean heuristic for A*"""
        return ((node[0] - goal[0])**2 + (node[1] - goal[1])**2)**0.5

def main():
    rclpy.init()
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()