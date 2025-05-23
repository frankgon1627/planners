#!/usr/bin/env python3
import rclpy
import numpy as np
from heapq import heappush, heappop
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from typing import Dict, Tuple, List

class AStarPlanner(Node):
    def __init__(self) -> None:
        super().__init__('a_star_planner')
        self.create_subscription(Odometry, '/dlio/odom_node/odom', self.odometry_callback, 10)
        self.create_subscription(OccupancyGrid, '/obstacle_detection/combined_map', self.occupancy_grid_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.path_publisher = self.create_publisher(Path, '/planners/a_star_path', 10)
        self.sparse_path_publisher = self.create_publisher(Path, '/planners/sparse_a_star_path', 10)

        self.timer = self.create_timer(0.1, self.generate_trajectory)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.odometry: Odometry | None = None
        self.occupancy_grid: OccupancyGrid | None = None
        self.resolution: float | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.resolution: float | None = None
        self.origin: Tuple[float] | None = None
        self.goal: PoseStamped | None = None
        self.get_logger().info("A* Planner Node Initialized")

        self.directions: List[Tuple[int]] = [
            (1, 0), (-1, 0), (0, 1), (0, -1), 
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

    def odometry_callback(self, msg: Odometry) -> None:
        """Processes Odometry messages from ROS 2"""
        self.odometry = msg
        
    def occupancy_grid_callback(self, msg: OccupancyGrid) -> None:
        """Processes OccupancyGrid messages from ROS 2"""
        self.occupancy_grid: OccupancyGrid = msg
        self.width: int = msg.info.width
        self.height: int = msg.info.height
        self.resolution: float = msg.info.resolution
        self.origin: List[float] = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.data: np.ndarray[float] = np.array(msg.data).reshape((self.height, self.width))

    def meters_to_grid(self, x_meters: float, y_meters: float) -> Tuple[int]:
        """
        Convert points in meter space to points in grid index space
        """
        grid_x: int = int((x_meters - self.origin[0]) / self.resolution)
        grid_y: int = int((y_meters - self.origin[1]) / self.resolution)
        return grid_x, grid_y

    def goal_callback(self, msg: PoseStamped) -> None:
        """Receives goal position"""
        self.goal = msg

        # convert goal to map frame if not in map frame
        if self.goal.header.frame_id != "odom":
            in_map_frame = False
            while not in_map_frame:
                try:
                    transform = self.tf_buffer.lookup_transform('odom', self.goal.header.frame_id, rclpy.time.Time())
                    self.goal = tf2_geometry_msgs.do_transform_pose_stamped(self.goal, transform)
                    self.get_logger().info(f"Converted Goal to: {self.goal.pose.position}")
                    in_map_frame = True
                except:
                    self.get_logger().warn("Failed to get Transform from base_link to map")

    def generate_trajectory(self) -> None:
        if self.occupancy_grid is None:
            self.get_logger().warn("No Occupancy Grid Received yet.")
            return
        
        if self.odometry is None:
            self.get_logger().warn("No Odometry Received yet.")
            return
        
        if self.goal is None:
            self.get_logger().warn("No Goal Received yet.")
            return
        
        # convert start and goal to grid coordinates
        sx, sy = self.meters_to_grid(self.odometry.pose.pose.position.x, self.odometry.pose.pose.position.y)
        gx, gy = self.meters_to_grid(self.goal.pose.position.x, self.goal.pose.position.y)
        start: Tuple[int, int] = (sy, sx)
        goal: Tuple[int, int] = (gy, gx)

        dense_path: List[Tuple[int, int]] = self.a_star(start, goal)
        if dense_path:
            self.publish_path(dense_path, sparse=False)
            sparse_path: np.ndarray[float] = self.douglas_peucker(dense_path, 0.25)
            self.publish_path(sparse_path, sparse=True)

    def is_valid(self, x: int, y: int) -> bool:
        """Checks if a position is within bounds and not an obstacle"""
        if 0 <= x < self.height and 0 <= y < self.width:
            return self.data[x, y] != 100
        return False

    def get_neighbors(self, current_node: Tuple[int]) -> List[Tuple[int, int]]:
        """Get the neighbors of the current cell"""
        x, y = current_node
        neighbors = []
        for direction in self.directions:
            new_x: int = x + direction[0]
            new_y: int = y + direction[1]
            if self.is_valid(new_x, new_y):
                neighbors.append((new_x, new_y))
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
                    new_cost: float = new_distance_cost * (1 + self.data[neighbor[0], neighbor[1]])
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

    def publish_path(self, path: List[Tuple[int, int]], sparse: bool = False) -> None:
        """Publishes the computed path as a ROS 2 Path message"""
        path_msg: Path = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for (gy, gx) in path:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.pose.position.x = gx * self.resolution + self.origin[0]
            pose.pose.position.y = gy * self.resolution + self.origin[1]
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