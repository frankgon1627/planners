#!/usr/bin/env python3

import cv2
import math
import rclpy
import heapq
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import tf2_ros
import tf2_geometry_msgs

from typing import Dict, Set, Tuple, List

SQRT2 = 2**0.5

class JPSPlanner(Node):
    def __init__(self) -> None:
        super().__init__('jps_planner')
        self.create_subscription(Odometry, '/dlio/odom_node/odom', self.odometry_callback, 10)
        self.create_subscription(OccupancyGrid, '/obstacle_detection/positive_obstacle_grid', self.occupancy_grid_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.timer = self.create_timer(0.1, self.generate_trajectory)
        self.path_publisher = self.create_publisher(Path, '/planners/jump_point_path', 10)
        self.dialted_occupancy_grid_publisher = self.create_publisher(OccupancyGrid, '/planners/dialted_occupancy_grid', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.odometry: Odometry | None = None
        self.occupancy_grid: OccupancyGrid | None = None
        self.resolution: float | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.resolution: float | None = None

        self.origin: Tuple[float, float] | None = None
        self.goal: PoseStamped | None = None

        self.dialation: float = 0.75

        self.get_logger().info("JPS Planner Node Initialized")

    def odometry_callback(self, msg: Odometry) -> None:
        """Processes Odometry messages from ROS 2"""
        self.odometry = msg

    def occupancy_grid_callback(self, msg: OccupancyGrid) -> None:
        """Processes OccupancyGrid messages from ROS 2"""
        self.occupancy_grid: OccupancyGrid = msg
        self.width: int = msg.info.width
        self.height: int = msg.info.height
        self.resolution: float = msg.info.resolution
        self.origin: Tuple[float] = (msg.info.origin.position.x, msg.info.origin.position.y)
        data: np.ndarray[float] = np.array(msg.data).reshape((self.height, self.width))

        expansion_pixels: int = int(np.ceil(self.dialation / self.resolution))
        obstacle_mask: np.ndarray = (data == 100).astype(np.uint8)
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expansion_pixels + 1, 2 * expansion_pixels + 1))
        dilated_mask: np.ndarray = cv2.dilate(obstacle_mask, kernel)
        self.data: np.ndarray = data.copy()
        self.data[dilated_mask == 1] = 100

        self.dialated_grid: OccupancyGrid = OccupancyGrid()
        self.dialated_grid.header = self.occupancy_grid.header
        self.dialated_grid.info = self.occupancy_grid.info
        self.dialated_grid.data = self.data.astype(np.int8).flatten().tolist()
        self.dialated_grid.info.origin = self.occupancy_grid.info.origin
        self.dialted_occupancy_grid_publisher.publish(self.dialated_grid)

    def meters_to_grid(self, x_meters: float, y_meters: float) -> Tuple[int, int]:
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
        if self.goal.header.frame_id != "map":
            in_map_frame = False
            while not in_map_frame:
                try:
                    transform = self.tf_buffer.lookup_transform('map', self.goal.header.frame_id, rclpy.time.Time())
                    self.goal = tf2_geometry_msgs.do_transform_pose_stamped(self.goal, transform)
                    self.get_logger().info(f"Converted Goal to: {self.goal.pose.position}")
                    in_map_frame = True
                except:
                    self.get_logger().warn("Failed to get Transform from base_link to map")

    def generate_trajectory(self) -> None:
        """Receives goal position and triggers path planning"""
        if self.occupancy_grid is None:
            self.get_logger().warn("No Occupancy Grid Received yet.")
            return
        
        if self.odometry is None:
            self.get_logger().warn("No Odometry Received yet.")
            return
        
        if self.goal is None:
            self.get_logger().warn("No Goal Pose Received yet.")
            return
        
        # convert start to grid coordinates
        self.get_logger().info(f"Odometry: {self.odometry.pose.pose.position}")
        sx, sy = self.meters_to_grid(self.odometry.pose.pose.position.x, self.odometry.pose.pose.position.y)

        # Convert goal to grid coordinates
        gx, gy = self.meters_to_grid(self.goal.pose.position.x, self.goal.pose.position.y)

        start: Tuple[int, int] = (sy, sx)
        goal: Tuple[int, int] = (gy, gx)

        # Run JPS Algorithm
        path = self.run_jps(start, goal)
        self.get_logger().info(f"Path: {path}")

        # Publish path if found
        if path:
            self.publish_path(path)

    def blocked(self, cX: int, cY: int, dX: int, dY: int) -> bool:
        if cX + dX < 0 or cX + dX >= self.data.shape[0]:
            return True
        if cY + dY < 0 or cY + dY >= self.data.shape[1]:
            return True
        if dX != 0 and dY != 0:
            if self.data[cX + dX, cY] == 100 and self.data[cX, cY + dY] == 100:
                return True
            if self.data[cX + dX, cY + dY] == 100:
                return True
        else:
            if dX != 0:
                if self.data[cX + dX, cY] == 100:
                    return True
            else:
                if self.data[cX, cY + dY] == 100:
                    return True
        return False


    def dblock(self, cX: int, cY: int, dX: int, dY: int) -> bool:
        if self.data[cX - dX, cY] == 100 and self.data[cX, cY - dY] == 100:
            return True
        else:
            return False

    def direction(self, cX: int, cY: int, pX: int, pY: int) -> Tuple[int, int]:
        dX: int = int(math.copysign(1, cX - pX))
        dY: int = int(math.copysign(1, cY - pY))
        if cX - pX == 0:
            dX = 0
        if cY - pY == 0:
            dY = 0
        return (dX, dY)

    def nodeNeighbours(self, cX: int, cY: int, parent: Tuple[int, int]) -> List[Tuple[int, int]]:
        neighbours: List[Tuple[int, int]] = []
        if type(parent) != tuple:
            for i, j in [
                (-1, 0),
                (0, -1),
                (1, 0),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                if not self.blocked(cX, cY, i, j):
                    neighbours.append((cX + i, cY + j))

            return neighbours
        dX, dY = self.direction(cX, cY, parent[0], parent[1])

        if dX != 0 and dY != 0:
            if not self.blocked(cX, cY, 0, dY):
                neighbours.append((cX, cY + dY))
            if not self.blocked(cX, cY, dX, 0):
                neighbours.append((cX + dX, cY))
            if (
                not self.blocked(cX, cY, 0, dY)
                or not self.blocked(cX, cY, dX, 0)
            ) and not self.blocked(cX, cY, dX, dY):
                neighbours.append((cX + dX, cY + dY))
            if self.blocked(cX, cY, -dX, 0) and not self.blocked(
                cX, cY, 0, dY
            ):
                neighbours.append((cX - dX, cY + dY))
            if self.blocked(cX, cY, 0, -dY) and not self.blocked(
                cX, cY, dX, 0
            ):
                neighbours.append((cX + dX, cY - dY))

        else:
            if dX == 0:
                if not self.blocked(cX, cY, dX, 0):
                    if not self.blocked(cX, cY, 0, dY):
                        neighbours.append((cX, cY + dY))
                    if self.blocked(cX, cY, 1, 0):
                        neighbours.append((cX + 1, cY + dY))
                    if self.blocked(cX, cY, -1, 0):
                        neighbours.append((cX - 1, cY + dY))

            else:
                if not self.blocked(cX, cY, dX, 0):
                    if not self.blocked(cX, cY, dX, 0):
                        neighbours.append((cX + dX, cY))
                    if self.blocked(cX, cY, 0, 1):
                        neighbours.append((cX + dX, cY + 1))
                    if self.blocked(cX, cY, 0, -1):
                        neighbours.append((cX + dX, cY - 1))
        return neighbours

    def jump(self, cX: int, cY: int, dX: int, dY: int, goal: Tuple[int, int]) -> Tuple[int, int]:

        nX: int = cX + dX
        nY: int = cY + dY
        if self.blocked(nX, nY, 0, 0):
            return None

        if (nX, nY) == goal:
            return (nX, nY)

        oX: int = nX
        oY: int = nY

        if dX != 0 and dY != 0:
            while True:
                if (
                    not self.blocked(oX, oY, -dX, dY)
                    and self.blocked(oX, oY, -dX, 0)
                    or not self.blocked(oX, oY, dX, -dY)
                    and self.blocked(oX, oY, 0, -dY)
                ):
                    return (oX, oY)

                if (
                    self.jump(oX, oY, dX, 0, goal) != None
                    or self.jump(oX, oY, 0, dY, goal) != None
                ):
                    return (oX, oY)

                oX += dX
                oY += dY

                if self.blocked(oX, oY, 0, 0):
                    return None

                if self.dblock(oX, oY, dX, dY):
                    return None

                if (oX, oY) == goal:
                    return (oX, oY)
        else:
            if dX != 0:
                while True:
                    if (
                        not self.blocked(oX, nY, dX, 1)
                        and self.blocked(oX, nY, 0, 1)
                        or not self.blocked(oX, nY, dX, -1)
                        and self.blocked(oX, nY, 0, -1)
                    ):
                        return (oX, nY)

                    oX += dX

                    if self.blocked(oX, nY, 0, 0):
                        return None

                    if (oX, nY) == goal:
                        return (oX, nY)

            else:
                while True:
                    if (
                        not self.blocked(nX, oY, 1, dY)
                        and self.blocked(nX, oY, 1, 0)
                        or not self.blocked(nX, oY, -1, dY)
                        and self.blocked(nX, oY, -1, 0)
                    ):
                        return (nX, oY)

                    oY += dY

                    if self.blocked(nX, oY, 0, 0):
                        return None

                    if (nX, oY) == goal:
                        return (nX, oY)

        return jump(nX, nY, dX, dY, goal)

    def identifySuccessors(self, cX: int, cY: int, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                            goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        successors = []
        neighbours = self.nodeNeighbours(cX, cY, came_from.get((cX, cY), 0))

        for cell in neighbours:
            dX = cell[0] - cX
            dY = cell[1] - cY

            jumpPoint: Tuple[int, int] = self.jump(cX, cY, dX, dY, goal)

            if jumpPoint != None:
                successors.append(jumpPoint)

        return successors

    def run_jps(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Executes Jump Point Search Algorithm"""
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        close_set: Set[Tuple[int, int]] = set()
        gscore: Dict[Tuple[int, int], int] = {start: 0}
        fscore: Dict[Tuple[int, int], float] = {start: self.heuristic(start, goal)}
        pqueue: List[float, Tuple[int, int]] = []
        heapq.heappush(pqueue, (fscore[start], start))

        while pqueue:
            current: Tuple[int, int] = heapq.heappop(pqueue)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path = path[::-1]
                return path
            
            close_set.add(current)
            successors = self.identifySuccessors(current[0], current[1], came_from, goal)

            for successor in successors:
                jumpPoint = successor

                if jumpPoint in close_set:  # and tentative_g_score >= gscore.get(jumpPoint,0):
                    continue

                tentative_g_score = gscore[current] + self.heuristic(current, jumpPoint)

                if tentative_g_score < gscore.get(jumpPoint, 0) or jumpPoint not in [j[1] for j in pqueue]:
                    came_from[jumpPoint] = current
                    gscore[jumpPoint] = tentative_g_score
                    fscore[jumpPoint] = tentative_g_score + self.heuristic(jumpPoint, goal)
                    heapq.heappush(pqueue, (fscore[jumpPoint], jumpPoint))
        return []

    def publish_path(self, path: List[Tuple[int, int]]) -> None:
        """Publishes the computed path as a ROS 2 Path message"""
        path_msg: Path = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for (gy, gx) in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = gx * self.resolution + self.origin[0]
            pose.pose.position.y = gy * self.resolution + self.origin[1]
            path_msg.poses.append(pose)

        self.path_publisher.publish(path_msg)
        self.get_logger().info("Published planned path.")

    def heuristic(self, node: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Euclidean heuristic for A*"""
        return ((node[0] - goal[0])**2 + (node[1] - goal[1])**2)**0.5


def main():
    rclpy.init()
    node: JPSPlanner = JPSPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
