#!/usr/bin/env python3

import cv2
import rclpy
import numpy as np

from rclpy.publisher import Publisher
from rclpy.node import Node
from nav_msgs.msg import MapMetaData, OccupancyGrid
from geometry_msgs.msg import Point, Polygon, Point32
from custom_msgs_pkg.msg import PolygonArray
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from typing import List, Tuple

class OccupancyGridParser(Node):
    def __init__(self) -> None:
        super().__init__('convex_hull_extractor')

        self.convex_hull_viz_publisher: Publisher[MarkerArray] = self.create_publisher(MarkerArray, '/convex_hulls_viz', 1)
        self.convex_hull_publisher: Publisher[PolygonArray] = self.create_publisher(PolygonArray, '/convex_hulls', 1)
        self.dialted_occupancy_grid_publisher = self.create_publisher(OccupancyGrid, '/planners/dialted_occupancy_grid', 10)

        # self.create_subscription(OccupancyGrid, '/cost_map', self.occupancy_grid_callback, 10)
        self.create_subscription(OccupancyGrid, '/planners/dialted_occupancy_grid', self.occupancy_grid_callback, 10)

        self.occupancy_grid: OccupancyGrid | None = None
        self.hulls: List[np.ndarray[float]] | None = None

        self.get_logger().info("Occupancy Grid Parser Node Initialized")

    def occupancy_grid_callback(self, msg: OccupancyGrid) -> None:
        """
        Dialates Occupancy grid and generates convex hulls around the obstacles in the occupancy grid.
        """
        self.occupancy_grid: OccupancyGrid = msg
        self.width: int = msg.info.width
        self.height: int = msg.info.height
        self.resolution: float = msg.info.resolution
        self.origin: Tuple[float] = (msg.info.origin.position.x, msg.info.origin.position.y)
        data: np.ndarray[float] = np.array(msg.data).reshape((self.height, self.width))

        # dialate the occupancy grid data
        expansion_pixels: int = int(np.ceil(self.dialation / self.resolution))
        obstacle_mask: np.ndarray = (data == 100).astype(np.uint8)
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expansion_pixels + 1, 2 * expansion_pixels + 1))
        dilated_mask: np.ndarray = cv2.dilate(obstacle_mask, kernel)
        self.data: np.ndarray = data.copy()
        self.data[dilated_mask == 1] = 100

        # create and publish the dialted occupancy grid
        self.dialated_grid: OccupancyGrid = OccupancyGrid()
        self.dialated_grid.header = self.occupancy_grid.header
        self.dialated_grid.info = self.occupancy_grid.info
        self.dialated_grid.data = self.data.astype(np.int8).flatten().tolist()
        self.dialated_grid.info.origin = self.occupancy_grid.info.origin
        self.dialted_occupancy_grid_publisher.publish(self.dialated_grid)

        # Extract occupied cells
        occupied_points: List[Tuple[float]] = []
        for i in range(self.height):
            for j in range(self.width):
                if data[i, j] == 100:  # Threshold for occupancy
                    x: float = self.origin[0] + j*self.resolution
                    y: float = self.origin[1] + i*self.resolution
                    occupied_points.append((x, y))

        # Group occupied cells into clusters (simple grid-based clustering)
        occupied_points: np.ndarray[float] = np.array(occupied_points)
        if len(occupied_points) == 0:
            return []

        clusters: List[np.ndarray[float]] = self.cluster_points(occupied_points, self.resolution)
        self.hulls = [self.compute_convex_hull(cluster) for cluster in clusters]

        self.get_logger().info(f"Generated {len(self.hulls)} convex hull obstacles.")
        self.publish_convex_hulls()

    def cluster_points(self, points: np.ndarray[float], resolution: float) -> List[np.ndarray[float]]:
        """
        Clusters points based on proximity using DBSCAN.
        """
        clustering: DBSCAN = DBSCAN(eps=3*resolution, min_samples=2).fit(points)
        labels: List[int] = clustering.labels_

        clusters: List[np.ndarray[float]] = []
        for label in set(labels):
            if label != -1:  # -1 indicates noise in DBSCAN
                clusters.append(points[labels == label])
        return clusters

    def compute_convex_hull(self, cluster: np.ndarray[float]) -> np.ndarray[float]:
        """
        Computes the convex hull of a cluster of points. Each hull is represented by its vertices
        in a counterclockwise order
        """
        # cannot make a hull out of a cluster of two points
        if len(cluster) < 3:
            return cluster  
        hull: ConvexHull = ConvexHull(cluster)
        return cluster[hull.vertices]
    
    def publish_convex_hulls(self) -> None:
        """
        Publishes the convex hulls to both RVIZ and as a polygon array
        """
        convex_hulls: PolygonArray = PolygonArray()
        marker_array: MarkerArray = MarkerArray()

        for i, hull in enumerate(self.hulls):
            # data structure for publishing to other nodes
            polygon: Polygon = Polygon()

            # data structure for RVIZ visualization
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "convex_hulls"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # iterate over each point in the hull and add to both topics
            for row in hull:
                point32: Point32 = Point32()
                point32.x = row[0]
                point32.y = row[1]
                polygon.points.append(point32)
            
                point: Point = Point()
                point.x = row[0]
                point.y = row[1]
                point.z = 0.05
                marker.points.append(point)
            convex_hulls.polygons.append(polygon)
            
            # close the hull for visualization
            if len(hull) > 0:
                marker.points.append(Point(x=hull[0][0], y=hull[0][1], z=0.05))

            # Set marker properties
            marker.scale.x = 0.1  
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  

            marker_array.markers.append(marker) 

        self.convex_hull_viz_publisher.publish(marker_array)
        self.convex_hull_publisher.publish(convex_hulls)

def main(args=None):
    rclpy.init(args=args)
    mpc_planner: OccupancyGridParser = OccupancyGridParser()
    rclpy.spin(mpc_planner)
    mpc_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
