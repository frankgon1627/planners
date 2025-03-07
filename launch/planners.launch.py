from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='planners',
            executable='mpc_planner_corridors',
            name='mpc_planner_corridors'
        ),
        Node(
            package='planners',
            executable='occupancy_grid_parser.py',
            name='occupancy_grid_parser',
            shell=True
        ),
        Node(
            package='planners',
            executable='jump_point_planner.py',
            name='jump_point_planner',
            shell=True
        )
    ])