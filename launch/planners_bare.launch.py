from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='planners',
            executable='mpc_planner_bare',
            name='mpc_planner_bare'
        ),
        Node(
            package='planners',
            executable='a_star.py',
            name='a_star',
            shell=True
        )
    ])
