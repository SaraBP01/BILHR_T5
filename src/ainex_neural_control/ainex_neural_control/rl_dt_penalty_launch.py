#!/usr/bin/env python3

"""
Launch file for RL-DT Penalty Kick Tutorial
File: src/ainex_neural_control/rl_dt_penalty_launch.py (or launch/rl_dt_penalty_launch.py)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Configuration arguments
        DeclareLaunchArgument(
            'max_episodes',
            default_value='100',
            description='Maximum number of episodes to run'
        ),
        
        DeclareLaunchArgument(
            'exploration_mode',
            default_value='true',
            description='Start in exploration mode'
        ),
        
        # Startup log
        LogInfo(msg='Starting RL-DT Penalty Kick Tutorial'),
        
        # Main RL-DT node
        Node(
            package='ainex_neural_control',
            executable='rl_dt_penalty_node',
            name='rl_dt_penalty_node',
            output='screen',
            parameters=[{
                'max_episodes': LaunchConfiguration('max_episodes'),
                'exploration_mode': LaunchConfiguration('exploration_mode'),
            }],
            remappings=[
                ('/joint_states', '/nao_robot/joint_states'),
                ('/joint_commands', '/nao_robot/joint_commands'),
                ('/camera/image_raw', '/nao_robot/camera/image_raw'),
            ]
        ),
        
        # Manual reward control node
        Node(
            package='ainex_neural_control', 
            executable='manual_reward_node',
            name='manual_reward_node',
            output='screen'
        ),
<<<<<<< HEAD
    ])
=======
    ])
>>>>>>> e83f897 (k)
