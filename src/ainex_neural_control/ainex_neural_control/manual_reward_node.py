#!/usr/bin/env python3

"""
Node for manual reward input
File: src/ainex_neural_control/manual_reward_node.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
import sys
import select
import tty
import termios

class ManualRewardNode(Node):
    def __init__(self):
        super().__init__('manual_reward_node')
        
        # Publishers
        self.reward_pub = self.create_publisher(Float32, '/penalty_reward', 10)
        self.reset_pub = self.create_publisher(Bool, '/reset_episode', 10)
        
        # Terminal settings for input without pressing enter
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        
        # Timer to check input
        self.timer = self.create_timer(0.1, self.check_input)
        
        self.print_instructions()

    def print_instructions(self):
        """Prints usage instructions"""
        instructions = """
        ========================================
        RL-DT Penalty Kick - Manual Reward Input
        ========================================
        
        Controls:
        [g] - Successful goal (+20 points)
        [m] - Missed shot (-2 points)
        [f] - Robot fell (-20 points)
        [r] - Reset episode
        [q] - Quit
        
        Waiting for input...
        """
        print(instructions)

    def check_input(self):
        """Checks for keyboard input"""
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            self.process_key(key)

    def process_key(self, key):
        """Processes pressed key"""
        reward_msg = Float32()
        reset_msg = Bool()
        
        if key.lower() == 'g':
            # Successful goal
            reward_msg.data = 20.0
            self.reward_pub.publish(reward_msg)
            self.get_logger().info("Successful goal! Reward: +20")
            
        elif key.lower() == 'm':
            # Missed shot
            reward_msg.data = -2.0
            self.reward_pub.publish(reward_msg)
            self.get_logger().info("Missed shot. Reward: -2")
            
        elif key.lower() == 'f':
            # Robot fell
            reward_msg.data = -20.0
            self.reward_pub.publish(reward_msg)
            self.get_logger().info("Robot fell! Reward: -20")
            
        elif key.lower() == 'r':
            # Reset episode
            reset_msg.data = True
            self.reset_pub.publish(reset_msg)
            self.get_logger().info("Episode reset")
            
        elif key.lower() == 'q':
            # Quit
            self.get_logger().info("Closing node...")
            self.cleanup()
            rclpy.shutdown()

    def cleanup(self):
        """Restores terminal configuration"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def main(args=None):
    rclpy.init(args=args)
    
    node = ManualRewardNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()