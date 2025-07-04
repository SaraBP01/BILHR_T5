#Authors: Giulia Amenduni Scalabrino, Sara Barrionuevo Pino, Edoardo Calderoni, Natalia Martín Elvira, Esther Utasá Cebollero


import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16
from ainex_motion.joint_controller import JointController
import rclpy
from rclpy.node import Node
import math
import time

from servo_service.msg import *
from servo_service.srv import *

# Define class for the node responsible for the motion commands of the robot
class MotionNode(Node):
    def __init__(self):
        super().__init__('ainex_motion_node')
        # Subcribe the node to keyboard command
        self.subscription = self.create_subscription(Int16, 'keyboard_command', self.command_callback, 10)
        # Initialize the joint controller
        self.joint_controller = JointController(self)
        # Bring the robot to crouch position before it receives any commands
        self.crouch()
    
    # Define callback function to assign a motion to a given key input number
    def command_callback(self, msg):
        cmd = msg.data
        if cmd == 1:
            self.move_left_arm_home()
        elif cmd == 2:
            self.left_arm_wave()
        elif cmd == 3:
            self.mirror_both_arms()
        elif cmd == 4:
            self.unlock_all()
        # Additionally, command 5 brings the robot back to crouch position
        elif cmd == 5:
            self.crouch()

    # Define the function for each of the motions

    def crouch(self):
        self.joint_controller.setPosture('crouch', 0.8)
        
    def move_left_arm_home(self):
        # Bring the left arm to a safe home position of 90 degrees
        self.joint_controller.setJointPositions(
            ["l_sho_pitch", "l_sho_roll", "l_el_pitch", "l_el_yaw"],
            [-1.55, -1.70, 0.06, -1.55], 1
        )
        time.sleep(1)

    def move_right_arm_home(self):
        # Bring the right arm to a safe home position of 90 degrees
        self.joint_controller.setJointPositions(
            ["r_sho_pitch", "r_sho_roll", "r_el_pitch", "r_el_yaw"],
            [1.55, 1.70, 0.06, 1.55], 1
        )
        time.sleep(1)
        
    def left_arm_wave(self):
        self.move_left_arm_home()
        # Define the 'wave' trajectory and perform it three times with the left arm
        for _ in range(3):
            self.joint_controller.setJointPositions(["l_el_yaw"], [1.55], 1)
            time.sleep(1)
            self.joint_controller.setJointPositions(["l_el_yaw"], [-1.55],1)
            time.sleep(1)

    def mirror_both_arms(self):
        self.move_left_arm_home()
        self.move_right_arm_home()
        # Define the 'wave' trajectory and perform it three times with both arms
        for _ in range(3):
            self.joint_controller.setJointPositions(["l_el_yaw", "r_el_yaw"], [1.55, -1.55], 1)
            time.sleep(1)
            self.joint_controller.setJointPositions(["l_el_yaw", "r_el_yaw"], [-1.55, 1.55], 1)
            time.sleep(1)
        time.sleep(1)

    # Define a function for unlocking the joints
    def unlock_all(self):
        self.joint_controller.setJointLock('all', False)
        time.sleep(1)

        positions = self.joint_controller.getJointPositions(["r_sho_pitch", "r_sho_roll", "r_el_pitch", "r_el_yaw"])
        self.get_logger().info(f"Joint positions: {positions}")

def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
