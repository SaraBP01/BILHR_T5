#!/usr/bin/env python3
"""
TUM - ICS AiNex CameraSubscriberCompressed Demo for ROS 2 Jazzy
----------------------------------------
Subscribes to JPEG-compressed images and raw images on /camera_image/compressed and /camera_image,
shows frames with OpenCV, and displays CameraInfo.

Requires:
  sudo apt install python3-numpy python3-opencv

Msgs:
    sensor_msgs/CompressedImage
    sensor_msgs/CameraInfo
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from rclpy.callback_groups import ReentrantCallbackGroup

import pickle
import sys
sys.path.append('/home/bilhr2025/BILHR-T2/ainex_bilhr_ws/src/ainex_motion/ainex_motion')
from joint_controller import JointController

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.cb_group = ReentrantCallbackGroup()

        # QoS: Reliable to ensure camera_info is received
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe compressed images
        self.sub_compressed = self.create_subscription(
            CompressedImage,
            'camera_image/compressed',
            self.image_callback_compressed,
            sensor_qos,
            callback_group=self.cb_group,
        )
        self.sub_compressed

        # Subscribe camera info
        self.sub_camerainfo = self.create_subscription(
            CameraInfo,
            'camera_info',
            self.camera_info_callback,
            sensor_qos,
            callback_group=self.cb_group
        )
        self.sub_camerainfo

        # State variables
        self.camera_info_received = False

        self.frame = None

        self.positions = []  
        self.blob_center = None
        self.shoulder_position = None 
        self.joint_controller = JointController(self) 


    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.get_logger().info(
                f'Camera Info received: {msg.width}x{msg.height}\n'
                f'K: {msg.k}\n'
                f'D: {msg.d}'
            )
            print(f'Camera Info received: {msg.width}x{msg.height}')
            print(f'Intrinsic matrix K: {msg.k}')
            print(f'Distortion coeffs D: {msg.d}')
            self.camera_info_received = True

    def image_callback_compressed(self, msg: CompressedImage):
        try:
            # Decode the compressed image
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn('JPEG decode returned None')
                return
            self.frame = frame

           # Convert the current frame from BGR to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define HSV limits for purple
            lower_purple = np.array([125, 50, 50])
            upper_purple = np.array([155, 255, 255])
          
            #Isolate the purple areas
            #Create a binary mask (black and white)
            #Pixels inside the purple range would be white, and black outside the range
          
            mask = cv2.inRange(hsv, lower_purple, upper_purple)

            # Find contours (blobs) in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #Process the largest blob
          
            if contours:
                # Get the largest contour
                largest_blob = max(contours, key=cv2.contourArea)
                #Calculate the center
                M = cv2.moments(largest_blob)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Draw the blob and its center
                    cv2.drawContours(frame, [largest_blob], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    # Log to terminal
                    self.blob_center = (cx, cy)
                    self.get_logger().info(f'Purple blob center: ({cx}, {cy})')
            else:
                self.get_logger().info('No purple blob detected.')
            self.get_logger().info(f'Recorded images {len(self.positions)}')
              
            #Save the update frame
            self.frame = frame  # Save updated frame with drawing
        except Exception as exc:
            self.get_logger().error(f'Decode error in compressed image: {exc}')


    def process_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Quit
        if key == ord('c'):
            self.show_compressed = True
            self.get_logger().info('Switched to compressed image')
        
        if key == ord('s'):  # Save blob center and shoulder position
            if self.blob_center is not None:
                shoulder = self.joint_controller.getJointPositions(["l_sho_roll", "l_sho_pitch"])  
                if shoulder is not None:
                    data_point = (*self.blob_center, *shoulder)
                    self.positions.append(data_point)
                    self.get_logger().info(f'Saved: Blob {self.blob_center}, Shoulder {shoulder}')
                else:
                    self.get_logger().warn('Shoulder position not available.')
            else:
                self.get_logger().warn('No blob detected to save.')
        return True

    def display_loop(self):
        while rclpy.ok():
            if self.frame is not None:
                # Display the compressed image
                cv2.imshow('Camer Subscrber', self.frame)

            if not self.process_key():
                break

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    node.get_logger().info('CameraSubscriber node started')

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        with open('src/ainex_vision/ainex_vision/data_collected/blob_shoulder_data_T4.pkl', 'wb') as f:
            pickle.dump(node.positions, f)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()