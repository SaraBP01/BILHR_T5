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
        
        # ArUco detection variables (replacing blob detection)
        self.target_marker = None  # The chosen target marker (closest to center)
        self.all_markers = []      # All detected markers
        self.shoulder_position = None 
        self.joint_controller = JointController(self) 

        # ArUco setup
        self.aruco_dict = None
        self.aruco_params = None
        self.setup_aruco()

    def setup_aruco(self):
        """Setup ArUco detector"""
        try:
            # Try new OpenCV
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.get_logger().info('ArUco setup OK (new OpenCV)')
        except:
            try:
                # Try old OpenCV
                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
                self.get_logger().info('ArUco setup OK (old OpenCV)')
            except Exception as e:
                self.get_logger().warn(f'ArUco setup failed: {e}')
                self.aruco_dict = None

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

            # ArUco detection (replacing purple blob detection)
            if self.aruco_dict is not None:
                self.detect_aruco_markers(frame)
            else:
                self.get_logger().info('ArUco not available')
            
            self.get_logger().info(f'Recorded images {len(self.positions)}')
              
            # Save the updated frame
            self.frame = frame  # Save updated frame with drawing
        except Exception as exc:
            self.get_logger().error(f'Decode error in compressed image: {exc}')

    def detect_aruco_markers(self, frame):
        """Detect ArUco markers and select target (replaces blob detection)"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners = None
            ids = None
            
            try:
                # New OpenCV
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
            except:
                # Old OpenCV
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None and len(ids) > 0:
                # Process detected markers
                detected_markers = []
                image_center_x = frame.shape[1] // 2
                
                for i in range(len(ids)):
                    corner = corners[i][0]
                    marker_id = ids[i][0]
                    
                    # Calculate marker center
                    center_x = int(corner[:, 0].mean())
                    center_y = int(corner[:, 1].mean())
                    distance_to_center = abs(center_x - image_center_x)
                    
                    marker_info = {
                        'id': marker_id,
                        'center_x': center_x,
                        'center_y': center_y,
                        'distance_to_center': distance_to_center
                    }
                    detected_markers.append(marker_info)
                
                # Store all markers
                self.all_markers = detected_markers
                
                # Select target marker (closest to center - this is the "goalkeeper")
                self.target_marker = min(detected_markers, key=lambda m: m['distance_to_center'])
                
                # Draw all detected markers
                for marker in detected_markers:
                    # Color: Green for target (goalkeeper), Blue for posts
                    color = (0, 255, 0) if marker == self.target_marker else (255, 0, 0)
                    thickness = 3 if marker == self.target_marker else 2
                    
                    # Draw marker center
                    cv2.circle(frame, (marker['center_x'], marker['center_y']), 8, color, thickness)
                    
                    # Draw marker ID
                    label = f"ID:{marker['id']}"
                    if marker == self.target_marker:
                        label += " [GOAL]"
                    else:
                        label += " [POST]"
                    
                    cv2.putText(frame, label, 
                               (marker['center_x'] + 15, marker['center_y']), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw center reference line
                cv2.line(frame, (image_center_x, 0), (image_center_x, frame.shape[0]), (255, 255, 255), 2)
                
                # Log detection
                marker_list = [f"ID{m['id']}" for m in detected_markers]
                self.get_logger().info(f'ArUco detected: {", ".join(marker_list)}. Target: ID{self.target_marker["id"]} (goalkeeper)')
                
            else:
                # No markers detected
                self.target_marker = None
                self.all_markers = []
                self.get_logger().info('No ArUco markers detected.')
                
        except Exception as e:
            self.get_logger().error(f'ArUco detection error: {e}')

    def process_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Quit
        if key == ord('c'):
            self.show_compressed = True
            self.get_logger().info('Switched to compressed image')
        
        if key == ord('s'):  # Save target marker and shoulder position
            if self.target_marker is not None:
                shoulder = self.joint_controller.getJointPositions(["l_sho_roll", "l_sho_pitch"])  
                if shoulder is not None:
                    data_point = {
                        'target_x': self.target_marker['center_x'],
                        'target_y': self.target_marker['center_y'], 
                        'target_id': self.target_marker['id'],
                        'shoulder': shoulder,
                        'all_markers': self.all_markers
                    }
                    self.positions.append(data_point)
                    self.get_logger().info(f'Saved: Target ID{self.target_marker["id"]} at ({self.target_marker["center_x"]}, {self.target_marker["center_y"]}), Shoulder {shoulder}')
                else:
                    self.get_logger().warn('Shoulder position not available.')
            else:
                self.get_logger().warn('No target marker detected to save.')
        return True

    def display_loop(self):
        while rclpy.ok():
            if self.frame is not None:
                # Display the compressed image
                cv2.imshow('Camera Subscriber', self.frame)

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
        with open('src/ainex_vision/ainex_vision/data_collected/aruco_target_data_T4.pkl', 'wb') as f:
            pickle.dump(node.positions, f)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()