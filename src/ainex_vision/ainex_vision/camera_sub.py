
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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, CameraInfo
from rclpy.callback_groups import ReentrantCallbackGroup

import pickle
import sys
import os # Import os for path handling

# Ensure the path to joint_controller.py is correct for your system
sys.path.append('/home/bilhr2025/BILHR-T2/ainex_bilhr_ws/src/ainex_motion/ainex_motion')
# Note: Ensure joint_controller.py does NOT call rclpy.shutdown() anywhere,
# especially not in its __del__ method or a cleanup function.
try:
    from joint_controller import JointController
except ImportError as e:
    print(f"ERROR: Could not import JointController. Please ensure the path is correct and the file exists. Error: {e}")
    # Exit or handle gracefully if JointController is critical
    sys.exit(1)


class CameraSubscriber(Node):
    def __init__(self): # CORRECT: Double underscores
        super().__init__('camera_subscriber') # CORRECT: Double underscores
        self.cb_group = ReentrantCallbackGroup()

        # QoS for compressed images (using pre-defined sensor data QoS)
        # This profile is typically BEST_EFFORT reliability, KEEP_LAST history, and depth=10
        sensor_qos = qos_profile_sensor_data

        # QoS for camera info - ADAPTED TO MATCH PUBLISHER'S BEST_EFFORT
        # Based on 'ros2 topic info /camera_info --verbose' output,
        # the publisher uses BEST_EFFORT reliability.
        camera_info_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # <-- CRITICAL FIX: Changed from RELIABLE to BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST,           # Best practice to explicitly set history
            depth=1                                    # Common depth for non-buffered streams
        )

        # Subscribe to compressed images
        self.sub_compressed = self.create_subscription(
            CompressedImage,
            'camera_image/compressed',
            self.image_callback_compressed,
            sensor_qos,
            callback_group=self.cb_group,
        )
        # self.sub_compressed # REMOVED: This line was redundant
        self.get_logger().info('Subscribed to /camera_image/compressed')


        # Subscribe to camera info
        self.sub_camerainfo = self.create_subscription(
            CameraInfo,
            'camera_info',
            self.camera_info_callback,
            camera_info_qos,
            callback_group=self.cb_group
        )
        # self.sub_camerainfo # REMOVED: This line was redundant
        self.get_logger().info('Subscribed to /camera_info')

        # State variables
        self.camera_info_received = False
        self.frame = None
        self.positions = [] # List to store collected data points

        # ArUco detection variables (replacing blob detection)
        self.target_marker = None
        self.all_markers = []
        self.shoulder_position = None # This variable is assigned but not used later

        # Initialize JointController within a try-except block
        try:
            self.joint_controller = JointController(self)
            self.get_logger().info('JointController initialized successfully.')
        except Exception as e:
            self.get_logger().error(
                f'Failed to initialize JointController: {e}. '
                'Joint control functionality may be unavailable. '
                'Please ensure joint_controller.py is correct and accessible.',
                throttle_duration_sec=5.0 # Throttle to prevent log spam
            )
            self.joint_controller = None # Ensure it's None if init fails

        # ArUco setup
        self.aruco_dict = None
        self.aruco_params = None
        self.setup_aruco()
        self.get_logger().info('CameraSubscriber node fully initialized.')


    def setup_aruco(self):
        """Setup ArUco detector"""
        try:
            # Try new OpenCV syntax for ArUco
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.get_logger().info('ArUco setup OK (new OpenCV)')
        except AttributeError: # Catches if getPredefinedDictionary/DetectorParameters are not found
            try:
                # Try old OpenCV syntax for ArUco
                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
                self.get_logger().info('ArUco setup OK (old OpenCV)')
            except Exception as e: # Catch any other exceptions during old OpenCV setup
                self.get_logger().warn(f'ArUco setup failed: {e}. ArUco detection will be disabled.')
                self.aruco_dict = None # Ensure ArUco is disabled if setup fails

    def camera_info_callback(self, msg: CameraInfo):
        # Only log camera info once to avoid spamming console
        if not self.camera_info_received:
            self.get_logger().info(
                f'Camera Info received: {msg.width}x{msg.height}\n'
                f'K: {list(msg.k)}\n' # Convert tuple to list for cleaner logging
                f'D: {list(msg.d)}'  # Convert array('d') to list for cleaner logging
            )
            # REMOVED: Redundant print statements, relying on ROS 2 logger now
            self.camera_info_received = True

    def image_callback_compressed(self, msg: CompressedImage):
        try:
            # Decode the compressed image
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                self.get_logger().warn('JPEG decode returned None, skipping frame processing.', throttle_duration_sec=1.0)
                return
            
            self.frame = frame # Assign the decoded frame (can be drawn on later)

            # Perform ArUco detection if setup was successful
            if self.aruco_dict is not None:
                self.detect_aruco_markers(frame) # Pass frame to draw markers directly onto it
                # self.frame is updated within detect_aruco_markers if drawings occur.
            else:
                self.get_logger().debug('ArUco not available, skipping marker detection.')
            
            # Log the number of recorded images (consider making this DEBUG or throttling)
            self.get_logger().debug(f'Recorded images: {len(self.positions)}')

        except Exception as exc: # Catch any exceptions during image processing
            self.get_logger().error(f'Error processing compressed image: {exc}', throttle_duration_sec=1.0)

    def detect_aruco_markers(self, frame):
        """Detect ArUco markers and select target (replaces blob detection)"""
        if self.aruco_dict is None: # Exit early if ArUco not initialized
            return

        try:
            # Convert to grayscale for marker detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            corners = None
            ids = None
            
            try:
                # Try new OpenCV ArucoDetector interface
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, rejected = detector.detectMarkers(gray) # 'rejected' can be useful for debugging
            except AttributeError: # Fallback for older OpenCV
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None and len(ids) > 0:
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
                
                self.all_markers = detected_markers
                
                # Select target marker (closest to center - this is the "goalkeeper")
                self.target_marker = min(detected_markers, key=lambda m: m['distance_to_center'])
                
                # Draw all detected markers on the frame
                for marker in detected_markers:
                    # Color: Green for target (goalkeeper), Blue for posts
                    color = (0, 255, 0) if marker == self.target_marker else (255, 0, 0)
                    thickness = 3 if marker == self.target_marker else 2
                    
                    # Draw marker center
                    cv2.circle(frame, (marker['center_x'], marker['center_y']), 8, color, thickness)
                    
                    # Draw marker ID and role
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
                
                # Log detection (throttled to avoid excessive output)
                marker_list = [f"ID{m['id']}" for m in detected_markers]
                self.get_logger().info(
                    f'ArUco detected: {", ".join(marker_list)}. Target: ID{self.target_marker["id"]} (goalkeeper)',
                    throttle_duration_sec=0.5 # Log every 0.5 seconds at most
                )
                
            else:
                # No markers detected
                self.target_marker = None
                self.all_markers = []
                self.get_logger().info('No ArUco markers detected.', throttle_duration_sec=1.0)
                
        except Exception as e: # Catch any other generic exceptions during detection
            self.get_logger().error(f'ArUco detection error: {e}', throttle_duration_sec=1.0)

    def process_key(self):
        """Processes key presses for quit, save, etc."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('\'q\' pressed. Quitting application.')
            return False  # Quit
        
        # 'c' key for compressed image mode (currently no raw image sub)
        if key == ord('c'):
            self.get_logger().info('\'c\' pressed. (Functionality for switching image types is not implemented).')
        
        # 's' key to save target marker and shoulder position
        if key == ord('s'):
            if self.target_marker is not None:
                if self.joint_controller: # Check if JointController was successfully initialized
                    try:
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
                            self.get_logger().warn('Shoulder position not available from JointController.')
                    except Exception as e:
                        self.get_logger().error(f'Error getting joint positions from JointController: {e}')
                else:
                    self.get_logger().warn('JointController not initialized. Cannot get shoulder position.')
            else:
                self.get_logger().warn('No target marker detected to save.')
        return True # Continue loop

    def display_loop(self):
        """Main loop for displaying frames and processing keys."""
        while rclpy.ok(): # Loop while ROS 2 context is active
            if self.frame is not None:
                cv2.imshow('Camera Subscriber', self.frame)
            else:
                self.get_logger().debug('Waiting for first camera frame...') # Debug level for less verbosity

            # Process key presses and check if 'q' was pressed
            if not self.process_key():
                break # Exit the loop if process_key returns False

            # Spin once to allow callbacks to be processed
            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows() # Close all OpenCV windows when loop exits

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    # The 'node started' log can also be placed here or in __init__, consistent placement is fine.
    node.get_logger().info('CameraSubscriber node started.')

    try:
        node.display_loop() # Start the main display and processing loop
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received, shutting down.')
    except Exception as e:
        node.get_logger().error(f'An unexpected error occurred: {e}')
    finally:
        # --- Clean-up and Data Saving ---

        # 1. Ensure the output directory exists before saving
        # os.path.expanduser('~') resolves to the user's home directory.
        output_dir = os.path.expanduser('~/BILHR-T2/ainex_bilhr_ws/src/ainex_vision/ainex_vision/data_collected/')
        try:
            os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
            node.get_logger().info(f'Ensured output directory exists: {output_dir}')
        except OSError as e:
            node.get_logger().error(f'Could not create output directory {output_dir}: {e}. Data might not be saved.')
            output_dir = None # Indicate that output directory is not available

        # 2. Save collected data (only if directory could be prepared)
        if output_dir:
            file_path = os.path.join(output_dir, 'aruco_target_data_T4.pkl')
            try:
                if node.positions: # Only save if there's data to save
                    with open(file_path, 'wb') as f:
                        pickle.dump(node.positions, f)
                    node.get_logger().info(f'Successfully saved collected data to {file_path}')
                else:
                    node.get_logger().warn('No data collected, skipping data save.')
            except Exception as e:
                node.get_logger().error(f'Failed to save data to {file_path}: {e}')

        # 3. Destroy the node and shut down rclpy context
        if node: # Ensure node object exists before destroying
            node.destroy_node()
            node.get_logger().info('Node destroyed.') # Log before shutdown if possible
        
        # IMPORTANT: rclpy.shutdown() should only be called once in your entire application.
        # If you still get 'rcl_shutdown already called' errors, verify that JointController
        # or any other library/module you use does NOT call rclpy.shutdown() internally.
        # It's a common mistake in Python ROS 2 when libraries handle their own context.
        # Adding a log right before it to see if it's reached after previous errors.
        print("Attempting final rclpy shutdown (printed via Python print, not ROS logger).") # Use print for this final message
        rclpy.shutdown()

<<<<<<< HEAD
if __name__ == '__main__':
    main()
=======

if __name__ == '__main__': # CORRECT: Double underscores
    main()
>>>>>>> d9c53fb (k)
