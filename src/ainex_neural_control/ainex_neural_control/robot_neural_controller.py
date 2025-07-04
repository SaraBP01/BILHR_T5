#!/usr/bin/env python3
# Authors: Giula Amenduni Scalabrino, Sara Barrionuevo Pino, Edoardo Calderoni, Natalia Martín Eliva & Esther Utasá Cebollero

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import math
import pickle
import os

from ainex_motion.joint_controller import JointController

# Define the main neural controller node
class RobotNeuralController(Node):
    def __init__(self):
        super().__init__('robot_neural_controller')
        
        # Initialize the CV bridge for ROS <-> OpenCV image conversion
        self.bridge = CvBridge()
        
        # Initialize the joint controller using your existing architecture
        self.joint_controller = JointController(self)
        
        # Start robot in crouch position 
        self.get_logger().info('Setting robot to crouch position...')
        self.joint_controller.setPosture('crouch', 1.0)
        time.sleep(2)
        
        # Load trained neural network and normalization parameters
        self.load_neural_network()
        
        # Define purple color range for blob detection in HSV
        self.lower_color = np.array([100, 50, 50])    # Purple lower bound in HSV
        self.upper_color = np.array([155, 255, 255])  # Purple upper bound in HSV

        # Camera image dimensions (will be updated)
        self.image_width = 640
        self.image_height = 480

        # Define safety limits for certain robot joints
        self.joint_limits = {
            'l_sho_roll': (-4.0, 0.5),   # Left shoulder roll 
            'l_sho_pitch': (-2.0, 0.0),  # Left shoulder pitch 
        }

        # Configure QoS to match sensor publisher
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        # Subscribe to camera compressed images
        self.image_sub = self.create_subscription(
            CompressedImage,
            'camera_image/compressed',  # camera topic
            self.image_callback,
            sensor_qos,
        )

        # Publisher for detected blob position 
        self.blob_pub = self.create_publisher(
            Point,
            '/detected_blob_position',
            10
        )

        # Periodic heartbeat timer
        self.create_timer(3.0, self.heartbeat_callback)

        # Track last movement time to avoid too frequent updates
        self.last_movement_time = time.time()
        self.movement_interval = 0.2  # Minimum time between movements (seconds)

    def heartbeat_callback(self):
        """Debug: Show that node is alive and waiting"""
        self.get_logger().info('Neural controller alive - waiting for camera images...')

    def load_neural_network(self):
        """Load trained neural network weights and normalization parameters"""
        self.network_loaded = False # Initialize to False
        self.norm_params = None
        self.nn_weights = None
        self.nn_biases = None

        base_path = os.path.dirname(__file__) # Gets directory of robot_neural_controller.py
        norm_params_path = os.path.join(base_path, 'normalization_params.pkl')
        weights_path = os.path.join(base_path, 'trained_weights.pkl')

        # Load normalization parameters
        try:
            with open(norm_params_path, 'rb') as f:
                self.norm_params = pickle.load(f)
            # Add a check for expected keys in norm_params, e.g., 'mins' and 'maxs'
            if 'mins' in self.norm_params and 'maxs' in self.norm_params:
                self.get_logger().info(f'Successfully loaded normalization params from {norm_params_path}. Keys: {list(self.norm_params.keys())}')
            else:
                self.get_logger().error(f"Normalization params file {norm_params_path} is missing 'mins' or 'maxs' keys. Loaded: {self.norm_params}")
                self.norm_params = None # Invalidate due to missing keys
        except FileNotFoundError:
            self.get_logger().error(f'CRITICAL: Normalization params file not found at {norm_params_path}. Network cannot operate.')
            # self.network_loaded remains False
            return # Cannot proceed without norm_params for de-normalization
        except Exception as e:
            self.get_logger().error(f'CRITICAL: Error loading or validating normalization params from {norm_params_path}: {e}. Network cannot operate.')
            # self.network_loaded remains False
            return

        if self.norm_params is None: # If loading/validation failed
             self.get_logger().error("Aborting network loading due to normalization parameter issues.")
             return # self.network_loaded is already False

        # Load trained neural network weights
        try:
            with open(weights_path, 'rb') as f:
                weights_data = pickle.load(f)
            
            if 'weights' in weights_data and 'biases' in weights_data:
                self.nn_weights = weights_data['weights']
                self.nn_biases = weights_data['biases']
                # Basic validation of loaded weights/biases structure (e.g., are they lists, expected number of layers)
                if isinstance(self.nn_weights, list) and isinstance(self.nn_biases, list) and len(self.nn_weights) == 3 and len(self.nn_biases) == 3: # Assuming 2-hidden layer (3 sets of w/b)
                    self.get_logger().info(f'Successfully loaded neural network weights and biases from {weights_path}.')
                    self.network_loaded = True # Success!
                else:
                    self.get_logger().error(f"Loaded weights/biases from {weights_path} have unexpected structure. Weights type: {type(self.nn_weights)}, Biases type: {type(self.nn_biases)}, Num weight sets: {len(self.nn_weights) if isinstance(self.nn_weights, list) else 'N/A'}.")
                    # self.network_loaded remains False
            else:
                self.get_logger().error(f"Keys 'weights' and 'biases' not found in {weights_path}. Available keys: {list(weights_data.keys()) if isinstance(weights_data, dict) else 'Not a dict'}.")
                # self.network_loaded remains False
        except FileNotFoundError:
            self.get_logger().error(f'Weights file not found at {weights_path}.')
            # self.network_loaded remains False
        except Exception as e:
            self.get_logger().error(f'Error loading neural network weights from {weights_path}: {e}')
            # self.network_loaded remains False

    def image_callback(self, msg):
        """Process incoming camera images and control robot"""
        try:
            # Convert compressed image to OpenCV
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            # Update image dimensions
            self.image_height, self.image_width = cv_image.shape[:2]
            self.get_logger().debug(f'Processing image: {self.image_width}x{self.image_height}')

            # Detect purple blob in the image
            blob_position = self.detect_color_blob(cv_image)
            if blob_position is not None:
                self.get_logger().info(f'Purple blob detected at: ({blob_position[0]}, {blob_position[1]})')

                # Publish detected blob position
                blob_msg = Point()
                blob_msg.x = float(blob_position[0])
                blob_msg.y = float(blob_position[1])
                blob_msg.z = 0.0
                self.blob_pub.publish(blob_msg)

                # Check if enough time has passed since last movement
                current_time = time.time()
                if current_time - self.last_movement_time >= self.movement_interval:
                    # Predict joint positions using neural network
                    predicted_joints = self.predict_joint_positions(blob_position)
                    # Apply safety limits
                    safe_joints = self.apply_safety_limits(predicted_joints)
                    # Move the robot using JointController
                    self.joint_controller.setJointPositions(
                        ["l_sho_roll", "l_sho_pitch"], 
                        safe_joints.tolist(), 
                        duration=0.3  # Smooth movement
                    )
                    # self.get_logger().info(f'Moving joints: roll={safe_joints[0]:.3f}, pitch={safe_joints[1]:.3f}')
                    self.last_movement_time = current_time  

            else:
                self.get_logger().debug('No purple blob detected')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def detect_color_blob(self, image):
        """Detect purple colored blob in image"""
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Create mask for purple color
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        # Noise reduction
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            # Filter out very small detections (noise)
            if area > 200:  # Minimum area threshold
                # Calculate center of mass
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        return None

    def predict_joint_positions(self, blob_position):
        """Use neural network to predict joint positions from blob position"""
        
        if not hasattr(self, 'network_loaded') or not self.network_loaded:
            # Fall back to simple mapping when no neural network is loaded
            self.get_logger().warn('Neural network not loaded, using simple mapping')
            return self.simple_blob_to_joint_mapping(blob_position)
        
        try:
            # Normalize input for the network
            normalized_input = self.normalize_input_for_network(blob_position)
            
            # Forward pass through the neural network
            # Ensure normalized_input is a 2D array if it's a single sample
            current_activation = np.array(normalized_input).reshape(1, -1)

            # First hidden layer (ReLU)
            z1 = np.dot(current_activation, self.nn_weights[0]) + self.nn_biases[0]
            activation1 = np.maximum(0, z1)

            # Second hidden layer (ReLU)
            z2 = np.dot(activation1, self.nn_weights[1]) + self.nn_biases[1]
            activation2 = np.maximum(0, z2)

            # Output layer (Linear)
            network_output_2d = np.dot(activation2, self.nn_weights[2]) + self.nn_biases[2]
            
            # network_output will be a 2D array (e.g., [[roll, pitch]]), extract the 1D array
            network_output = network_output_2d[0]
            
            # Denormalize the output to get actual joint angles
            actual_joints = self.denormalize_network_output(network_output)
            
            # Apply reduction factor to smooth movements
            reduction_factor = 0.5  # smooth movement
            reduced_joints = actual_joints * reduction_factor
            
            self.get_logger().info(f' Original joints: {actual_joints}')
            self.get_logger().info(f' Reduced joints: {reduced_joints}')
            
            return reduced_joints
        
        except Exception as e:
            self.get_logger().error(f'Error in neural network prediction: {e}')
            # Fall back to simple mapping
            return self.simple_blob_to_joint_mapping(blob_position)
        

    def normalize_input_for_network(self, blob_position):
        """Normalize blob position for neural network input using loaded parameters"""
        if self.norm_params is None:
            self.get_logger().error('No normalization params loaded!')
            # simple normalization
            norm_x = blob_position[0] / self.image_width
            norm_y = blob_position[1] / self.image_height
            return np.array([norm_x, norm_y])
        
        # Ensure image_width and image_height are positive to avoid division by zero
        if self.image_width <= 0 or self.image_height <= 0:
            self.get_logger().error(f"Invalid image dimensions: width={self.image_width}, height={self.image_height}")
            return np.array([0.5, 0.5]) # Return a neutral value

        norm_x = (self.image_width - blob_position[0]) / self.image_width
        self.get_logger().debug(f"Normalizing. Original cx: {blob_position[0]}, Inverted norm_x: {norm_x}")
        norm_y = blob_position[1] / self.image_height
        # Clip to ensure values are in [0,1] range
        norm_x = np.clip(norm_x, 0.0, 1.0)
        norm_y = np.clip(norm_y, 0.0, 1.0)
        return np.array([norm_x, norm_y])

    def denormalize_network_output(self, norm_output):
        """Denormalize network output from [0,1] to actual joint angles using loaded parameters."""
        if self.norm_params is None or 'mins' not in self.norm_params or 'maxs' not in self.norm_params:
            self.get_logger().error('Normalization parameters (mins/maxs) not loaded or incomplete!')
            # Fallback or raise error, here returning norm_output to avoid crash but log severity
            return norm_output 

        try:
            # Assuming norm_params['mins'] and norm_params['maxs'] are lists/arrays of 4 elements:
            # [blob_x_min, blob_y_min, joint1_min, joint2_min]
            # [blob_x_max, blob_y_max, joint1_max, joint2_max]
            # We need the joint mins and maxs for de-normalization (last 2 elements)
            actual_joint_min = np.array(self.norm_params['mins'][2:])
            actual_joint_max = np.array(self.norm_params['maxs'][2:])

            # Ensure norm_output is a numpy array for element-wise operations
            norm_output_np = np.array(norm_output)

            # Denormalize from [0,1] (network output) to actual joint ranges
            denorm_joints = norm_output_np * (actual_joint_max - actual_joint_min) + actual_joint_min
            
            # self.get_logger().info(f"De-normalizing with min: {actual_joint_min}, max: {actual_joint_max}")
            return denorm_joints
            
        except Exception as e:
            self.get_logger().error(f"Error during de-normalization: {e}")
            # Fallback or raise error
            return norm_output # Or a safe default

    def simple_blob_to_joint_mapping(self, blob_position):
        """Simple mapping from blob position (fallback when network fails)"""
        # Normalize blob position to [0,1]
        norm_x = blob_position[0] / self.image_width
        norm_y = blob_position[1] / self.image_height
        # Clip to ensure values are in [0,1] range
        norm_x = np.clip(norm_x, 0.0, 1.0)
        norm_y = np.clip(norm_y, 0.0, 1.0)
        #self.get_logger().debug(f'Simple mapping: Blob ({blob_position[0]}, {blob_position[1]}) -> Normalized ({norm_x:.3f}, {norm_y:.3f})')
        # Map directly to joint ranges
        joint_roll = norm_x * (0.1 - (-3.5)) + (-3.5)  # Map to roll range
        joint_pitch = norm_y * (-0.3 - (-1.6)) + (-1.6)  # Map to pitch range
        return np.array([joint_roll, joint_pitch])

    def apply_safety_limits(self, joint_positions):
        """Apply safety limits to prevent dangerous joint positions"""
        safe_positions = joint_positions.copy()
        joint_names = ["l_sho_roll", "l_sho_pitch"]

        for i, joint_name in enumerate(joint_names):
            if joint_name in self.joint_limits:
                min_limit, max_limit = self.joint_limits[joint_name]
                original_value = safe_positions[i]
                safe_positions[i] = np.clip(safe_positions[i], min_limit, max_limit)
                if abs(safe_positions[i] - original_value) > 0.001:
                    self.get_logger().warn(f'Joint {joint_name} clamped from {original_value:.3f} to {safe_positions[i]:.3f}')
        return safe_positions
        
# Main function to launch the ROS node
def main(args=None):
    rclpy.init(args=args)
    try:
        controller = RobotNeuralController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("\nShutting down neural controller...")

    finally:
        try:
            controller.joint_controller.setPosture('crouch', 1.0)
            time.sleep(1)
        except:
            pass
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()