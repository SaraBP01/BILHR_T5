#!/usr/bin/env python3

"""
ROS2 Node for RL-DT Penalty Kick Learning with Multiple ArUco Detection
File: src/ainex_neural_control/rl_dt_penalty_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .rl_dt_penalty_kick import RLDT  # Import our RL-DT class

class RLDTPenaltyNode(Node):
    def __init__(self):
        super().__init__('rl_dt_penalty_node')
        
        # Initialize RL-DT agent
        self.agent = RLDT(max_reward=20, discount_factor=0.9)
        
        # ROS2 Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/rl_dt_status', 10)
        
        # ROS2 Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera_image', self.image_callback, 10)
        self.reward_sub = self.create_subscription(
            Float32, '/penalty_reward', self.reward_callback, 10)
        self.reset_sub = self.create_subscription(
            Bool, '/reset_episode', self.reset_callback, 10)
        
        # CV Bridge for images
        self.bridge = CvBridge()
        
        # ArUco detector setup with extensive logging
        self.aruco_dict = None
        self.aruco_params = None
        self.setup_aruco_detector()
        
        # Robot state
        self.current_joint_states = None
        self.marker_position = None
        self.marker_id = None
        self.all_detected_markers = []
        self.leg_position = 0.0
        self.current_image = None
        
        # Episode control
        self.episode_active = False
        self.current_state = None
        self.last_action = None
        self.episode_count = 0
        self.step_count = 0
        self.max_steps_per_episode = 20
        
        # Debug counters
        self.image_count = 0
        self.detection_count = 0
        
        # Timer for main loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("RL-DT Penalty Kick Node initialized")

    def setup_aruco_detector(self):
        """Setup ArUco detector with extensive error handling and logging"""
        self.get_logger().info("Setting up ArUco detector...")
        
        # Check OpenCV version
        cv_version = cv2.__version__
        self.get_logger().info(f"OpenCV version: {cv_version}")
        
        try:
            # Try newer OpenCV versions first (4.7+)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.get_logger().info("ArUco initialized (OpenCV 4.7+)")
            
        except AttributeError:
            try:
                # Try older OpenCV versions (4.6 and older)
                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
                self.get_logger().info("ArUco initialized (OpenCV 4.6 or older)")
                
            except Exception as e:
                self.get_logger().error(f"ArUco initialization failed: {e}")
                self.aruco_dict = None
                self.aruco_params = None
                
        if self.aruco_dict is not None:
            self.get_logger().info("ArUco detector successfully initialized")
        else:
            self.get_logger().error("ArUco detector initialization failed - will use fallback detection")

    def joint_callback(self, msg):
        """Callback to receive joint states"""
        self.current_joint_states = msg
        
        # Extract left leg position
        if 'LAnklePitch' in msg.name:
            idx = msg.name.index('LAnklePitch')
            self.leg_position = msg.position[idx] * 1000
        elif 'l_ankle_pitch' in msg.name:
            idx = msg.name.index('l_ankle_pitch')
            self.leg_position = msg.position[idx] * 1000
        
        # Log joint reception (only first time)
        if not hasattr(self, '_joint_received'):
            self._joint_received = True
            self.get_logger().info(f"Joint states received. Available joints: {msg.name}")

    def image_callback(self, msg):
        """Callback to process camera images with extensive debugging"""
        try:
            # Convert image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
            self.image_count += 1
            
            # Log image reception
            if self.image_count == 1:
                self.get_logger().info(f"First image received! Size: {cv_image.shape}")
            elif self.image_count % 30 == 0:
                self.get_logger().info(f"Processed {self.image_count} images")
            
            # Detect markers
            self.marker_position, self.marker_id = self.detect_multiple_aruco_markers(cv_image)
            
            # Log detection results
            if self.marker_position is not None:
                self.detection_count += 1
                if self.detection_count % 10 == 0:
                    self.get_logger().info(f"Total detections: {self.detection_count}")
                    
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def detect_multiple_aruco_markers(self, image):
        """Detect ArUco markers with extensive debugging"""
        if self.aruco_dict is None:
            self.get_logger().debug("ArUco not available, using fallback detection")
            return self.detect_fallback_target(image)
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Log image properties
            if self.image_count == 1:
                self.get_logger().info(f"Image properties - Gray shape: {gray.shape}, dtype: {gray.dtype}")
            
            # Detect markers
            corners, ids, rejected = None, None, None
            
            try:
                # Try newer OpenCV method
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, rejected = detector.detectMarkers(gray)
                detection_method = "OpenCV 4.7+"
                
            except:
                try:
                    # Try older OpenCV method  
                    corners, ids, rejected = cv2.aruco.detectMarkers(
                        gray, self.aruco_dict, parameters=self.aruco_params)
                    detection_method = "OpenCV 4.6-"
                    
                except Exception as e:
                    self.get_logger().error(f"Both ArUco detection methods failed: {e}")
                    return self.detect_fallback_target(image)
            
            # Log detection attempt
            if self.image_count % 30 == 0:
                rejected_count = len(rejected) if rejected is not None else 0
                self.get_logger().info(f"Detection attempt #{self.image_count} using {detection_method}")
                self.get_logger().info(f"Rejected markers: {rejected_count}")
            
            # Process detected markers
            if ids is not None and len(ids) > 0:
                detected_markers = []
                image_center_x = image.shape[1] // 2
                
                for i in range(len(ids)):
                    corner = corners[i][0]
                    marker_id = ids[i][0]
                    
                    # Calculate marker center
                    center_x = int(corner[:, 0].mean())
                    center_y = int(corner[:, 1].mean())
                    distance_to_center = abs(center_x - image_center_x)
                    
                    detected_markers.append({
                        'id': marker_id,
                        'center_x': center_x,
                        'center_y': center_y,
                        'distance_to_center': distance_to_center,
                        'corner': corner
                    })
                
                # Store all detected markers
                self.all_detected_markers = detected_markers.copy()
                
                # Select target marker (closest to center)
                target_marker = min(detected_markers, key=lambda m: m['distance_to_center'])
                
                # Draw all markers
                for marker in detected_markers:
                    color = (0, 255, 0) if marker == target_marker else (0, 100, 255)
                    thickness = 3 if marker == target_marker else 2
                    
                    # Draw marker center
                    cv2.circle(image, (marker['center_x'], marker['center_y']), 8, color, thickness)
                    
                    # Draw marker info
                    label = f"ID:{marker['id']}"
                    if marker == target_marker:
                        label += " [TARGET]"
                    
                    cv2.putText(image, label, 
                               (marker['center_x'] + 15, marker['center_y'] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw center reference line
                cv2.line(image, (image_center_x, 0), (image_center_x, image.shape[0]), (255, 255, 255), 1)
                cv2.putText(image, "CENTER", (image_center_x + 5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Log successful detection
                marker_info = [f"ID{m['id']}@{m['center_x']}(d={m['distance_to_center']})" 
                              for m in detected_markers]
                self.get_logger().info(
                    f"✓ Detected {len(detected_markers)} markers: {', '.join(marker_info)}. "
                    f"Target: ID{target_marker['id']} at ({target_marker['center_x']}, {target_marker['center_y']})"
                )
                
                # Show image with detections
                cv2.imshow('RL-DT ArUco Detection', image)
                cv2.waitKey(1)
                
                return target_marker['center_x'], target_marker['id']
            
            else:
                # No markers detected - show image anyway for debugging
                cv2.putText(image, "NO MARKERS DETECTED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('RL-DT ArUco Detection', image)
                cv2.waitKey(1)
                
                if self.image_count % 30 == 0:
                    self.get_logger().warn(f"No markers detected in image #{self.image_count}")
                
                return None, None
            
        except Exception as e:
            self.get_logger().error(f"ArUco detection error: {e}")
            return self.detect_fallback_target(image)

    def detect_fallback_target(self, image):
        """Fallback detection using color detection"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detect white regions
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([179, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest rectangular contour
                for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Draw detected target
                            cv2.circle(image, (cx, cy), 8, (0, 0, 255), 3)
                            cv2.putText(image, "FALLBACK TARGET", 
                                       (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            # Show fallback detection
                            cv2.imshow('RL-DT ArUco Detection', image)
                            cv2.waitKey(1)
                            
                            self.get_logger().info(f"Fallback target detected at ({cx}, {cy})")
                            return cx, 999
                
            # Show image even if no fallback detection
            cv2.putText(image, "NO TARGETS DETECTED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('RL-DT ArUco Detection', image)
            cv2.waitKey(1)
            
            return None, None
            
        except Exception as e:
            self.get_logger().error(f"Fallback detection error: {e}")
            return None, None

    def reward_callback(self, msg):
        """Callback to receive manual rewards"""
        if self.episode_active and self.last_action == 'KICK':
            reward = msg.data
            self.process_reward(reward)

    def reset_callback(self, msg):
        """Callback to reset episode"""
        if msg.data:
            self.reset_episode()

    def get_current_state(self):
        """Gets current robot state"""
        if self.marker_position is None or self.leg_position is None:
            return None
        return self.agent.discretize_state(self.marker_position, self.leg_position)

    def execute_action(self, action):
        """Executes an action on the real robot"""
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        
        if action == 'MOVE_LEFT':
            joint_msg.name = ['LAnklePitch']
            current_pos = self.leg_position / 1000.0
            new_pos = current_pos - 0.01
            joint_msg.position = [new_pos]
            
        elif action == 'MOVE_RIGHT':
            joint_msg.name = ['LAnklePitch']
            current_pos = self.leg_position / 1000.0
            new_pos = current_pos + 0.01
            joint_msg.position = [new_pos]
            
        elif action == 'KICK':
            self.execute_kick_sequence()
            return
        
        self.joint_pub.publish(joint_msg)
        self.get_logger().info(f"Executing action: {action}")

    def execute_kick_sequence(self):
        """Executes the kicking sequence"""
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = ['LKneePitch', 'LAnklePitch']
        joint_msg.position = [0.5, 0.0]
        
        self.joint_pub.publish(joint_msg)
        self.get_logger().info("Executing KICK action")

    def process_reward(self, reward):
        """Processes received reward"""
        if self.current_state is None or self.last_action is None:
            return
        
        final_reward = reward
        if self.last_action == 'KICK' and reward == 0 and self.marker_id is not None:
            auto_rewards = {
                52: 25.0,   # Center marker
                53: 20.0,   # Left post
                54: 20.0,   # Right post  
                999: 15.0   # Fallback
            }
            
            if self.marker_id in auto_rewards:
                final_reward = auto_rewards[self.marker_id]
                self.get_logger().info(f"Auto reward for marker ID {self.marker_id}: {final_reward}")
        
        next_state = self.get_current_state()
        if next_state is None:
            next_state = self.current_state
        
        self.agent.add_experience(
            self.current_state, 
            self.last_action, 
            final_reward, 
            next_state
        )
        
        self.get_logger().info(f"Reward processed: {final_reward}")
        
        if self.last_action == 'KICK':
            self.end_episode()

    def control_loop(self):
        """Main control loop"""
        if not self.episode_active:
            if self.can_start_episode():
                self.start_episode()
            return
        
        if self.step_count >= self.max_steps_per_episode:
            self.end_episode()
            return
        
        current_state = self.get_current_state()
        if current_state is None:
            return
        
        action = self.agent.select_action(current_state)
        self.execute_action(action)
        
        self.current_state = current_state
        self.last_action = action
        self.step_count += 1
        
        # Publish detailed status
        status_msg = String()
        mode = "Exploration" if self.agent.exploration_mode else "Exploitation"
        status = f"Episode: {self.episode_count}, Step: {self.step_count}, Mode: {mode}, Action: {action}"
        
        if self.marker_id is not None:
            status += f", Target_ID: {self.marker_id}"
        
        if len(self.all_detected_markers) > 0:
            marker_count = len(self.all_detected_markers)
            marker_ids = [str(m['id']) for m in self.all_detected_markers]
            status += f", Detected: {marker_count} markers [{','.join(marker_ids)}]"
        
        status_msg.data = status
        self.status_pub.publish(status_msg)

    def can_start_episode(self):
        """Checks if a new episode can be started"""
        return (self.marker_position is not None and 
                self.current_joint_states is not None)

    def start_episode(self):
        """Starts a new episode"""
        self.episode_active = True
        self.step_count = 0
        self.episode_count += 1
        self.current_state = self.get_current_state()
        self.last_action = None
        
        marker_info = "No markers detected"
        if len(self.all_detected_markers) > 0:
            marker_list = [f"ID{m['id']}" for m in self.all_detected_markers]
            marker_info = f"{len(self.all_detected_markers)} markers: {', '.join(marker_list)}"
            if self.marker_id is not None:
                marker_info += f" (Target: ID{self.marker_id})"
        
        self.get_logger().info(f"Starting episode {self.episode_count}. {marker_info}")

    def end_episode(self):
        """Ends current episode"""
        self.episode_active = False
        self.agent.value_iteration()
        
        if len(self.agent.episode_rewards) > 0:
            total_reward = self.agent.episode_rewards[-1]
            cumulative = sum(self.agent.episode_rewards)
            self.get_logger().info(f"Episode {self.episode_count} completed. Reward: {total_reward}, Cumulative: {cumulative}")
        
        if self.episode_count % 10 == 0:
            self.save_progress_plot()

    def reset_episode(self):
        """Resets current episode"""
        self.episode_active = False
        self.step_count = 0
        self.current_state = None
        self.last_action = None
        self.get_logger().info("Episode reset")

    def save_progress_plot(self):
        """Saves progress plot"""
        try:
            fig = self.agent.plot_cumulative_reward()
            filename = f"rl_dt_progress_episode_{self.episode_count}.png"
            fig.savefig(filename)
            plt.close(fig)
            self.get_logger().info(f"Progress plot saved: {filename}")
        except Exception as e:
            self.get_logger().error(f"Error saving plot: {e}")

    def shutdown(self):
        """Cleanup when closing node"""
        self.get_logger().info("Shutting down RL-DT Penalty Node")
        
        if len(self.agent.experiences) > 0:
            final_plot = self.agent.plot_cumulative_reward()
            final_plot.savefig("final_rl_dt_results.png")
            plt.close(final_plot)
            
            self.get_logger().info(f"Final Statistics:")
            self.get_logger().info(f"Total Episodes: {self.episode_count}")
            self.get_logger().info(f"Total Experiences: {len(self.agent.experiences)}")
            self.get_logger().info(f"Images processed: {self.image_count}")
            self.get_logger().info(f"Successful detections: {self.detection_count}")

def main(args=None):
    rclpy.init(args=args)
    node = RLDTPenaltyNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()