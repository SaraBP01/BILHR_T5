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

# Comment out the problematic import for now
# from .rl_dt_penalty_kick import RLDT  # Import our RL-DT class

# Temporary placeholder class
class RLDT:
    def __init__(self, max_reward=20, discount_factor=0.9):
        self.max_reward = max_reward
        self.discount_factor = discount_factor
        self.episode_rewards = []
        self.experiences = []
        self.exploration_mode = True
    
    def discretize_state(self, marker_pos, leg_pos):
        return f"state_{int(marker_pos)}_{int(leg_pos)}"
    
    def select_action(self, state):
        return np.random.choice(['MOVE_LEFT', 'MOVE_RIGHT', 'KICK'])
    
    def add_experience(self, state, action, reward, next_state):
        self.experiences.append((state, action, reward, next_state))
    
    def value_iteration(self):
        pass
    
    def plot_cumulative_reward(self):
        fig, ax = plt.subplots()
        ax.plot(self.episode_rewards)
        return fig

class RLDTPenaltyNode(Node):
    def __init__(self):  
        super().__init__('rl_dt_penalty_node') 
        
        # Initialize RL-DT agent
        self.agent = RLDT(max_reward=20, discount_factor=0.9)
        
        # ROS2 Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/rl_dt_status', 10)
        
        # CV Bridge for images
        self.bridge = CvBridge()
        
        # ArUco detector setup with extensive logging
        self.aruco_dict = None
        self.aruco_params = None
        self.setup_aruco_detector()
        
    
        if self.aruco_dict is not None and self.aruco_params is not None:
            self.get_logger().info("ArUco initialized successfully, creating subscriptions...")
            self.create_subscriptions()
        else:
            self.get_logger().error("ArUco initialization failed! Node will not function properly.")
            return
        
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

    def create_subscriptions(self):
        """Create ROS2 subscriptions only after ArUco is properly initialized"""
        # ROS2 Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        
        
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.image_sub = self.create_subscription(
            Image, '/camera_image', self.image_callback, qos_profile)
        
        self.reward_sub = self.create_subscription(
            Float32, '/penalty_reward', self.reward_callback, 10)
        self.reset_sub = self.create_subscription(
            Bool, '/reset_episode', self.reset_callback, 10)
        
        self.get_logger().info("All subscriptions created successfully")

    def setup_aruco_detector(self):
        self.get_logger().info("Setting up ArUco detector...")

        # Check OpenCV version
        cv_version = cv2.__version__.split(".")
        major = int(cv_version[0])
        minor = int(cv_version[1])

        try:
            if major >= 4 and minor >= 7:
                # Use new API (4.7+)
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
                self.aruco_params = cv2.aruco.DetectorParameters()
                self.use_new_aruco_api = True
                self.get_logger().info("ArUco initialized with OpenCV 4.7+ API")
            else:
                # Use legacy API (4.6 and older)
                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
                self.use_new_aruco_api = False
                self.get_logger().info("ArUco initialized with OpenCV <=4.6 API")
        except Exception as e:
            self.get_logger().error(f"ArUco initialization failed: {e}")
            self.aruco_dict = None
            self.aruco_params = None
            self.use_new_aruco_api = False
                    
            if self.aruco_dict is not None and self.aruco_params is not None:
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
        self.image_count += 1
        
        try:
            # Convert image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Validate image
            if cv_image is None or cv_image.size == 0:
                self.get_logger().error("Invalid image data received")
                return
                
         
            if self.aruco_dict is None or self.aruco_params is None:
                self.get_logger().warn("ArUco not initialized, skipping detection")
                return
            
            # Store current image
            self.current_image = cv_image
            
            # Detect markers safely
            marker_pos, marker_id = self.detect_multiple_aruco_markers(cv_image)
            
            if marker_pos is not None:
                self.marker_position = marker_pos
                self.marker_id = marker_id
                self.detection_count += 1
                
                if self.image_count % 30 == 0:
                    self.get_logger().info(f"Detection success rate: {self.detection_count}/{self.image_count}")
            
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

    def detect_multiple_aruco_markers(self, image):
        """Detect ArUco markers in the given image using OpenCV 4.6+ or 4.7+"""
        if self.aruco_dict is None or self.aruco_params is None:
            self.get_logger().warn("ArUco dictionary not initialized, using fallback")
            return self.detect_fallback_target(image)

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Choose detection method based on OpenCV version
            if getattr(self, "use_new_aruco_api", False):
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.aruco_params)

            # Draw detections and return center of first marker
            if ids is not None and len(ids) > 0:
                self.get_logger().info(f"Detected ArUco IDs: {ids.flatten().tolist()}")

                # Draw all detected markers
                cv2.aruco.drawDetectedMarkers(image, corners, ids)

                # Use the first detected marker for positioning
                marker_id = int(ids[0][0])
                corner = corners[0][0]
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))

                cv2.circle(image, (center_x, center_y), 8, (0, 255, 0), 3)
                cv2.putText(image, f"ID:{marker_id}",
                            (center_x + 10, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display the image
                cv2.imshow("RL-DT ArUco Detection", image)
                cv2.waitKey(1)

                return center_x, marker_id

            else:
                # No markers found
                cv2.putText(image, "NO MARKERS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("RL-DT ArUco Detection", image)
                cv2.waitKey(1)

                self.get_logger().warn("No ArUco markers detected.")
                return None, None

        except Exception as e:
            self.get_logger().error(f"Error during ArUco detection: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None, None
    
        
    def detect_fallback_target(self, image):
        """Fallback detection using color detection"""
        try:
            # Simple fallback - detect center of image as target
            center_x = image.shape[1] // 2
            center_y = image.shape[0] // 2
            
            cv2.circle(image, (center_x, center_y), 8, (0, 0, 255), 3)
            cv2.putText(image, "FALLBACK", (center_x + 15, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('RL-DT ArUco Detection', image)
            cv2.waitKey(1)
            
            return center_x, 999
            
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
        
        # Publish status
        status_msg = String()
        status_msg.data = f"Episode: {self.episode_count}, Step: {self.step_count}, Action: {action}"
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
        
        self.get_logger().info(f"Starting episode {self.episode_count}")

    def end_episode(self):
        """Ends current episode"""
        self.episode_active = False
        self.agent.value_iteration()
        
        if len(self.agent.episode_rewards) > 0:
            total_reward = self.agent.episode_rewards[-1]
            self.get_logger().info(f"Episode {self.episode_count} completed. Reward: {total_reward}")

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
        cv2.destroyAllWindows()

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