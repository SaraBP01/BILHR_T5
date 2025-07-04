from ainex_motion.joint_controller import JointController
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStatePublisher(Node):
    """
    Goal: Retreive the joint positions of the robot and publish as JointState messages
    """
    def __init__(self):
        super().__init__('joint_state_publisher')
        # instaintiate the JointController with the current node
        self.joint_controller = JointController(self)
        
        # TODO: define a publisher for the joint states 
        # with the topic name 'joint_states'
        # and message type sensor_msgs/JointState
        self.joint_states_pub = self.create_publisher(JointState, "joint_states", 10)
        

    def publish_joint_states(self):
        # TODO: Retrieve current joint positions with the getJointPositions method
        # and publish them to the 'joint_states' topic
        # Hint: Check the message definition here: https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/JointState.html
        # Set the joints velocity and effort to 0.0, header.stamp to the current time
        # Extract the list of joint names
        joint_names = list(self.joint_controller.joint_id.keys())
        
        # Initalize the joint state message, extract the positions of the joints according to joint names
        msg = JointState()
        joint_pos = self.joint_controller.getJointPositions(joint_names, unit = 'rad')
        # Set header stamp to the current time
        msg.header.stamp =  self.get_clock().now().to_msg()
        # Set message name to joint names, message positions to joint positions, velocities and effort to zero.
        msg.name = joint_names
        msg.position = joint_pos
        msg.velocity = [0.0] * len(joint_names)
        msg.effort = [0.0] * len(joint_names)
        self.get_logger().info(f"Published names: {joint_names}")
        # Publish the message
        self.joint_states_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    joint_state_publisher = JointStatePublisher()

    while rclpy.ok():
        joint_state_publisher.publish_joint_states()

    joint_state_publisher.destroy_node()
    rclpy.shutdown()
