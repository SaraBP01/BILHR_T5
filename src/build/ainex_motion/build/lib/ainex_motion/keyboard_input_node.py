import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16

# Define a class for the node responsible to read keyboard inputs
class KeyboardInputNode(Node):
    def __init__(self):
        super().__init__('keyboard_input_node')
        # Create a publisher for the keyboard input commands
        self.publisher_ = self.create_publisher(Int16, 'keyboard_command', 10)
        self.get_logger().info('Node started. Type a number between 1 and 5 and press Enter.')

        self.run_input_loop()

    def run_input_loop(self):
        try:
            while rclpy.ok():
                key = input(">> ")
                if key in ['1', '2', '3', '4', '5']:
                    # Initialize the message and assign the input key to it
                    msg = Int16()
                    msg.data = int(key)
                    # Publish the message
                    self.publisher_.publish(msg)
                    # self.get_logger().info(f'Published command: {msg.data}')
                else:
                    # The robot only performs a motion if an integer between 1 and 5 is received as input
                    print("Only numbers 1, 2, 3, 4 or 5.")
        except KeyboardInterrupt:
            print("Finalizing node...")

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardInputNode()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

