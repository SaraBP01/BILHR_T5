from setuptools import setup

package_name = 'ainex_neural_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='esther.utasa@gmail.com',
    description='Neural vision package with MLP and CMAC controllers',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_neural_controller = ainex_neural_control.robot_neural_controller:main', 
            'robot_cmac_controller = ainex_neural_control.robot_cmac_controller:main',
            'rl_dt_penalty_node = ainex_neural_control.rl_dt_penalty_node:main',
            'manual_reward_node = ainex_neural_control.manual_reward_node:main',
        ],
    },
)
