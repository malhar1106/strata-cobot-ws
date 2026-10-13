from setuptools import find_packages, setup

package_name = 'color_box_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='OpenCV HSV color-box detection node for a camera feed',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'color_box_detector_node = color_box_detector.color_box_detector_node:main',
            'ore_tf_broadcaster_node = color_box_detector.ore_tf_broadcaster_node:main',
        ],
    },
)
