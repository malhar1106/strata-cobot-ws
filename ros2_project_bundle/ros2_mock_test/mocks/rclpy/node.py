"""
Minimal stand-in for rclpy, just enough surface area for
ColorBoxDetectorNode to import and run without a real ROS2 install.

This is a TEST DOUBLE -- it does not do real pub/sub, discovery, or
message serialization. It only proves the node's Python logic
(callback wiring, parameter handling, image conversion calls) runs
without error. It is NOT a substitute for testing against a real
ROS2 install / colcon build -- do that before deploying.
"""


class _Logger:
    def info(self, msg):
        print(f'[INFO] {msg}')

    def error(self, msg):
        print(f'[ERROR] {msg}')

    def debug(self, msg):
        print(f'[DEBUG] {msg}')


class _FakePublisher:
    """Records every message published so the test can inspect it."""
    def __init__(self, topic):
        self.topic = topic
        self.published = []

    def publish(self, msg):
        self.published.append(msg)
        print(f'[PUBLISH -> {self.topic}] {type(msg).__name__}')


class _FakeParam:
    def __init__(self, value):
        self._value = value

    def get_parameter_value(self):
        return self

    @property
    def string_value(self):
        return self._value


class Node:
    """Stand-in for rclpy.node.Node."""
    def __init__(self, name):
        self._name = name
        self._params = {}
        self._logger = _Logger()

    def declare_parameter(self, name, default):
        self._params[name] = default

    def get_parameter(self, name):
        return _FakeParam(self._params[name])

    def create_subscription(self, msg_type, topic, callback, qos):
        print(f'[SUBSCRIBE] {topic} ({msg_type.__name__})')
        return callback  # test drives the callback directly

    def create_publisher(self, msg_type, topic, qos):
        return _FakePublisher(topic)

    def get_logger(self):
        return self._logger

    def destroy_node(self):
        pass


def init(args=None):
    print('[rclpy.init] (mock)')


def spin(node):
    print('[rclpy.spin] (mock - no-op, single callback test drives this manually)')


def shutdown():
    print('[rclpy.shutdown] (mock)')
