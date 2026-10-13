class Image:
    """Stand-in for sensor_msgs.msg.Image. Real one is a wire message;
    here it's just a container so cv_bridge mock and the node can pass
    something around with a .header attribute."""
    def __init__(self):
        self.header = None
        self.data = None
