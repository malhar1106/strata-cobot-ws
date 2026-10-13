class CvBridge:
    """Stand-in for cv_bridge.CvBridge. The real one decodes the wire
    encoding (rgb8/bgr8/etc.) into a numpy array and back. Here, since
    our fake Image just stores a numpy array directly in .data, this
    is a passthrough -- it exists so the node's calls to
    imgmsg_to_cv2/cv2_to_imgmsg don't need special-casing in the node
    code itself."""

    def imgmsg_to_cv2(self, msg, desired_encoding='bgr8'):
        return msg.data

    def cv2_to_imgmsg(self, cv_image, encoding='bgr8'):
        from sensor_msgs.msg import Image
        msg = Image()
        msg.data = cv_image
        return msg
