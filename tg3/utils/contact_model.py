from skimage.metrics import structural_similarity as ssim


class ContactModel:
    """ Simple contact detection model based on image similarity (SSIM). """

    def __init__(self, reference=None, threshold=0.95, delay=5, ros2_topic=None):
        self.reference, self.threshold, self.delay = reference, threshold, delay
        self._previous_image = [None,]*delay
        self.ros2_topic = ros2_topic

    def predict(self, image):
        """ Predicts contact based on similarity between current tactile image and reference image."""
        reference = self.reference if self.reference is not None else image
        ssim_value = ssim(image.squeeze(), reference.squeeze(), data_range=255)
        is_contact = (ssim_value < self.threshold)

        if self.ros2_topic is not None:
            self.ros2_topic(is_contact, ssim_value)

        return is_contact, ssim_value
    
    def delta(self, image):
        """ Detects significant change in tactile image compared to delayed previous image. """
        reference = self._previous_image[0] if self._previous_image[0] is not None else image
        self._previous_image = self._previous_image[1:] + [image]

        ssim_value = ssim(image, reference, data_range=255)
        is_change = (ssim_value < self.threshold)

        return is_change, ssim_value