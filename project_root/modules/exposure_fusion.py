import cv2
import numpy as np

class ExposureFusion:
    def __init__(self):
        self.images = []

    def add_image(self, image_path):
        image = cv2.imread(image_path)
        if image is not None:
            self.images.append(image.astype(np.float32))

    def apply_fusion(self):
        if len(self.images) < 2:
            return None
        fusion_image = np.mean(self.images, axis=0)
        return cv2.convertScaleAbs(fusion_image)
