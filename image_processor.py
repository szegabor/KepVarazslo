
import numpy as np
import cv2

class ImageProcessor:
    def combine_images(self, images):
        """Kombinál több képet egyetlen képbe átlagolással."""
        if not images:
            return None
        
        # Átalakítjuk a képeket float típusúvá az átlagolás előtt
        combined_image = np.zeros_like(images[0], dtype=np.float32)
        
        for image in images:
            combined_image += image.astype(np.float32)
        
        # Az átlagolt kép létrehozása
        combined_image /= len(images)
        
        # Visszaalakítjuk uint8 típusra
        return combined_image.astype(np.uint8)
