import cv2

class ImageEnhancer:
    def __init__(self):
        """Alapértelmezett beállításokkal rendelkező konstruktor."""
        pass

    def detect_blur(self, image, threshold=100.0):
        """Elmosódottság felismerése egy képen a Laplace módszerrel."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold  # Ha kisebb a variancia, a kép elmosódott

    def detect_overexposure(self, image, threshold=240):
        """Túlexponáltság felismerése: túl sok világos pixel."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        overexposed_pixels = (gray > threshold).sum()
        total_pixels = gray.size
        overexposed_ratio = overexposed_pixels / total_pixels
        return overexposed_ratio > 0.1  # Ha több mint 10% túlexponált

    def enhance_contrast(self, image):
        """A kép kontrasztjának javítása hisztogram kiegyenlítéssel."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        enhanced_image = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)
