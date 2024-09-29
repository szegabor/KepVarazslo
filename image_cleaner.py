
class ImageCleaner:
    def __init__(self, enhancer):
        """Konstruktor az enhancer osztállyal."""
        self.enhancer = enhancer

    def is_low_quality(self, image):
        """Gyenge minőségű képet felismer (elmosódott vagy túlexponált)."""
        is_blurred = self.enhancer.detect_blur(image)
        is_overexposed = self.enhancer.detect_overexposure(image)
        return is_blurred or is_overexposed

    def suggest_deletion(self, photos):
        """Törlésre javasolja a gyenge minőségű képeket."""
        low_quality_photos = []
        for photo_name, image in photos.items():
            if self.is_low_quality(image):
                low_quality_photos.append(photo_name)
        return low_quality_photos
