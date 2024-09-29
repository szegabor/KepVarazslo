
class ImageSelector:
    def __init__(self):
        """Konstruktor az alapértelmezett beállításokkal."""
        pass

    def calculate_image_quality(self, image):
        """Kiszámítja a kép minőségét (jelenleg a kép mérete alapján)."""
        return image.shape[0] * image.shape[1]  # Kép területe, mint minőségmutató

    def select_best_photo(self, photo_group):
        """Kiválasztja a legjobb minőségű képet egy csoportból."""
        best_photo = None
        best_quality = -1
        for photo_name, photo in photo_group.items():
            quality = self.calculate_image_quality(photo)
            if quality > best_quality:
                best_quality = quality
                best_photo = photo_name
        return best_photo

    def select_best_photos(self, similar_groups):
        """Minden csoportból kiválasztja a legjobb fotót."""
        best_photos = []
        for group in similar_groups:
            best_photo = self.select_best_photo(group)
            best_photos.append(best_photo)
        return best_photos
