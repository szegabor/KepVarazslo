
import cv2
import os

class PhotoAlbum:
    def __init__(self, directory):
        """Az album inicializálása, a képek betöltése a könyvtárból."""
        self.directory = directory
        self.photos = self.load_photos()

    def load_photos(self):
        """Betölti az összes képet a megadott könyvtárból."""
        photo_files = [f for f in os.listdir(self.directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return {photo: cv2.imread(os.path.join(self.directory, photo)) for photo in photo_files}

    def get_photos(self):
        """Visszaadja az összes betöltött képet."""
        return self.photos
