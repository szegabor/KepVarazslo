
from photo_album import PhotoAlbum
from image_comparator import ImageComparator
from image_selector import ImageSelector
from image_enhancer import ImageEnhancer
from image_cleaner import ImageCleaner
from image_processor import ImageProcessor
from photo_manager_gui import PhotoManagerGUI

class PhotoManagerApp:
    def __init__(self):
        """Az alkalmazás inicializálása."""
        self.album = None
        self.comparator = ImageComparator()
        self.selector = ImageSelector()
        self.enhancer = ImageEnhancer()
        self.cleaner = ImageCleaner(self.enhancer)
        self.processor = ImageProcessor()

    def load_album(self, directory):
        """Betölti az albumot a megadott könyvtárból."""
        return PhotoAlbum(directory)

    def run(self):
        """Elindítja a grafikus felületet."""
        gui = PhotoManagerGUI(self)
        gui.root.mainloop()
