
from skimage.metrics import structural_similarity as ssim
import cv2

class ImageComparator:
    def compare_images(self, imageA, imageB):
        """Összehasonlítja két kép hasonlóságát az SSIM alapján."""
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(grayA, grayB, full=True)
        return score

    def find_similar_photos(self, photos, threshold=0.9):
        """Megkeresi a hasonló képeket a megadott threshold alapján."""
        similar_groups = []
        photo_names = list(photos.keys())
        for i, name1 in enumerate(photo_names):
            group = {name1: photos[name1]}
            for j, name2 in enumerate(photo_names):
                if i != j and self.compare_images(photos[name1], photos[name2]) > threshold:
                    group[name2] = photos[name2]
            if len(group) > 1:
                similar_groups.append(group)
        return similar_groups
