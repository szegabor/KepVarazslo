import cv2
import numpy as np
import os

class ImageQualityChecker:
    def __init__(self, blur_threshold=50, low_exposure=70, high_exposure=180, contrast_threshold=50):
        self.blur_threshold = blur_threshold
        self.low_exposure = low_exposure
        self.high_exposure = high_exposure
        self.contrast_threshold = contrast_threshold

    def is_blurry(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return cv2.Laplacian(image, cv2.CV_64F).var() < self.blur_threshold if image is not None else False

    def check_exposure(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "N/A"
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        dark_ratio = np.sum(hist[:self.low_exposure]) / image.size
        bright_ratio = np.sum(hist[self.high_exposure:]) / image.size
        return "alulexponált" if dark_ratio > 0.4 else "túlexponált" if bright_ratio > 0.4 else "megfelelő"

    def check_contrast(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return np.max(image) - np.min(image) > self.contrast_threshold if image is not None else False

    def check_image_quality(self, image_path):
        exposure_status = self.check_exposure(image_path)
        if exposure_status != "megfelelő":
            return exposure_status
        if self.is_blurry(image_path):
            return "homályos"
        if not self.check_contrast(image_path):
            return "alacsony kontraszt"
        return "jó minőség"

    def get_quality_score(self, image_path):
        """ Minőségi pontszám számítása részletesen, 100-as skálán """
        score = 0

        # Homályosság ellenőrzése (0-40 pont)
        laplacian_var = cv2.Laplacian(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var()
        blur_score = min(40, (laplacian_var / self.blur_threshold) * 40)
        score += blur_score

        # Expozíció ellenőrzése (0-20 pont)
        exposure_status = self.check_exposure(image_path)
        if exposure_status == "megfelelő":
            score += 20  # Teljes pontszám jó expozíció esetén
        elif exposure_status == "alulexponált" or exposure_status == "túlexponált":
            hist = cv2.calcHist([cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)], [0], None, [256], [0, 256])
            brightness = np.sum(hist) / hist.size
            exposure_score = max(0, 20 - abs(128 - brightness) / 6)
            score += exposure_score

        # Kontraszt ellenőrzése (0-20 pont)
        contrast = np.max(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)) - np.min(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        contrast_score = min(20, (contrast / self.contrast_threshold) * 20)
        score += contrast_score

        # Éldetektálás részletesség (0-10 pont)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        detail_score = min(10, np.mean(edge_magnitude) / 2)
        score += detail_score

        # Színtelítettség (0-10 pont)
        hsv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2HSV)
        saturation_channel = hsv[:, :, 1]
        saturation_score = min(10, np.mean(saturation_channel) / 25)
        score += saturation_score

        return round(score, 2)

