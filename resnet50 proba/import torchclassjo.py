import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from scipy.spatial.distance import cosine
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# GPU settings
# Be állítja  a GPU-t (ha rendelkezésre áll) a gyorsított számításokhoz.
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Osztály a képminőség ellenőrzéséhez
class ImageQualityChecker:
    def __init__(self, blur_threshold=50, low_exposure=70, high_exposure=180, contrast_threshold=50):
        self.blur_threshold = blur_threshold
        self.low_exposure = low_exposure
        self.high_exposure = high_exposure
        self.contrast_threshold = contrast_threshold

    # Homályosság ellenőrzése a Laplace-operator alapján
    def is_blurry(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Nem sikerült betölteni a képet: {image_path}")
            return False
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold

    # Kép expozíciójának ellenőrzése hisztogram analízissel
    def check_exposure(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Nem sikerült betölteni a képet: {image_path}")
            return "N/A"
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        dark_pixels_ratio = np.sum(hist[:self.low_exposure]) / image.size
        bright_pixels_ratio = np.sum(hist[self.high_exposure:]) / image.size
        if dark_pixels_ratio > 0.4:
            return "alulexponált"
        elif bright_pixels_ratio > 0.4:
            return "túlexponált"
        return "megfelelő expozíció"

    # Kép kontrasztjának ellenőrzése az intenzitáskülönbségek alapján
    def check_contrast(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Nem sikerült betölteni a képet: {image_path}")
            return False
        contrast = np.max(image) - np.min(image)
        return contrast > self.contrast_threshold

    # Teljes képminőség ellenőrzés
    def check_image_quality(self, image_path):
        exposure_status = self.check_exposure(image_path)
        if exposure_status != "megfelelő expozíció":
            return exposure_status
        if self.is_blurry(image_path):
            return "homályos"
        if not self.check_contrast(image_path):
            return "alacsony kontraszt"
        return "jó minőség"

# Osztály a különböző expozíciójú képek egyesítésére
class ExposureFusion:
    def __init__(self):
        self.images = []

    # Kép hozzáadása az egyesítési listához
    def add_image(self, image_path):
        self.images.append(cv2.imread(image_path).astype(np.float32))

    # Képek igazítása ORB jellemzőpontok alapján
    def align_images(self, base_image, image_to_align):
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(base_image, None)
        kp2, des2 = orb.detectAndCompute(image_to_align, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 4:
            print("Figyelmeztetés: Nem sikerült elegendő jellemzőpontot találni az igazításhoz.")
            return image_to_align
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        return cv2.warpPerspective(image_to_align, M, (base_image.shape[1], base_image.shape[0]))

    # Alkalmazza az exposure fusion technikát az egyesített képekhez
    def apply_fusion(self):
        if not self.images:
            raise ValueError("Nincsenek hozzáadott képek a fúzióhoz.")
        base_image = self.images[0]
        aligned_images = [base_image] + [self.align_images(base_image, img) for img in self.images[1:]]
        weights = [cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F) for img in aligned_images]
        weight_sum = np.sum(weights, axis=0)
        normalized_weights = [w / weight_sum for w in weights]
        fusion_image = np.zeros_like(aligned_images[0], dtype=np.float32)
        for img, weight in zip(aligned_images, normalized_weights):
            for c in range(3):
                fusion_image[:, :, c] += img[:, :, c] * weight
        return cv2.convertScaleAbs(fusion_image)

# Osztály a képek közötti hasonlóság mérésére egy előképzett mélytanulási modell alapján
class ImageComparatorAI:
    def __init__(self):
        # Előképzett ResNet50 modell beállítása és GPU-ra helyezése, ha elérhető
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1]).to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Jellemzők kinyerése a resnet50 modell segítségével
    def extract_deep_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.model(image).squeeze().cpu().numpy()

    # Képek hasonlóságának mérése jellemzővektorok alapján
    def compare_images(self, img1_path, img2_path):
        features1 = self.extract_deep_features(img1_path)
        features2 = self.extract_deep_features(img2_path)
        return 1 - cosine(features1, features2)

    # Hasonlósági mátrix létrehozása az összes kép között
    def build_similarity_matrix(self, image_paths):
        """Létrehozza a szimmetrikus hasonlósági mátrixot a képek között, csak a felső háromszög számításaival."""
        n = len(image_paths)
        similarity_matrix = np.zeros((n, n))  # Kezdetben minden érték 0
        
        for i in range(n):
            for j in range(i + 1, n):  # Csak a felső háromszög számításai (i < j)
                similarity = self.compare_images(image_paths[i], image_paths[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Szimmetrikus mátrix kitöltése
        
        return similarity_matrix


# Osztály a képek klaszterezésére és expozíciós fúzió felkínálására
class ImageClusterProcessor:
    def __init__(self, quality_checker, comparator, exposure_fusion, eps=0.2, min_samples=2):
        self.quality_checker = quality_checker
        self.comparator = comparator
        self.exposure_fusion = exposure_fusion
        self.eps = eps
        self.min_samples = min_samples

    # Képek betöltése és minőségi ellenőrzés
    def load_and_filter_images(self, folder_path):
        good_quality_images = []
        low_high_exposure_images = []  # Új lista az alul- vagy túlexponált képekhez
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            if filename.endswith(('jpg', 'jpeg', 'png')):
                quality_status = self.quality_checker.check_image_quality(image_path)
                if quality_status == "jó minőség":
                    good_quality_images.append(image_path)
                elif quality_status in ["alulexponált", "túlexponált"]:
                    low_high_exposure_images.append(image_path)  # Alul- vagy túlexponált képeket is megtartjuk
                else:
                    print(f"{filename} nem felel meg: {quality_status}")
        return good_quality_images + low_high_exposure_images  # Mindkét listát visszaadjuk

    # Hasonlóság alapú klaszterezés hasonlósági mátrix alapján
    def create_clusters(self, image_paths):
        similarity_matrix = self.comparator.build_similarity_matrix(image_paths)
        distance_matrix = 1 - similarity_matrix
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
        clusters = db.fit_predict(distance_matrix)

        # Képek csoportosítása klaszterenként
        clustered_images = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id != -1:
                if cluster_id not in clustered_images:
                    clustered_images[cluster_id] = []
                clustered_images[cluster_id].append(image_paths[idx])

        return clustered_images

    # Klaszterek feldolgozása és expozíciós fúzió felajánlása, ha szükséges
    def process_clusters(self, clusters):
        for cluster_id, images in clusters.items():
            print(f"\nKlaszter {cluster_id} képei:")
            for image_path in images:
                print(f"  - {os.path.basename(image_path)}")

            low_exposure_images = [img for img in images if self.quality_checker.check_exposure(img) == "alulexponált"]
            high_exposure_images = [img for img in images if self.quality_checker.check_exposure(img) == "túlexponált"]

            if low_exposure_images or high_exposure_images:
                for img_path in low_exposure_images + high_exposure_images:
                    self.exposure_fusion.add_image(img_path)

                # Felhasználói jóváhagyás expozíciós fúzióhoz
                if input(f"Szeretné használni az Exposure Fusion funkciót a klaszter {cluster_id} képeinek egyesítéséhez? (i/n): ").strip().lower() == "i":
                    fusion_image = self.exposure_fusion.apply_fusion()
                    output_path = f"exposure_fusion_cluster_{cluster_id}.jpg"
                    success = cv2.imwrite(output_path, fusion_image)
                    if success:
                        print(f"A klaszter {cluster_id} fúziós képe elmentésre került: {output_path}")
                    else:
                        print(f"Hiba történt a {output_path} mentése során.")
                    cv2.imshow(f"Exposure Fusion Cluster {cluster_id}", fusion_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print(f"A klaszter {cluster_id} nem tartalmaz különböző expozíciójú képeket.")

# Fő program futtatása
folder_path = "c:\\Users\\varda\\Documents\\szakdolgozat\\opencv app\\proba kod\\dataset\\skittles"
quality_checker = ImageQualityChecker()
comparator = ImageComparatorAI()
exposure_fusion = ExposureFusion()
processor = ImageClusterProcessor(quality_checker, comparator, exposure_fusion)

good_quality_images = processor.load_and_filter_images(folder_path)
clusters = processor.create_clusters(good_quality_images)
processor.process_clusters(clusters)

# Hőtérkép megjelenítése
if good_quality_images:
    similarity_matrix = comparator.build_similarity_matrix(good_quality_images)
    file_names = [os.path.basename(path) for path in good_quality_images]

    # Alsó háromszögmátrix létrehozása
    lower_triangle = np.tril(similarity_matrix)  # Csak az alsó háromszöget tartja meg

    # Hőtérkép megjelenítése az alsó háromszögmátrix alapján
    plt.figure(figsize=(10, 8))
    sns.heatmap(lower_triangle, annot=True, cmap="coolwarm", xticklabels=file_names, yticklabels=file_names, mask=(lower_triangle == 0))
    plt.title("Képek közötti alsó háromszögmátrix hasonlósági mátrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

else:
    print("Nincsenek jó minőségű képek a további feldolgozáshoz.")

