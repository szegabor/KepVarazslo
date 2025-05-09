import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.quality_checker import ImageQualityChecker
from modules.comparator import ImageComparatorAI
from modules.cluster_processor import ImageClusterProcessor
from modules.exposure_fusion import ExposureFusion

class ImageProcessingPipeline:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.quality_checker = ImageQualityChecker()
        self.comparator = ImageComparatorAI()
        self.cluster_processor = ImageClusterProcessor(self.comparator)
        self.exposure_fusion = ExposureFusion()

    def load_images(self):
        return [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    def run(self):
        images = self.load_images()
        clusters = self.cluster_processor.create_clusters(images)  # Az összes kép klaszterezése

        for cluster_id, imgs in clusters.items():
            print(f"\n **Klaszter {cluster_id} képei:**")
            for img in imgs:
                score = self.quality_checker.get_quality_score(img)
                exposure = self.quality_checker.check_exposure(img)
                is_blurry = self.quality_checker.is_blurry(img)
                contrast = self.quality_checker.check_contrast(img)
                
                print(f"  - {os.path.basename(img)} → Expozíció: {exposure}, "
                      f"Homályos: {'Igen' if is_blurry else 'Nem'}, "
                      f"Kontraszt: {'Alacsony' if not contrast else 'Jó'}, "
                      f"Pontszám: {score}/100")

            #  Rossz minőségű képek keresése
            low_quality_images = [img for img in imgs if self.quality_checker.check_image_quality(img) != "jó minőség"]
            
            #  Ha vannak rossz minőségű képek a klaszterben, felajánlja az expozíciós fúziót
            if low_quality_images:
                self.offer_exposure_fusion(cluster_id, imgs)

            best_images = self.get_best_images(imgs)
            print(f"\n **Klaszter {cluster_id} legjobb képe(i):**")
            for img in best_images:
                print(f"  - {os.path.basename(img)}")

        #  Rossz minőségű képek törlési lehetősége
        self.handle_deletion(images)

        #  Hőtérkép generálása
        self.generate_heatmap(images)

    def offer_exposure_fusion(self, cluster_id, images):
        """ Csak a rossz minőségű képekkel rendelkező klasztereknél ajánlja fel az expozíciós fúziót. """
        low_quality_images = [img for img in images if self.quality_checker.check_image_quality(img) != "jó minőség"]

        if low_quality_images:
            print(f"\n **A klaszter {cluster_id} tartalmaz rossz minőségű képeket.**")
            print(f"Szeretné használni az Exposure Fusion funkciót a klaszter {cluster_id} rossz minőségű képeinek egyesítéséhez? (i/n): ", end="")
            user_input = input().strip().lower()

            if user_input == "i":
                for img_path in images:  # Minden kép bekerül a fúzióba
                    self.exposure_fusion.add_image(img_path)

                if len(self.exposure_fusion.images) > 1:
                    fusion_image = self.exposure_fusion.apply_fusion()
                    output_path = os.path.join(self.folder_path, f"exposure_fusion_cluster_{cluster_id}.jpg")
                    success = cv2.imwrite(output_path, fusion_image)
                    if success:
                        print(f" A klaszter {cluster_id} fúziós képe elmentésre került: {output_path}")
                    else:
                        print(f" Hiba történt a {output_path} mentése során.")
                else:
                    print(" Nem elegendő kép áll rendelkezésre az fúzióhoz.")

    def get_best_images(self, images):
        best_score = -1
        best_images = []
        for img in images:
            score = self.quality_checker.get_quality_score(img)
            if score > best_score:
                best_score = score
                best_images = [img]
            elif score == best_score:
                best_images.append(img)
        return best_images

    def handle_deletion(self, images):
        """ Listázza a törlésre javasolt képeket és felajánlja a törlést. """
        deletion_candidates = [img for img in images if self.quality_checker.check_image_quality(img) != "jó minőség"]
        if deletion_candidates:
            print("\n **Javasolt törlésre kerülő képek:**")
            for img in deletion_candidates:
                print(f"  - {os.path.basename(img)}")

            user_input = input("\nSzeretné törölni a javasolt képeket? (i/n): ").strip().lower()
            if user_input == 'i':
                for img in deletion_candidates:
                    if os.path.exists(img):
                        try:
                            os.remove(img)
                            print(f" Törölve: {os.path.basename(img)}")
                        except Exception as e:
                            print(f" Hiba történt a törléskor: {img}, {e}")
            else:
                print(" A képek megtartásra kerültek.")

    def generate_heatmap(self, image_paths):
        similarity_matrix = self.comparator.build_similarity_matrix(image_paths)
        file_names = [os.path.basename(path) for path in image_paths]
        lower_triangle = np.tril(similarity_matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(lower_triangle, annot=True, cmap="coolwarm", xticklabels=file_names, yticklabels=file_names, mask=(lower_triangle == 0))
        plt.title(" Képek közötti hasonlósági mátrix")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
