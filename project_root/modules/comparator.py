import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from scipy.spatial.distance import cosine
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageComparatorAI:
    def __init__(self):
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1]).to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.model(image).squeeze().cpu().numpy()

    def compare_images(self, img1, img2):
        return 1 - cosine(self.extract_features(img1), self.extract_features(img2))

    def build_similarity_matrix(self, image_paths):
        """ Létrehozza a hasonlósági mátrixot az összes kép között """
        n = len(image_paths)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.compare_images(image_paths[i], image_paths[j])
                similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
        
        return similarity_matrix
