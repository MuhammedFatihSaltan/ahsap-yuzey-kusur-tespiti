import cv2
import numpy as np

class ImagePreprocessor:
    """Görüntü ön işleme ve kontrast iyileştirme sınıfı."""
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü gri tona dönüştürür."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def apply_clahe(self, gray_image: np.ndarray) -> np.ndarray:
        """Adaptif Histogram Eşitleme (CLAHE) uygular."""
        return self.clahe.apply(gray_image)

    def reduce_noise(self, gray_image: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """Gürültü azaltma filtresi uygular."""
        if method == "gaussian":
            return cv2.GaussianBlur(gray_image, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(gray_image, 5)
        elif method == "bilateral":
            return cv2.bilateralFilter(gray_image, 9, 75, 75)
        return gray_image

    def preprocess(self, image: np.ndarray) -> dict:
        """Tüm ön işleme adımlarını çalıştırıp ara çıktıları döndürür."""
        gray = self.to_grayscale(image)
        enhanced = self.apply_clahe(gray)
        denoised = self.reduce_noise(enhanced, method="gaussian")
        
        return {
            "gray": gray,
            "enhanced": enhanced,
            "denoised": denoised
        }