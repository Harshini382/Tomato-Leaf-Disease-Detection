import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class ImageMetrics:
    """Calculate various image quality and analysis metrics."""
    
    @staticmethod
    def calculate_psnr(original, processed):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        Higher PSNR indicates better quality.
        
        Args:
            original: Original image array
            processed: Processed/reconstructed image array
            
        Returns:
            float: PSNR value in dB
        """
        # Convert to same dtype and range
        original = original.astype(np.float32)
        processed = processed.astype(np.float32)
        
        # Normalize to 0-255 range if needed
        if original.max() <= 1.0:
            original = original * 255
        if processed.max() <= 1.0:
            processed = processed * 255
        
        # Calculate MSE
        mse = np.mean((original - processed) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return round(psnr, 2)
    
    @staticmethod
    def calculate_entropy(image):
        """
        Calculate Shannon entropy of the image.
        Measures the amount of information/randomness in the image.
        Higher entropy indicates more variation in pixel intensities.
        
        Args:
            image: Image array (grayscale or single channel)
            
        Returns:
            float: Entropy value
        """
        # Convert to uint8 if necessary
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Flatten the image
        flat_image = image.flatten()
        
        # Calculate histogram
        hist, _ = np.histogram(flat_image, bins=256, range=(0, 256))
        
        # Normalize histogram to get probabilities
        hist = hist / hist.sum()
        
        # Calculate entropy
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        
        return round(entropy, 2)
    
    @staticmethod
    def calculate_contrast_index(image):
        """
        Calculate contrast index using RMS (Root Mean Square) contrast.
        Measures the standard deviation of pixel values.
        
        Args:
            image: Image array
            
        Returns:
            float: Contrast index value
        """
        # Convert to float
        image = image.astype(np.float32)
        
        # Normalize to 0-1 range if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Calculate mean
        mean = np.mean(image)
        
        # Calculate RMS contrast
        rms_contrast = np.sqrt(np.mean((image - mean) ** 2))
        
        return round(rms_contrast, 4)
    
    @staticmethod
    def calculate_histogram_spread(image):
        """
        Calculate histogram spread analysis.
        Analyzes how the pixel values are distributed across the histogram.
        
        Args:
            image: Image array (grayscale)
            
        Returns:
            dict: Contains spread metrics including:
                  - mean: Mean of histogram values
                  - std: Standard deviation of histogram values
                  - spread: Ratio of occupied bins to total bins
                  - skewness: Distribution skewness
                  - kurtosis: Distribution peakedness
        """
        # Convert to uint8 if necessary
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Calculate histogram
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        
        # Calculate metrics
        mean_hist = np.mean(hist)
        std_hist = np.std(hist)
        
        # Calculate spread (percentage of non-empty bins)
        non_zero_bins = np.count_nonzero(hist)
        spread = (non_zero_bins / 256) * 100
        
        # Calculate skewness and kurtosis using normalized histogram
        hist_normalized = hist / hist.sum()
        x = np.arange(256)
        
        # Mean and variance of distribution
        mu = np.sum(x * hist_normalized)
        sigma = np.sqrt(np.sum((x - mu) ** 2 * hist_normalized))
        
        # Skewness
        if sigma > 0:
            skewness = np.sum(((x - mu) / sigma) ** 3 * hist_normalized)
        else:
            skewness = 0
        
        # Kurtosis
        if sigma > 0:
            kurtosis = np.sum(((x - mu) / sigma) ** 4 * hist_normalized) - 3
        else:
            kurtosis = 0
        
        return {
            'mean_hist_value': round(mean_hist, 2),
            'std_hist_value': round(std_hist, 2),
            'spread_percentage': round(spread, 2),
            'skewness': round(skewness, 4),
            'kurtosis': round(kurtosis, 4),
            'mean_pixel_intensity': round(mu, 2),
            'std_pixel_intensity': round(sigma, 2)
        }
    
    @staticmethod
    def calculate_all_metrics(original_image, processed_image):
        """
        Calculate all metrics for the image.
        
        Args:
            original_image: Original image array
            processed_image: Processed/enhanced image array
            
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        # Ensure images are in the same format for PSNR
        if len(original_image.shape) == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original_image
        
        if len(processed_image.shape) == 3:
            processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        else:
            processed_gray = processed_image
        
        metrics = {
            'psnr': ImageMetrics.calculate_psnr(original_gray, processed_gray),
            'entropy': ImageMetrics.calculate_entropy(processed_gray),
            'contrast_index': ImageMetrics.calculate_contrast_index(processed_gray),
            'histogram_spread': ImageMetrics.calculate_histogram_spread(processed_gray)
        }
        
        return metrics
