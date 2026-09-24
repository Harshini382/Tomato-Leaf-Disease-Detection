import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import io
import base64
from tensorflow.keras.models import Model
from image_metrics import ImageMetrics

matplotlib.use('Agg')

class FeatureVisualization:
    """Extract and visualize CNN feature maps and histograms."""
    
    @staticmethod
    def plot_to_base64(fig):
        """Convert matplotlib figure to base64 encoded string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    @staticmethod
    def extract_feature_maps(feature_extractor, image, layer_index=-2):
        """
        Extract feature maps from a specific layer of the feature extractor.
        
        Args:
            feature_extractor: Keras feature extractor model
            image: Input image (normalized 0-1)
            layer_index: Index of layer to extract features from
            
        Returns:
            Feature maps array
        """
        if feature_extractor is None:
            # Use alternative method: extract simple image features using edge detection
            return FeatureVisualization._extract_simple_features(image)
        
        # Get the layer output
        layer_model = Model(inputs=feature_extractor.input, 
                           outputs=feature_extractor.layers[layer_index].output)
        
        # Expand dimensions for batch
        image_batch = np.expand_dims(image, axis=0)
        
        # Get feature maps
        feature_maps = layer_model.predict(image_batch, verbose=0)
        
        return feature_maps[0]
    
    @staticmethod
    def _extract_simple_features(image):
        """
        Extract simple feature maps using edge detection and texture analysis.
        This is used when CNN feature extractor is not available.
        
        Args:
            image: Input image (normalized 0-1)
            
        Returns:
            Feature maps array (simulated)
        """
        # Convert to uint8
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
        
        # Ensure 3 channels
        if len(img_uint8.shape) == 2:
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        elif img_uint8.shape[-1] == 4:
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2RGB)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Create multiple feature maps using different methods
        feature_maps_list = []
        
        # 1. Sobel edge detection - X gradient
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)
        sobelx = (sobelx / (sobelx.max() + 1e-8) * 255).astype(np.uint8)
        feature_maps_list.append(sobelx)
        
        # 2. Sobel edge detection - Y gradient
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobely = np.abs(sobely)
        sobely = (sobely / (sobely.max() + 1e-8) * 255).astype(np.uint8)
        feature_maps_list.append(sobely)
        
        # 3. Laplacian edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        laplacian = (laplacian / (laplacian.max() + 1e-8) * 255).astype(np.uint8)
        feature_maps_list.append(laplacian)
        
        # 4. Canny edge detection
        canny = cv2.Canny(gray, 100, 200)
        feature_maps_list.append(canny)
        
        # 5. CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        feature_maps_list.append(clahe_img)
        
        # 6. Gaussian blur (texture simulation)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        feature_maps_list.append(blur)
        
        # 7. Median blur
        median = cv2.medianBlur(gray, 5)
        feature_maps_list.append(median)
        
        # 8. Bilateral filter (edge-preserving smoothing)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        feature_maps_list.append(bilateral)
        
        # Stack all feature maps and convert to 3D (height, width, channels)
        feature_maps = np.stack(feature_maps_list[:8], axis=-1)
        
        # Resize to 224x224 to match expected CNN output size
        feature_maps = cv2.resize(feature_maps, (224, 224))
        
        return feature_maps
    
    @staticmethod
    def visualize_feature_maps(feature_maps, title="Feature Maps", num_maps=12):
        """
        Visualize feature maps as a grid.
        
        Args:
            feature_maps: Feature maps array (height, width, channels)
            title: Title for the visualization
            num_maps: Number of maps to display
            
        Returns:
            base64 encoded image
        """
        num_channels = feature_maps.shape[-1]
        num_maps = min(num_maps, num_channels)
        
        # Create grid
        cols = 4
        rows = int(np.ceil(num_maps / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
        axes = axes.flatten() if num_maps > 1 else [axes]
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        for idx in range(num_maps):
            # Normalize the feature map
            feature_map = feature_maps[:, :, idx]
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            # Display
            axes[idx].imshow(feature_map, cmap='viridis')
            axes[idx].set_title(f'Filter {idx + 1}', fontsize=10, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide extra subplots
        for idx in range(num_maps, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return FeatureVisualization.plot_to_base64(fig)
    
    @staticmethod
    def visualize_histogram(image, title="Image Histogram", bins=256):
        """
        Create histogram visualization for grayscale image.
        
        Args:
            image: Image array
            title: Title for the visualization
            bins: Number of bins for histogram
            
        Returns:
            base64 encoded image
        """
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist = hist.flatten()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        # Plot histogram
        colors = ['#667eea', '#764ba2']
        ax1.bar(range(len(hist)), hist, color=colors[0], alpha=0.7, edgecolor=colors[1])
        ax1.set_title('Pixel Intensity Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Pixel Intensity (0-255)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative histogram
        cumsum_hist = np.cumsum(hist)
        cumsum_hist = (cumsum_hist / cumsum_hist[-1]) * 100
        
        ax2.plot(cumsum_hist, linewidth=2.5, color=colors[1])
        ax2.fill_between(range(len(cumsum_hist)), cumsum_hist, alpha=0.3, color=colors[0])
        ax2.set_title('Cumulative Histogram', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Pixel Intensity (0-255)', fontsize=10)
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        return FeatureVisualization.plot_to_base64(fig)
    
    @staticmethod
    def visualize_histogram_rgb(image, title="RGB Histogram"):
        """
        Create RGB histogram visualization for color image.
        
        Args:
            image: RGB image array
            title: Title for the visualization
            
        Returns:
            base64 encoded image
        """
        # Convert to uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Make sure it's RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Calculate histograms
        colors = ('red', 'green', 'blue')
        color_values = ((1, 0, 0), (0, 0.7, 0), (0, 0.4, 1))
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        for i, (color_name, color_val) in enumerate(zip(colors, color_values)):
            hist = cv2.calcHist([image_bgr], [i], None, [256], [0, 256]).flatten()
            ax.plot(hist, color=color_val, linewidth=2, label=color_name.capitalize())
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Pixel Intensity (0-255)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return FeatureVisualization.plot_to_base64(fig)
    
    @staticmethod
    def visualize_image_comparison(original, processed, title="Image Comparison"):
        """
        Visualize original and processed images side by side with PSNR metrics.
        
        Args:
            original: Original image array
            processed: Processed image array
            title: Title for the visualization
            
        Returns:
            base64 encoded image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Convert to uint8
        orig_display = (original * 255).astype(np.uint8) if original.max() <= 1.0 else original.astype(np.uint8)
        proc_display = (processed * 255).astype(np.uint8) if processed.max() <= 1.0 else processed.astype(np.uint8)
        
        # Calculate PSNR between original and processed images
        psnr_value = ImageMetrics.calculate_psnr(orig_display, proc_display)
        
        ax1.imshow(orig_display, cmap='gray' if len(orig_display.shape) == 2 else None)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(proc_display, cmap='gray' if len(proc_display.shape) == 2 else None)
        ax2.set_title(f'Processed Image (CLAHE)\nPSNR: {psnr_value} dB', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return FeatureVisualization.plot_to_base64(fig)
    
    @staticmethod
    def generate_histogram_statistics(image):
        """
        Generate detailed histogram statistics.
        
        Args:
            image: Image array
            
        Returns:
            dict: Statistics about the histogram
        """
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Flatten
        flat_image = image.flatten()
        
        # Calculate statistics
        stats = {
            'mean': round(float(np.mean(flat_image)), 2),
            'median': round(float(np.median(flat_image)), 2),
            'std': round(float(np.std(flat_image)), 2),
            'min': int(np.min(flat_image)),
            'max': int(np.max(flat_image)),
            'mode': int(np.bincount(flat_image).argmax()),
            'q1': round(float(np.percentile(flat_image, 25)), 2),
            'q3': round(float(np.percentile(flat_image, 75)), 2),
        }
        
        return stats
