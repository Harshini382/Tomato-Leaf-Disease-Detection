import time
import numpy as np
import cv2

# Prefer skimage when available, but allow fallback for environments without it
try:
    from skimage import feature, color, exposure
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.measure import shannon_entropy
    SKIMAGE_AVAILABLE = True
except Exception:
    feature = None
    SKIMAGE_AVAILABLE = False
    def shannon_entropy(img):
        # simple entropy fallback using histogram
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        if hist.size == 0:
            return 0.0
        return float(-np.sum(hist * np.log2(hist)))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def psnr(img1, img2):
    """Compute PSNR between two images. Images must be same shape."""
    try:
        return sk_psnr(img1, img2, data_range=img2.max() - img2.min())
    except Exception:
        mse = np.mean((img1.astype('float32') - img2.astype('float32')) ** 2)
        if mse == 0:
            return float('inf')
        PIXEL_MAX = 255.0
        return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def image_histogram(image, bins=256):
    """Return histogram (counts, bin_edges) for a grayscale or single-channel image."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    counts, bins = np.histogram(image.ravel(), bins=bins, range=(0, 256))
    return counts, bins


def image_entropy(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(shannon_entropy(image))


def to_grayscale(image):
    """Convert BGR or RGB image to grayscale (uint8)."""
    if image.ndim == 2:
        return image.copy().astype('uint8')
    # assume image in BGR or RGB; convert using cv2 which expects BGR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def clahe_enhance(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to a grayscale uint8 image and return enhanced image."""
    if gray_image.dtype != np.uint8:
        gray = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    else:
        gray = gray_image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def compute_basic_stats(image):
    """Return dict: mean, std, entropy, histogram counts."""
    if image.ndim == 3:
        # compute per-channel mean/std and aggregate
        means = list(map(float, cv2.mean(image)[:3]))
        stds = [float(np.std(image[:, :, i])) for i in range(3)]
        ent = None
        hist = None
        return {
            'mean': means,
            'std': stds,
            'entropy': ent,
            'histogram': hist,
        }
    else:
        return {
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'entropy': float(image_entropy(image)),
            'histogram': image_histogram(image)[0],
        }


def glcm_features(gray_image, distances=(1,), angles=(0,), levels=256, symmetric=True, normed=True):
    """Compute GLCM and common properties (contrast, correlation, energy, homogeneity).

    Returns a dict with each property averaged across distances and angles.
    """
    if gray_image.ndim == 3:
        gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_image
    # ensure integer levels
    image = (gray.copy() if gray.dtype == np.uint8 else cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    out = {p: 0.0 for p in props}

    if SKIMAGE_AVAILABLE and feature is not None:
        try:
            glcm = feature.graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
            for p in props:
                vals = feature.graycoprops(glcm, p)
                out[p] = float(np.mean(vals))
            return out
        except Exception:
            # fall through to fallback calculations
            pass

    # Fallback: compute simple proxy statistics when skimage GLCM is not available
    # contrast ~ variance, correlation ~ 0, energy ~ sum(hist^2), homogeneity ~ 1/(1+variance)
    img = image.astype('float32')
    variance = float(np.var(img))
    hist = np.histogram(img.ravel(), bins=levels, range=(0, 256))[0].astype('float32')
    hist = hist / (hist.sum() + 1e-8)
    energy = float(np.sum(hist ** 2))
    homogeneity = float(1.0 / (1.0 + variance)) if variance >= 0 else 0.0
    out['contrast'] = variance
    out['correlation'] = 0.0
    out['energy'] = energy
    out['homogeneity'] = homogeneity
    return out


def extract_feature_vector(rgb_image):
    """Create a combined feature vector from the pipeline.

    Steps:
    - original stats (RGB)
    - grayscale stats
    - CLAHE(stats)
    - GLCM features on CLAHE image

    Returns (feature_vector, feature_names, meta_dict)
    """
    meta = {}
    # original stats
    orig_stats = compute_basic_stats(rgb_image)
    meta['original_stats'] = orig_stats

    gray = to_grayscale(rgb_image)
    gray_stats = compute_basic_stats(gray)
    meta['grayscale_stats'] = gray_stats

    clahe = clahe_enhance(gray)
    clahe_stats = compute_basic_stats(clahe)
    meta['clahe_stats'] = clahe_stats

    # psnr between original grayscale and clahe-enhanced
    try:
        meta['psnr_gray_clahe'] = float(psnr(gray, clahe))
    except Exception:
        meta['psnr_gray_clahe'] = None

    # glcm features
    glcm_feats = glcm_features(clahe, distances=(1,), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4))
    meta['glcm'] = glcm_feats

    # Flatten to vector
    features = []
    names = []
    # grayscale mean/std
    features.append(gray_stats['mean'])
    names.append('gray_mean')
    features.append(gray_stats['std'])
    names.append('gray_std')
    # clahe mean/std
    features.append(clahe_stats['mean'])
    names.append('clahe_mean')
    features.append(clahe_stats['std'])
    names.append('clahe_std')
    # psnr
    features.append(meta['psnr_gray_clahe'] if meta['psnr_gray_clahe'] is not None else 0.0)
    names.append('psnr_gray_clahe')
    # glcm
    for k, v in glcm_feats.items():
        features.append(v)
        names.append(f'glcm_{k}')

    return np.array(features, dtype='float32'), names, meta


def train_classifiers(X, y, test_size=0.2, random_state=42, cv=5):
    """Train RandomForest, XGBoost, LightGBM (if available) and evaluate.

    Returns dict of results for each model including fitted model and metrics.
    """
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # helper to evaluate a fitted model
    def evaluate(name, model):
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        y_pred = model.predict(X_test)
        y_prob = None
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            y_prob = None
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'train_time_seconds': float(train_time),
        }
        # cross-val f1
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            metrics['cv_f1_mean'] = float(np.mean(cv_scores))
            metrics['cv_f1_std'] = float(np.std(cv_scores))
        except Exception:
            metrics['cv_f1_mean'] = None
            metrics['cv_f1_std'] = None
        return {'model': model, 'metrics': metrics}

    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    results['RandomForest'] = evaluate('RandomForest', rf)

    # XGBoost
    if XGBClassifier is not None:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        results['XGBoost'] = evaluate('XGBoost', xgb)

    # LightGBM
    if LGBMClassifier is not None:
        lgbm = LGBMClassifier()
        results['LightGBM'] = evaluate('LightGBM', lgbm)

    return results


def select_best_classifier(results, key='cv_f1_mean'):
    """Select best classifier by a given metric key (prefers higher and lower std)."""
    best_name = None
    best_score = -np.inf
    for name, info in results.items():
        m = info.get('metrics', {})
        score = m.get(key)
        if score is None:
            # fallback to f1
            score = m.get('f1', -np.inf)
        if score is not None and score > best_score:
            best_score = score
            best_name = name
    return best_name, results.get(best_name)


def predict_with_confidence(model, X):
    """Return predictions and confidence scores (0..1).

    If model supports `predict_proba` use the max probability; otherwise use decision_function and sigmoid.
    """
    preds = model.predict(X)
    confs = None
    try:
        proba = model.predict_proba(X)
        confs = np.max(proba, axis=1)
    except Exception:
        try:
            df = model.decision_function(X)
            # if multiclass, take max of softmax-like sigmoid across outputs
            if df.ndim == 1:
                confs = 1.0 / (1.0 + np.exp(-df))
            else:
                # apply softmax
                ex = np.exp(df - np.max(df, axis=1, keepdims=True))
                confs = np.max(ex / np.sum(ex, axis=1, keepdims=True), axis=1)
        except Exception:
            confs = np.ones(len(preds)) * 0.5
    return preds, confs


__all__ = [
    'psnr', 'image_histogram', 'image_entropy', 'to_grayscale', 'clahe_enhance', 'compute_basic_stats',
    'glcm_features', 'extract_feature_vector', 'train_classifiers', 'select_best_classifier', 'predict_with_confidence'
]
