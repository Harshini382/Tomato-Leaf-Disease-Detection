"""Smoke test for methods.method1 using a synthetic image.

Creates a simple synthetic RGB image, encodes to bytes, calls method1.run_method1,
and prints key outputs to stdout for quick verification.
"""
import io
import sys
import os
import numpy as np
import cv2
from PIL import Image

# Ensure project root is on sys.path so package imports work when running from tools/
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from methods import method1


def make_synthetic_leaf_image(width=256, height=256):
    # simple green circular blob on light background
    img = np.full((height, width, 3), 220, dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.circle(img, center, min(width, height)//3, (34,139,34), -1)  # forest green
    # add some darker spots to mimic disease
    cv2.circle(img, (center[0]+30, center[1]-10), 20, (80,40,40), -1)
    cv2.circle(img, (center[0]-25, center[1]+20), 15, (80,40,40), -1)
    return img


def img_to_bytes(img_array):
    pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return buf.getvalue()


def run():
    img = make_synthetic_leaf_image()
    b = img_to_bytes(img)
    try:
        out = method1.run_method1(b)
    except Exception as e:
        import traceback
        print('Exception when calling method1.run_method1:', e)
        traceback.print_exc()
        return

    print('--- method1.run_method1 output keys ---')
    for k in out.keys():
        print('-', k)

    res = out.get('results')
    if res:
        print('\nPrediction:')
        print(' predicted_class:', res.get('predicted_class'))
        print(' confidence:', res.get('confidence'))
        probs = res.get('probabilities')
        if probs:
            print(' probabilities keys:', list(probs.keys())[:5])

    fv = out.get('feature_vector')
    names = out.get('feature_names')
    if fv is not None and names is not None:
        print('\nFeature vector:')
        for n, v in zip(names, fv.tolist() if hasattr(fv, 'tolist') else fv):
            print(f' {n}: {v:.4f}')


if __name__ == '__main__':
    run()
