#!/usr/bin/env python3
"""Quick test runner for methods.method2.run_method2

Finds a sample image in the repository (train/ or val/) and runs the Method 2
pipeline, printing a small JSON summary to stdout.
"""
import os
import sys
import json

import sys
import os
# Ensure project root is on sys.path so imports like `methods` resolve when
# running the script from tests/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from methods import method2


def find_sample():
    search_dirs = ['tomato/train', 'tomato/val', 'uploads']
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(root, f)
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_sample()
    if not path:
        print(json.dumps({'error': 'no sample image found in tomato/train, tomato/val or uploads'}))
        sys.exit(2)

    with open(path, 'rb') as fh:
        file_bytes = fh.read()

    out = method2.run_method2(file_bytes)

    results = out.get('results', {})
    glcm_vec = out.get('glcm_vector')
    deep_vec = out.get('deep_vector')
    combined = out.get('combined_vector')

    summary = {
        'sample_path': path,
        'predicted_class': results.get('predicted_class'),
        'confidence': float(results.get('confidence', 0.0)) if results else None,
        'classifier_used': out.get('classifier_used'),
        'glcm_vector_len': int(len(glcm_vec)) if hasattr(glcm_vec, '__len__') else None,
        'deep_vector_len': int(len(deep_vec)) if hasattr(deep_vec, '__len__') else None,
        'combined_vector_len': int(len(combined)) if hasattr(combined, '__len__') else None,
    }

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
