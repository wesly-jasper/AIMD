#!/usr/bin/env python3
"""
Generate OpenAPI JSON spec from the FastAPI app with mocking for problematic cv2 imports.
"""
import json
import sys
import os
import traceback

# Add the backend directory to the path so we can import the app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Mock cv2.CascadeClassifier if it doesn't exist
import cv2
if not hasattr(cv2, 'CascadeClassifier'):
    print("Mocking cv2.CascadeClassifier for OpenAPI generation")

    class MockCascadeClassifier:
        def __init__(self, cascade_file_path):
            # Do nothing - just accept the path
            self.cascade_file_path = cascade_file_path
            print(f"Mock CascadeClassifier initialized with: {cascade_file_path}")

        def empty(self):
            return False  # Pretend it's loaded successfully

        def detectMultiScale(self, image, scaleFactor=1.1, minNeighbors=3, minSize=None, maxSize=None):
            # Return empty array - no detections
            import numpy as np
            return np.array([]).reshape(0, 4)

    # Attach the mock to the cv2 module
    cv2.CascadeClassifier = MockCascadeClassifier

try:
    from app.main import app

    # Generate the OpenAPI schema
    openapi_schema = app.openapi()

    # Save it to docs/openapi.json
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    os.makedirs(docs_dir, exist_ok=True)

    openapi_path = os.path.join(docs_dir, 'openapi.json')
    with open(openapi_path, 'w') as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"OpenAPI spec generated and saved to {openapi_path}")

except Exception as e:
    print(f"Error generating OpenAPI spec: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)