#!/usr/bin/env python3
"""
Generate OpenAPI JSON spec from the FastAPI app.
"""
import json
import sys
import os
import traceback

# Add the backend directory to the path so we can import the app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

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