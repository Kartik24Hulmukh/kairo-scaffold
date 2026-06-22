import os
import json
import pytest
from fastapi.testclient import TestClient
import sys

# Adjust path to import from kernel.sidecar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kernel.sidecar.app import app

client = TestClient(app)

def test_ground_truth_fixtures():
    # Find root dir
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    ground_truth_path = os.path.join(root_dir, 'fixtures', 'golden', 'ground_truth.json')
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for entry in data['fixtures']:
        file_name = entry['file']
        expected_chunks = entry['expected_chunks']
        expected_pages = entry['expected_pages']
        
        file_path = os.path.join(root_dir, 'fixtures', 'golden', file_name)
        
        res = client.post("/index", json={"path": file_path})
        assert res.status_code == 200, f"Failed to index {file_name}: {res.text}"
        
        res_data = res.json()
        actual_chunks = res_data['chunks']
        actual_pages = res_data['pages']
        
        # Verify pages matches
        assert actual_pages == expected_pages, f"{file_name}: expected {expected_pages} pages, got {actual_pages}"
        
        # Verify chunks is within 2% tolerance
        if expected_chunks == 0:
            assert actual_chunks == 0, f"{file_name}: expected 0 chunks, got {actual_chunks}"
        else:
            diff = abs(actual_chunks - expected_chunks) / expected_chunks
            assert diff <= 0.02, f"{file_name}: expected {expected_chunks} chunks, got {actual_chunks} (diff {diff:.2%})"
