from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)

def test_health_endpoints():
    r1 = client.get("/health")
    assert r1.status_code == 200
    assert r1.json() == {"status": "ok"}
    
    r2 = client.get("/api/v1/health")
    assert r2.status_code == 200
    assert r2.json() == {"status": "ok"}

def test_fingerprint_generate_api(tmp_path):
    img_path = tmp_path / "test_fp.png"
    img = Image.new("RGB", (50, 50), color="green")
    img.save(str(img_path))
    
    response = client.post(
        "/api/v1/fingerprint/generate",
        json={
            "media_id": "test_med_01",
            "file_path": str(img_path)
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["media_id"] == "test_med_01"
    assert "fingerprints" in data
    assert "sha256" in data["fingerprints"]
    assert "phash" in data["fingerprints"]
    assert "embedding" in data["fingerprints"]

def test_similarity_search_api():
    response = client.post(
        "/api/v1/similarity/search",
        json={
            "media_id": "test_med_02",
            "file_path": "dummy.jpg"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["media_id"] == "test_med_02"
    assert "matches" in data
    assert len(data["matches"]) > 0

def test_source_provenance_evidence_report_flow(tmp_path):
    img_path = tmp_path / "test_flow.png"
    img = Image.new("RGB", (50, 50), color="yellow")
    img.save(str(img_path))
    
    analysis_id = "test_analysis_999"
    
    # 1. Trace source
    trace_res = client.post(
        "/api/v1/source/trace",
        json={
            "media_id": "flow_media_01",
            "file_path": str(img_path),
            "analysis_id": analysis_id
        }
    )
    assert trace_res.status_code == 200
    assert "earliest_known_occurrence" in trace_res.json()
    
    # 2. Get source
    get_source_res = client.get(f"/api/v1/source/{analysis_id}")
    assert get_source_res.status_code == 200
    assert "earliest_known_occurrence" in get_source_res.json()
    
    # 3. Get provenance
    get_prov_res = client.get(f"/api/v1/provenance/{analysis_id}")
    assert get_prov_res.status_code == 200
    prov_data = get_prov_res.json()
    assert "nodes" in prov_data
    assert "edges" in prov_data
    
    # 4. Get evidence
    get_ev_res = client.get(f"/api/v1/evidence/{analysis_id}")
    assert get_ev_res.status_code == 200
    ev_data = get_ev_res.json()
    assert "case_information" in ev_data
    assert "fingerprint_evidence" in ev_data
    
    # 5. Get report
    get_rep_res = client.get(f"/api/v1/report/{analysis_id}")
    assert get_rep_res.status_code == 200
    rep_data = get_rep_res.json()
    assert "json" in rep_data
    assert "markdown" in rep_data
    assert "# AIMD FORENSIC ANALYSIS REPORT" in rep_data["markdown"]
