"""
Automated unit test for web_app Flask routes and APIs.
"""
from web_app import app

def test_web_routes():
    print("[Test Web] Creating test client...")
    client = app.test_client()

    print("[Test Web] 1. Testing GET / (HTML5 Page)...")
    res = client.get('/')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"AI OBJECT COUNTER" in res.data, "HTML page missing title"
    print("[Test Web] GET / passed!")

    print("[Test Web] 2. Testing GET /api/classes...")
    res = client.get('/api/classes')
    assert res.status_code == 200
    data = res.get_json()
    assert "popular" in data and "all" in data
    assert len(data["all"]) >= 80
    print(f"[Test Web] GET /api/classes passed! ({len(data['all'])} classes)")

    print("[Test Web] 3. Testing GET /api/stats...")
    res = client.get('/api/stats')
    assert res.status_code == 200
    stats = res.get_json()
    assert "fps" in stats
    assert "mode" in stats
    print("[Test Web] GET /api/stats passed!")

    print("[Test Web] 4. Testing POST /api/settings...")
    res = client.post('/api/settings', json={
        "mode": "line_crossing",
        "target_classes": ["bottle", "cup"],
        "conf": 0.50,
        "line_y": 0.60
    })
    assert res.status_code == 200
    ret = res.get_json()
    assert ret["status"] == "success"
    assert ret["settings"]["mode"] == "line_crossing"
    assert ret["settings"]["conf"] == 0.50
    print("[Test Web] POST /api/settings passed!")

    print("[Test Web] 5. Testing POST /api/reset...")
    res = client.post('/api/reset')
    assert res.status_code == 200
    assert res.get_json()["status"] == "success"
    print("[Test Web] POST /api/reset passed!")

    print("[Test Web] 6. Testing GET /index.html...")
    res = client.get('/index.html')
    assert res.status_code == 200
    assert b"AI OBJECT COUNTER" in res.data
    print("[Test Web] GET /index.html passed!")

    print("[Test Web] 7. Testing POST /api/detect_frame with sample frame...")
    import cv2, numpy as np, base64
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', dummy_img)
    b64_str = base64.b64encode(buf).decode('utf-8')
    res = client.post('/api/detect_frame', json={"image": f"data:image/jpeg;base64,{b64_str}"})
    assert res.status_code == 200
    ret = res.get_json()
    assert ret["status"] == "success"
    assert "detections" in ret
    assert "stats" in ret
    print("[Test Web] POST /api/detect_frame passed!")

    print("\n✅ ALL WEB APP TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_web_routes()
