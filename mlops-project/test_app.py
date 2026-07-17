from app import app

client = app.test_client()

def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert b"Flask API is running" in response.data