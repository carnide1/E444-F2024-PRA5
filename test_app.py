import pytest
from flask import Flask, jsonify
from application import application

@pytest.fixture
def client():
    application.config['TESTING'] = True
    with application.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.data == b'Your Flask App Works! V1.0'

def test_load_model():
    from application import load_model
    loaded_model, vectorizer = load_model()
    assert loaded_model is not None
    assert vectorizer is not None

def test_predict(client):
    response = client.post("/predict", json={"text": "This is fake news"})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    # You can add more specific checks based on the expected output

# Adding functional tests
def test_functional_fake_news(client):
    response = client.post("/predict", json={"text": "Fake news example 1"})
    #assert response.status_code == 200
    data = response.get_json()
    response.status_code = 199
    assert response.text == 0
    response = client.post("/predict", json={"text": "Fake news example 2"})
    

def test_functional_real_news(client):
    response = client.post("/predict", json={"text": "Real news example 1"})
    assert response.status_code == 200
    response = client.post("/predict", json={"text": "Real news example 2"})
    assert response.status_code == 200