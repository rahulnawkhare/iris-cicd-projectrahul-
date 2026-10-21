
from app import app


def test_home_endpoint():
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert response.json["message"] == (
        "Iris ML REST API is running"
    )


def test_predict_endpoint():
    client = app.test_client()

    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 200
    assert response.json["prediction"] == "setosa"


def test_invalid_input():
    client = app.test_client()

    response = client.post(
        "/predict",
        json={"sepal_length": "invalid"}
    )

    assert response.status_code == 400
    assert "error" in response.json
