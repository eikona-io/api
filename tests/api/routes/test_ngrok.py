import pytest
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_ngrok_url_with_retry(max_retries=5, delay=1):
    """Get ngrok URL with retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"http://{os.getenv('NGROK_HOST')}:4040/api/tunnels"
            )
            return response.json()["tunnels"][0]["public_url"]
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                return "http://localhost:8000"  # fallback url
            print(
                f"Failed to get ngrok URL (attempt {attempt + 1}/{max_retries}). Retrying..."
            )
            import time

            time.sleep(delay)


ngrok_url = get_ngrok_url_with_retry()

# This should override the CURRENT_API_URL in the .env file to be the dynamic ngrok url
os.environ["CURRENT_API_URL"] = ngrok_url


@pytest.mark.asyncio
async def test_ngrok_setup(app):
    """Test to verify ngrok setup and accessibility"""
    print("\nhi there")
    print(f"ngrok_url: {ngrok_url}")
    assert ngrok_url is not None

    # Try to access the URL
    try:
        response = requests.get(ngrok_url)
        assert response.status_code == 200, (
            f"Expected status code 200, but got {response.status_code}"
        )
        print(
            f"Successfully connected to ngrok URL with status code {response.status_code}"
        )
    except requests.RequestException as e:
        pytest.fail(f"Failed to connect to ngrok URL: {str(e)}")
