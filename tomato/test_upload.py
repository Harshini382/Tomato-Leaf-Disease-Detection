import requests
import io
from PIL import Image
import numpy as np

# Create a test image
image = np.zeros((256, 256, 3), dtype=np.uint8)
image[:, :] = [50, 150, 50]  # Green color
pil_image = Image.fromarray(image)

# Save to bytes
buf = io.BytesIO()
pil_image.save(buf, format='PNG')
buf.seek(0)

# Create a session to handle cookies (for authentication)
session = requests.Session()

def check_authentication():
    """Check if user is authenticated by accessing dashboard"""
    response = session.get('http://127.0.0.1:5000/dashboard')
    return 'dashboard' in response.url or 'Upload' in response.text

def test_upload():
    """Test image upload after authentication"""
    print("Testing authenticated upload...")

    # First check if authenticated
    if not check_authentication():
        print("❌ User not authenticated")
        return False

    # Reset buffer position
    buf.seek(0)

    files = {'file': ('test.png', buf, 'image/png')}
    response = session.post('http://127.0.0.1:5000/predict', files=files)
    print(f"Upload response: {response.status_code}")
    print(f"Response URL: {response.url}")

    if response.status_code == 200:
        # Check if we're on the results page
        if 'predicted_class' in response.text or 'Disease Detection Results' in response.text:
            print("✅ Successfully reached results page!")
            return True
        else:
            print("❌ Upload succeeded but not on results page")
            print("Response content preview:", response.text[:500])
            return False
    else:
        print(f"❌ Upload failed with status {response.status_code}")
        return False

# Test the flow
print("=== Testing Tomato Disease Detection App ===\n")

# Test upload directly without checking authentication
print("Testing direct upload...")
buf.seek(0)

files = {'file': ('test.png', buf, 'image/png')}
response = session.post('http://127.0.0.1:5000/predict', files=files)
print(f"Upload response: {response.status_code}")
print(f"Response URL: {response.url}")

if response.status_code == 200:
    # Check if we're on the results page
    if 'predicted_class' in response.text or 'Disease Detection Results' in response.text:
        print("✅ Successfully reached results page!")
        print("Response contains prediction data")
    else:
        print("❌ Response does not contain prediction data")
        print("Response content preview:", response.text[:1000])
else:
    print(f"❌ Upload failed with status {response.status_code}")

print("\nTest completed.")