# #!/usr/bin/env python3
# """
# Comprehensive test script for Tomato Disease Detection Web Application
# """

# import os
# import requests
# import tempfile
# import io
# from PIL import Image
# import numpy as np
# import cv2
# from flask import Flask
# import threading
# import time
# import json

# # Test configuration
# TEST_HOST = 'http://localhost:5000'
# UPLOAD_TIMEOUT = 30

# def create_test_image(disease_class='healthy', size=(256, 256)):
#     """Create a synthetic test image for testing."""
#     # Create a simple colored image
#     if disease_class == 'healthy':
#         # Green healthy leaf
#         image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
#         image[:, :] = [50, 150, 50]  # Green
#     else:
#         # Diseased appearance - brown/yellow spots
#         image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
#         image[:, :] = [100, 120, 60]  # Base green-yellow
#         # Add some brown spots
#         for _ in range(10):
#             x = np.random.randint(0, size[0]-20)
#             y = np.random.randint(0, size[1]-20)
#             image[x:x+20, y:y+20] = [60, 40, 20]  # Brown spots

#     # Convert to PIL Image
#     pil_image = Image.fromarray(image)
#     return pil_image

# def test_home_page():
#     """Test the home page loads correctly."""
#     print("🧪 Testing home page...")

#     try:
#         response = requests.get(f"{TEST_HOST}/", timeout=10)
#         if response.status_code == 200:
#             print("✅ Home page loads successfully")
#             # Check for key elements
#             content = response.text.lower()
#             checks = [
#                 'tomato leaf disease detection' in content,
#                 'upload image' in content,
#                 'file' in content,
#                 'submit' in content
#             ]
#             if all(checks):
#                 print("✅ Home page contains expected elements")
#                 return True
#             else:
#                 print("❌ Home page missing expected elements")
#                 return False
#         else:
#             print(f"❌ Home page returned status {response.status_code}")
#             return False
#     except Exception as e:
#         print(f"❌ Home page test failed: {e}")
#         return False

# def test_file_upload_valid():
#     """Test uploading a valid image file."""
#     print("🧪 Testing valid file upload...")

#     try:
#         # Create a test image
#         test_image = create_test_image('healthy')

#         # Save to temporary file
#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
#             test_image.save(tmp_file, format='JPEG')
#             tmp_file_path = tmp_file.name

#         # Upload the file
#         with open(tmp_file_path, 'rb') as f:
#             files = {'file': ('test_healthy.jpg', f, 'image/jpeg')}
#             response = requests.post(f"{TEST_HOST}/predict",
#                                    files=files,
#                                    timeout=UPLOAD_TIMEOUT)

#         # Clean up
#         os.unlink(tmp_file_path)

#         if response.status_code == 200:
#             print("✅ Valid file upload successful")
#             content = response.text.lower()
#             # Check for result page elements
#             checks = [
#                 'analysis results' in content,
#                 'processing stages' in content,
#                 'prediction' in content,
#                 'confidence' in content
#             ]
#             if all(checks):
#                 print("✅ Result page contains expected elements")
#                 return True
#             else:
#                 print("❌ Result page missing expected elements")
#                 return False
#         else:
#             print(f"❌ File upload returned status {response.status_code}")
#             print(f"Response: {response.text[:500]}")
#             return False

#     except Exception as e:
#         print(f"❌ Valid file upload test failed: {e}")
#         return False

# def test_file_upload_invalid_type():
#     """Test uploading an invalid file type."""
#     print("🧪 Testing invalid file type upload...")

#     try:
#         # Create a text file
#         with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
#             tmp_file.write(b"This is not an image file")
#             tmp_file_path = tmp_file.name

#         # Try to upload
#         with open(tmp_file_path, 'rb') as f:
#             files = {'file': ('test.txt', f, 'text/plain')}
#             response = requests.post(f"{TEST_HOST}/predict",
#                                    files=files,
#                                    timeout=UPLOAD_TIMEOUT)

#         # Clean up
#         os.unlink(tmp_file_path)

#         # Should redirect back to home with flash message
#         if response.status_code == 200 and 'upload image' in response.text.lower():
#             print("✅ Invalid file type properly rejected")
#             return True
#         else:
#             print(f"❌ Invalid file type test failed - status: {response.status_code}")
#             return False

#     except Exception as e:
#         print(f"❌ Invalid file type test failed: {e}")
#         return False

# def test_file_upload_no_file():
#     """Test submitting form without a file."""
#     print("🧪 Testing upload without file...")

#     try:
#         response = requests.post(f"{TEST_HOST}/predict", timeout=UPLOAD_TIMEOUT)

#         if response.status_code == 200 and 'upload image' in response.text.lower():
#             print("✅ No file upload properly handled")
#             return True
#         else:
#             print(f"❌ No file upload test failed - status: {response.status_code}")
#             return False

#     except Exception as e:
#         print(f"❌ No file upload test failed: {e}")
#         return False

# def test_file_upload_large_file():
#     """Test uploading a file that's too large."""
#     print("🧪 Testing large file upload...")

#     try:
#         # Create a large image (should exceed 16MB limit)
#         large_image = create_test_image('healthy', size=(4000, 4000))

#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
#             large_image.save(tmp_file, format='JPEG', quality=100)
#             tmp_file_path = tmp_file.name

#         # Check file size
#         file_size = os.path.getsize(tmp_file_path)
#         print(f"   Test file size: {file_size / (1024*1024):.1f} MB")

#         with open(tmp_file_path, 'rb') as f:
#             files = {'file': ('large_test.jpg', f, 'image/jpeg')}
#             response = requests.post(f"{TEST_HOST}/predict",
#                                    files=files,
#                                    timeout=UPLOAD_TIMEOUT)

#         # Clean up
#         os.unlink(tmp_file_path)

#         # Should return 413 (Request Entity Too Large) or redirect with error
#         if response.status_code in [413, 200] and ('large' in response.text.lower() or 'upload image' in response.text.lower()):
#             print("✅ Large file properly rejected")
#             return True
#         else:
#             print(f"❌ Large file test failed - status: {response.status_code}")
#             return False

#     except Exception as e:
#         print(f"❌ Large file test failed: {e}")
#         return False

# def test_prediction_content():
#     """Test that prediction results contain expected content."""
#     print("🧪 Testing prediction content...")

#     try:
#         # Upload a test image
#         test_image = create_test_image('healthy')

#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
#             test_image.save(tmp_file, format='JPEG')
#             tmp_file_path = tmp_file.name

#         with open(tmp_file_path, 'rb') as f:
#             files = {'file': ('test_content.jpg', f, 'image/jpeg')}
#             response = requests.post(f"{TEST_HOST}/predict",
#                                    files=files,
#                                    timeout=UPLOAD_TIMEOUT)

#         # Clean up
#         os.unlink(tmp_file_path)

#         if response.status_code == 200:
#             content = response.text

#             # Check for required elements
#             required_elements = [
#                 'predicted_class',
#                 'confidence',
#                 'original_image',
#                 'grayscale_image',
#                 'clahe_image',
#                 'processing_grid',
#                 'probability-item'
#             ]

#             missing_elements = []
#             for element in required_elements:
#                 if element not in content:
#                     missing_elements.append(element)

#             if not missing_elements:
#                 print("✅ Prediction content contains all required elements")
#                 return True
#             else:
#                 print(f"❌ Missing elements: {missing_elements}")
#                 return False
#         else:
#             print(f"❌ Prediction content test failed - status: {response.status_code}")
#             return False

#     except Exception as e:
#         print(f"❌ Prediction content test failed: {e}")
#         return False

# def test_responsive_design():
#     """Test responsive design elements."""
#     print("🧪 Testing responsive design...")

#     try:
#         response = requests.get(f"{TEST_HOST}/", timeout=10)

#         if response.status_code == 200:
#             content = response.text

#             # Check for responsive meta tag
#             if 'viewport' in content and 'width=device-width' in content:
#                 print("✅ Responsive viewport meta tag present")
#             else:
#                 print("❌ Missing responsive viewport meta tag")
#                 return False

#             # Check for CSS media queries (basic check)
#             if '@media' in content:
#                 print("✅ CSS media queries present")
#             else:
#                 print("❌ Missing CSS media queries")
#                 return False

#             return True
#         else:
#             print(f"❌ Responsive design test failed - status: {response.status_code}")
#             return False

#     except Exception as e:
#         print(f"❌ Responsive design test failed: {e}")
#         return False

# def wait_for_server(timeout=30):
#     """Wait for the Flask server to be ready."""
#     print("⏳ Waiting for Flask server to start...")
#     start_time = time.time()

#     while time.time() - start_time < timeout:
#         try:
#             response = requests.get(f"{TEST_HOST}/", timeout=5)
#             if response.status_code == 200:
#                 print("✅ Flask server is ready")
#                 return True
#         except:
#             pass
#         time.sleep(1)

#     print("❌ Flask server failed to start within timeout")
#     return False

# def main():
#     """Run all web application tests."""
#     print("🌐 Tomato Disease Detection Web App - Comprehensive Test Suite")
#     print("=" * 60)

#     # Wait for server to be ready
#     if not wait_for_server():
#         print("❌ Cannot proceed with tests - server not ready")
#         return

#     # Define test cases
#     test_cases = [
#         ("Home Page Load", test_home_page),
#         ("Valid File Upload", test_file_upload_valid),
#         ("Invalid File Type", test_file_upload_invalid_type),
#         ("No File Upload", test_file_upload_no_file),
#         ("Large File Upload", test_file_upload_large_file),
#         ("Prediction Content", test_prediction_content),
#         ("Responsive Design", test_responsive_design)
#     ]

#     results = []
#     for test_name, test_func in test_cases:
#         print(f"\n🔬 Running {test_name}...")
#         try:
#             result = test_func()
#             results.append(result)
#         except Exception as e:
#             print(f"❌ Test crashed: {e}")
#             results.append(False)

#     # Summary
#     print("\n" + "=" * 60)
#     print("📊 Web Application Test Results Summary:")
#     passed = 0
#     for i, (test_name, _) in enumerate(test_cases):
#         status = "✅ PASS" if results[i] else "❌ FAIL"
#         print(f"  {test_name}: {status}")
#         if results[i]:
#             passed += 1

#     total = len(test_cases)
#     print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

#     if passed == total:
#         print("🎉 All web application tests passed!")
#         print("🚀 The Tomato Disease Detection System is fully functional and ready for production use.")
#     elif passed >= total * 0.8:  # 80% pass rate
#         print("⚠️ Most tests passed. System is functional but has some issues to address.")
#     else:
#         print("❌ Multiple tests failed. System requires fixes before production use.")

# if __name__ == "__main__":
#     main()
