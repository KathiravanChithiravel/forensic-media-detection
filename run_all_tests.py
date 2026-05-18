import requests
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

# ==========================================
# 1. BACKEND API TESTING (Requests)
# ==========================================
def test_api_home_page():
    print("\n[API Test] Checking Home Page...")
    url = "http://127.0.0.1:5000/"
    try:
        response = requests.get(url)
        assert response.status_code == 200, "Home page failed to load!"
    except requests.exceptions.ConnectionError:
        pytest.fail("Server is off. Run app.py first.")

def test_api_invalid_login():
    print("\n[API Test] Checking Negative Login...")
    url = "http://127.0.0.1:5000/login"
    data = {"username": "test@gmail.com", "password": "wrongpassword"}
    response = requests.post(url, data=data)
    assert response.status_code == 200, "Validation failed!"

# ==========================================
# 2. SECURITY & ACCESS CONTROL TESTING 
# ==========================================
def test_security_admin_access():
    print("\n[Security Test] Checking if normal users are blocked from Admin Dashboard...")
    # Trying to access admin dashboard without logging in or using a normal user account
    url = "http://127.0.0.1:5000/admin/dashboard" 
    # Important: Since we are not sending an admin session cookie, the server shouldn't allow access
    response = requests.get(url, allow_redirects=False)
    # Status code 302 means it correctly blocked access and redirected to login/home
    assert response.status_code == 302 or response.status_code == 301 or response.status_code == 401 or response.status_code == 403, "🚨 SECURITY FLAW: Unauthorized Admin Access Allowed!"
    print("✅ Security Test Passed: Admin path is protected!")

# ==========================================
# 3. LOAD & PERFORMANCE TESTING
# ==========================================
def test_load_performance():
    print("\n[Performance Test] Shooting 10 quick requests to find Server Response Time...")
    url = "http://127.0.0.1:5000/"
    total_time = 0
    
    for i in range(10): # 10 requests at the same time
        start_time = time.time()
        requests.get(url)
        end_time = time.time()
        total_time += (end_time - start_time)
        
    avg_speed = total_time / 10
    print(f"📊 Server Speed: {avg_speed:.4f} seconds per request.")
    assert avg_speed < 1.0, "Performance Fail: API takes more than 1 second to load!"

# ==========================================
# 4. FRONTEND UI TESTING (Selenium)
# ==========================================
def test_ui_login_form():
    print("\n[UI Test] Opening browser for Login Test...")
    driver = webdriver.Chrome()
    try:
        driver.get("http://127.0.0.1:5000/login")
        driver.maximize_window()
        wait = WebDriverWait(driver, 5)
        
        wait.until(EC.element_to_be_clickable((By.NAME, "username"))).send_keys("kathir")
        wait.until(EC.element_to_be_clickable((By.NAME, "password"))).send_keys("1234")
        
        login_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']")))
        driver.execute_script("arguments[0].click();", login_btn)
        time.sleep(2)
    finally:
        driver.quit()

# ==========================================
# 5. END-TO-END AUTOMATION TESTING 
# ==========================================
def test_e2e_media_upload():
    print("\n[E2E Test] Starting Full Media Upload Pipeline...")
    driver = webdriver.Chrome()
    try:
        driver.get("http://127.0.0.1:5000/login")
        driver.maximize_window()
        wait = WebDriverWait(driver, 5)
        
        # Login
        wait.until(EC.element_to_be_clickable((By.NAME, "username"))).send_keys("kathir")
        wait.until(EC.element_to_be_clickable((By.NAME, "password"))).send_keys("1234")
        login_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']")))
        driver.execute_script("arguments[0].click();", login_btn)
        time.sleep(1)
        
        # Go to Upload
        driver.get("http://127.0.0.1:5000/upload")
        
        # Upload absolute file (logo.png)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "static", "images", "logo.png")
        
        file_input = wait.until(EC.presence_of_element_located((By.ID, "fileInput")))
        file_input.send_keys(file_path)
        
        # Submit Scan
        submit_btn = wait.until(EC.presence_of_element_located((By.ID, "submitBtn")))
        driver.execute_script("arguments[0].click();", submit_btn)
        
        time.sleep(5) # Let it analyze for 5 secs before closing
    finally:
        driver.quit()

# Ithu direct ah "python run_all_tests.py" nu azhaithaale motha test um run aaga setup:
if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
