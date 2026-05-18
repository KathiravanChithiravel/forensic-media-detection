from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def test_jira_ticket_101():
    print("Testing JIRA TICKET: DETECT-101 (Admin Login Restriction)")
    driver = webdriver.Chrome()
    try:
        # Step 1: Going to the Admin Login page
        driver.get("http://127.0.0.1:5000/admin/login")
        driver.maximize_window()
        wait = WebDriverWait(driver, 5)
        
        # Step 2: Fill in normal user credentials
        print("Entering normal credentials (kathir / 1234) into Admin portal...")
        wait.until(EC.element_to_be_clickable((By.NAME, "username"))).send_keys("kathir")
        wait.until(EC.element_to_be_clickable((By.NAME, "password"))).send_keys("1234")
        
        # Step 3: Click the Login Button
        login_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']")))
        driver.execute_script("arguments[0].click();", login_btn)
        time.sleep(2) # Waiting for page to reload
        
        # Step 4: Verify the output (Expected: Red Error Message)
        print("Checking if System blocked the user with the right message...")
        page_source = driver.page_source
        expected_error = "Invalid administrative credentials. Access Denied."
        
        if expected_error in page_source:
            print("✅ JIRA TICKET DETECT-101 PASSED: QA Verification Complete! The system correctly blocked normal users.")
        else:
            print("❌ JIRA TICKET DETECT-101 FAILED (BUG): System did not show the expected error message or allowed the user in!")
            
    finally:
        time.sleep(3)
        driver.quit()

if __name__ == "__main__":
    test_jira_ticket_101()
