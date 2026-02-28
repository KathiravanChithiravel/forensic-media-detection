# 🛡️ Multi-Layer AI Forensic Audit Engine

A premium, professional-grade forensic analysis tool designed to detect AI-generated deepfakes and digitally manipulated media. Utilizing a multi-stack approach, this system evaluates pixel distribution, error levels, and metadata to provide a high-confidence authenticity verdict.

![Version](https://img.shields.io/badge/Version-1.0.0_Patch_Update-blue?style=for-the-badge)
![Tech](https://img.shields.io/badge/Stack-Flask_%7C_OpenCV_%7C_SQLAlchemy-darkgreen?style=for-the-badge)

## 🚀 Key Forensic Capabilities

*   **Error Level Analysis (ELA):** Identifies inconsistencies in compression levels across different image regions.
*   **Deep AI Analysis:** Heuristic texture and frequency auditing to detect synthetic AI-generated patterns.
*   **Noise Variance Mapping:** Detects local re-sampling and inconsistencies in sensor noise distribution.
*   **Advanced Histograms:** Audits for tonal gaps indicative of heavy post-processing or cloning.
*   **Video Spatiotemporal Audit:** Frame-by-frame analysis for motion jitter and temporal inconsistencies.
*   **Automated Email Reporting:** Direct dispatch of forensic verdicts and probability matrices to users.

## 🛠️ Technical Implementation

*   **Backend:** Python / Flask
*   **Database:** SQLAlchemy (SQLite) for Audit Logging
*   **Image Processing:** OpenCV, Pillow, NumPy
*   **Security:** Flask-Login for Investigator Authentication
*   **Messaging:** Flask-Mail for secure report dispatch

## 📦 Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/forensic-media-detection.git
    cd forensic-media-detection
    ```

2.  **Install Dependencies:**
    ```bash
    python -m pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Set up your secure keys and email credentials:
    ```bash
    # Windows (PowerShell)
    $env:SECRET_KEY = "your_secret_key"
    $env:MAIL_USERNAME = "investigator@example.com"
    $env:MAIL_PASSWORD = "your_app_password"
    ```

4.  **Initialise Database:**
    ```bash
    python reset_db.py
    ```

5.  **Run Application:**
    ```bash
    python app.py
    ```

## 🔒 Security Notice
This tool is intended for digital forensic investigations. It provides probabilistic evidence of media authenticity. Final conclusions should always be corroborated by trained forensic investigators.

## 📜 License
Internal Audit License - v1.0
