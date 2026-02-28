"""Karma AI — One-click startup script.

Automatically:
  1. Starts the AI Voice Detector API (Dataset/api.py) on port 8000
  2. Starts ngrok tunnel on port 5000
  3. Updates BASE_URL in .env with the new ngrok URL
  4. Configures Twilio webhooks to point to the ngrok URL
  5. Launches the Flask server

Usage:
  python start.py

Requirements:
  pip install pyngrok
  (or install ngrok CLI and set authtoken)
"""

import os
import re
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", 5000))
ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")

# AI Voice Detector paths
DATASET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Dataset"))
DATASET_PYTHON = os.path.join(DATASET_DIR, "venv", "Scripts", "python.exe")
CLASSIFIER_API_SCRIPT = os.path.join(DATASET_DIR, "api.py")
CLASSIFIER_MODEL_DIR = os.path.join(DATASET_DIR, "model_output", "best_model")


def check_ngrok_installed():
    """Check if pyngrok is available, install if not."""
    try:
        from pyngrok import ngrok  # noqa: F401
        return True
    except ImportError:
        print("[!] pyngrok not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
        return True


def start_ngrok(port: int) -> str:
    """Start ngrok tunnel and return the public HTTPS URL."""
    from pyngrok import ngrok, conf

    # Check if NGROK_AUTHTOKEN is set
    authtoken = os.getenv("NGROK_AUTHTOKEN", "")
    if authtoken:
        conf.get_default().auth_token = authtoken

    print(f"[*] Starting ngrok tunnel on port {port}...")
    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url

    # Ensure HTTPS
    if public_url.startswith("http://"):
        public_url = public_url.replace("http://", "https://")

    print(f"[+] Ngrok tunnel: {public_url}")
    return public_url


def update_env_base_url(ngrok_url: str):
    """Update BASE_URL in .env file."""
    with open(ENV_FILE, "r") as f:
        content = f.read()

    # Replace existing BASE_URL
    if "BASE_URL=" in content:
        content = re.sub(r"BASE_URL=.*", f"BASE_URL={ngrok_url}", content)
    else:
        content += f"\nBASE_URL={ngrok_url}\n"

    with open(ENV_FILE, "w") as f:
        f.write(content)

    # Also update the current environment
    os.environ["BASE_URL"] = ngrok_url
    print(f"[+] Updated .env BASE_URL = {ngrok_url}")


def setup_twilio_webhooks(ngrok_url: str):
    """Configure Twilio phone number webhooks."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    phone_number = os.getenv("TWILIO_PHONE_NUMBER")

    if not all([account_sid, auth_token, phone_number]):
        print("[!] Twilio credentials not set in .env — skipping webhook setup")
        print("    Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER")
        return

    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)

        numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
        if not numbers:
            print(f"[!] Phone number {phone_number} not found in Twilio account")
            return

        numbers[0].update(
            voice_url=f"{ngrok_url}/voice",
            voice_method="POST",
            status_callback=f"{ngrok_url}/call-status",
            status_callback_method="POST",
        )

        print(f"[+] Twilio webhooks configured:")
        print(f"    Voice URL:       {ngrok_url}/voice")
        print(f"    Status Callback: {ngrok_url}/call-status")
        print(f"    Phone Number:    {phone_number}")
    except Exception as e:
        print(f"[!] Failed to configure Twilio: {e}")


def start_classifier_api():
    """Start the AI Voice Detector API (Dataset/api.py) as a background process.

    Returns the Popen handle, or None if it couldn't start.
    """
    # Check all required paths exist
    if not os.path.exists(DATASET_PYTHON):
        print(f"[!] Dataset venv not found at {DATASET_PYTHON}")
        print("    Voice classifier will NOT be available (all callers treated as human)")
        return None

    if not os.path.exists(CLASSIFIER_API_SCRIPT):
        print(f"[!] Dataset/api.py not found at {CLASSIFIER_API_SCRIPT}")
        print("    Voice classifier will NOT be available")
        return None

    if not os.path.isdir(CLASSIFIER_MODEL_DIR):
        print(f"[!] Classifier model not found at {CLASSIFIER_MODEL_DIR}")
        print("    Run Dataset/train.py first. Voice classifier will NOT be available")
        return None

    print("[*] Starting AI Voice Detector API on port 8000...")
    proc = subprocess.Popen(
        [DATASET_PYTHON, CLASSIFIER_API_SCRIPT],
        cwd=DATASET_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for the API to become healthy (model loading can take a while)
    print("[*] Waiting for classifier model to load (this may take 15-30s)...")
    for i in range(60):  # up to 60 seconds
        # Check process is still alive
        if proc.poll() is not None:
            print(f"[!] Classifier API exited with code {proc.returncode}")
            output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            if output:
                for line in output.strip().split("\n")[-5:]:
                    print(f"    {line}")
            return None

        time.sleep(1)
        try:
            req = Request("http://localhost:8000/health")
            with urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print("[+] AI Voice Detector API ready on port 8000")
                    return proc
        except Exception:
            if (i + 1) % 10 == 0:
                print(f"    Still loading... ({i + 1}s)")

    print("[!] AI Voice Detector API failed to start within 60s")
    print("    Voice classifier will NOT be available")
    return proc


def main():
    print("=" * 55)
    print("  KARMA AI — Automatic Startup")
    print("=" * 55)
    print()

    # Step 1: Start AI Voice Detector API
    classifier_proc = start_classifier_api()
    print()

    # Step 2: Start ngrok
    check_ngrok_installed()
    ngrok_url = start_ngrok(PORT)
    print()

    # Step 3: Update .env
    update_env_base_url(ngrok_url)
    print()

    # Step 4: Configure Twilio
    setup_twilio_webhooks(ngrok_url)
    print()

    # Step 5: Print URLs
    print("=" * 55)
    print("  KARMA AI is starting...")
    print("=" * 55)
    print(f"  Web Voice Call:  http://localhost:{PORT}/")
    print(f"  Live Dashboard:  http://localhost:{PORT}/dashboard/live-calls.html")
    print(f"  Analytics:       http://localhost:{PORT}/dashboard/analytics.html")
    print(f"  Archive:         http://localhost:{PORT}/dashboard/archive.html")
    print(f"  Twilio Webhook:  {ngrok_url}/voice")
    print(f"  Health Check:    http://localhost:{PORT}/health")
    if classifier_proc:
        print(f"  Voice Classifier: http://localhost:8000/health")
    else:
        print(f"  Voice Classifier: UNAVAILABLE (all callers treated as human)")
    print("=" * 55)
    print()

    # Step 6: Launch Flask server
    # Use subprocess so ngrok stays alive
    try:
        subprocess.run([
            sys.executable, "app.py"
        ], cwd=os.path.dirname(__file__))
    finally:
        # Clean up classifier API process
        if classifier_proc and classifier_proc.poll() is None:
            print("[*] Shutting down AI Voice Detector API...")
            classifier_proc.terminate()
            classifier_proc.wait(timeout=5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[*] Shutting down Karma AI...")
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except Exception:
            pass
        print("[*] Shutdown complete.")
