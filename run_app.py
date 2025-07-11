import subprocess
import time
import sys
import os
import requests

def is_ollama_running():
    try:
        r = requests.get("http://localhost:11434", timeout=2)
        return r.status_code == 200
    except:
        return False

def start_ollama():
    print("🧠 Starting Ollama server...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

def wait_until_ollama_responds(timeout=15):
    print("⌛ Waiting for Ollama server...")
    for i in range(timeout):
        if is_ollama_running():
            print("✅ Ollama is ready.")
            return True
        time.sleep(1)
    print("❌ Ollama did not respond in time.")
    return False

def run():
    if not is_ollama_running():
        start_ollama()
        if not wait_until_ollama_responds():
            print("❌ Failed to start Ollama.")
            sys.exit(1)

    # Now Ollama is up. Run your actual app.
    os.environ["STREAMLIT_WATCH_DIR"] = "false"  # <- Add this line
    streamlit_path = r"C:\Users\abine\OneDrive\Desktop\LLM\venv\Scripts\streamlit.exe"
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([streamlit_path, "run", app_path])

if __name__ == "__main__":
    run()
