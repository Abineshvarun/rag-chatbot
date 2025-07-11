import subprocess
# import os
# import time

# # Step 1: Start Ollama server
# try:
#     ollama_process = subprocess.Popen("ollama serve", shell=True)
#     print("✅ Ollama server started.")
# except Exception as e:
#     print(f"❌ Failed to start Ollama: {e}")

# # Step 2: Wait for Ollama to initialize
# time.sleep(5)

# # Step 3: Start Streamlit app using full path to streamlit.exe
# try:
#     streamlit_path = r"C:\Users\abine\OneDrive\Desktop\LLM\venv\Scripts\streamlit.exe"  # ✅ Replace if different
#     app_path = os.path.join(os.getcwd(), "app.py")

#     print(f"🚀 Launching Streamlit app at: {app_path}")
#     subprocess.call([streamlit_path, "run", app_path])
# except Exception as e:
#     print(f"❌ Failed to launch Streamlit: {e}")