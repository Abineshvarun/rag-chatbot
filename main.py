import os
import sys
import subprocess
import tempfile
import urllib.request
import time
import threading
import tkinter as tk
from tkinter import messagebox
import webbrowser
import ctypes

class StreamlitLauncher:
    def __init__(self):
        self.progress_window = None
        self.status_label = None
        
    def show_progress_window(self):
        """Show progress window during setup"""
        self.progress_window = tk.Tk()
        self.progress_window.title("Setting up your application...")
        self.progress_window.geometry("400x150")
        self.progress_window.resizable(False, False)
        
        # Center the window
        self.progress_window.eval('tk::PlaceWindow . center')
        
        # Status label
        self.status_label = tk.Label(
            self.progress_window, 
            text="Initializing...", 
            font=("Arial", 10)
        )
        self.status_label.pack(pady=30)
        
        # Progress info
        info_label = tk.Label(
            self.progress_window, 
            text="Please wait while we set up everything for you.\nThis may take a few minutes on first run.",
            font=("Arial", 8),
            fg="gray"
        )
        info_label.pack(pady=10)
        
        self.progress_window.update()
    
    def update_status(self, message):
        """Update progress status"""
        if self.status_label:
            self.status_label.config(text=message)
            self.progress_window.update()
    
    def close_progress_window(self):
        """Close progress window"""
        if self.progress_window:
            self.progress_window.destroy()
    
    def is_admin(self):
        """Check if running as administrator"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    
    def run_as_admin(self):
        """Restart as administrator if needed"""
        if not self.is_admin():
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, " ".join(sys.argv), None, 1
            )
            sys.exit(0)
    
    def install_ollama_silently(self):
        """Download and install Ollama silently"""
        self.update_status("Downloading Ollama...")
        
        ollama_url = "https://ollama.com/download/OllamaSetup.exe"
        installer_path = os.path.join(tempfile.gettempdir(), "OllamaSetup.exe")
        
        # Download Ollama installer
        urllib.request.urlretrieve(ollama_url, installer_path)
        
        self.update_status("Installing Ollama...")
        
        # Install silently
        subprocess.run([installer_path, "/S"], check=True)
        
        # Clean up installer
        try:
            os.remove(installer_path)
        except:
            pass
    
    def setup_ollama_service(self):
        """Start Ollama service and download model"""
        self.update_status("Starting Ollama service...")
        
        # Start Ollama service in background
        subprocess.Popen(
            ["ollama", "serve"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        # Wait for service to start
        time.sleep(5)
        
        self.update_status("Downloading AI model (this may take a while)...")
        
        # Download model with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    ["ollama", "pull", "llama3.2:1b"],  # Using smaller model for faster download
                    timeout=600,  # 10 minute timeout
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                if result.returncode == 0:
                    break
                else:
                    if attempt == max_retries - 1:
                        raise Exception(f"Model download failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self.update_status(f"Download timeout, retrying... (Attempt {attempt + 2})")
                # Kill and restart Ollama
                subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(3)
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
                time.sleep(5)
    
    def create_virtual_env_and_install_deps(self):
        """Create virtual environment and install dependencies"""
        self.update_status("Setting up Python environment...")
        
        venv_dir = os.path.join(os.getcwd(), "app_env")
        
        if not os.path.exists(venv_dir):
            subprocess.run([sys.executable, "-m", "venv", venv_dir], 
                          check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        # Install required packages
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
        packages = [
            "streamlit==1.28.0",
            "ollama",
            "chromadb",
            "langchain",
            "pymupdf",
            "python-docx",
            "python-pptx"
        ]
        
        for package in packages:
            subprocess.run([pip_path, "install", package], 
                          check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    
    def launch_streamlit(self):
        """Launch the Streamlit application"""
        self.update_status("Launching application...")
        
        venv_dir = os.path.join(os.getcwd(), "app_env")
        streamlit_path = os.path.join(venv_dir, "Scripts", "streamlit.exe")
        
        # Launch Streamlit
        process = subprocess.Popen(
            [streamlit_path, "run", "app.py", "--server.headless=true"],
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        # Wait a moment for Streamlit to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open("http://localhost:8501")
        
        return process
    
    def run(self):
        """Main execution function"""
        try:
            # Show progress window
            self.show_progress_window()
            
            # Check if Ollama is installed
            try:
                subprocess.run(["ollama", "--version"], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                             check=True)
                ollama_installed = True
            except:
                ollama_installed = False
            
            # Install Ollama if needed
            if not ollama_installed:
                self.run_as_admin()  # Request admin rights for installation
                self.install_ollama_silently()
            
            # Setup Ollama service and model
            self.setup_ollama_service()
            
            # Setup Python environment
            self.create_virtual_env_and_install_deps()
            
            # Close progress window
            self.close_progress_window()
            
            # Launch Streamlit
            streamlit_process = self.launch_streamlit()
            
            # Show success message
            messagebox.showinfo(
                "Success!", 
                "Application is now running!\nYour browser should open automatically.\n\nClose this window when you're done using the app."
            )
            
            # Wait for user to close the message box, then terminate
            streamlit_process.terminate()
            
        except Exception as e:
            self.close_progress_window()
            messagebox.showerror("Error", f"Setup failed: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    launcher = StreamlitLauncher()
    launcher.run()
