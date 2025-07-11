import PyInstaller.__main__
import os
import shutil

# Clean previous builds
if os.path.exists('dist'):
    shutil.rmtree('dist')
if os.path.exists('build'):
    shutil.rmtree('build')

# PyInstaller arguments
args = [
    'main.py',
    '--onefile',
    '--windowed',  # No console window
    '--name=MyStreamlitApp',
    '--add-data=app.py;.',  # Include your Streamlit app
    '--add-data=documents;documents',  # Include any data folders
    '--icon=icon.ico',  # Optional: Add an icon
    '--hidden-import=streamlit',
    '--hidden-import=ollama',
    '--hidden-import=tkinter',
    '--collect-data=streamlit',
    '--collect-data=altair',
    '--distpath=./dist'
]

PyInstaller.__main__.run(args)

print("✅ EXE created successfully in ./dist/ folder")
