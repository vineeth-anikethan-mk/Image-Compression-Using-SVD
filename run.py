#!/usr/bin/env python3
"""
SVD Image Compression Studio - Fast Modern Web Application Launcher
"""
import uvicorn
import webbrowser
import threading
import time

def open_browser(url):
    time.sleep(1.2)
    print(f"Opening browser at {url}...")
    webbrowser.open(url)

if __name__ == "__main__":
    port = 8000
    url = f"http://127.0.0.1:{port}"
    print(f"\n=======================================================")
    print(f"✨ SVD Image Compression Studio (Modern Web UI)")
    print(f"🌐 Server running at: {url}")
    print(f"=======================================================\n")
    
    # Open browser in a separate thread
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)
