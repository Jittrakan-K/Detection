"""
Index entrypoint for AI Object Counter.
Runs the web application and automatically opens the browser.
"""
import os
import sys
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run the main web application
    import web_app
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  AI Object Counter Web Dashboard started!")
    print(f"  --> Opening http://localhost:{port}")
    print(f"=======================================================\n")
    web_app.app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
