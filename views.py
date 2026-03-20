from django.shortcuts import render
from django.http import HttpResponse
import subprocess
import sys
import os

# ── Point this to your existing working script ──────────────────────────────
GESTURE_SCRIPT = r"C:\Users\Smit\PyCharmMiscProject\live_gesture.py"


def index(request):
    return render(request, 'index.html')


def start_camera(request):
    """
    Launches hand_gesture_recognition_v3.py in a SEPARATE process.
    Django keeps running normally.
    Your OpenCV window opens on the desktop exactly as before.
    Press Q in the OpenCV window to stop the camera.
    """
    if not os.path.exists(GESTURE_SCRIPT):
        return HttpResponse(f"❌ Script not found: {GESTURE_SCRIPT}")

    subprocess.Popen([sys.executable, GESTURE_SCRIPT])

    return render(request, 'index.html', {
        'message': '✅ Camera started! OpenCV window will open on your desktop. Press Q to quit it.'
    })


def stop_camera(request):
    """Kills the gesture script process on Windows."""
    os.system('taskkill /f /fi "IMAGENAME eq python.exe" /fi "WINDOWTITLE eq Fruit*"')
    return render(request, 'index.html', {
        'message': '🛑 Camera stopped.'
    })