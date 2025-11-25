#!/usr/bin/env python3
"""
VLA Web Driver - Autonomous Driving with Web Interface

Features:
- 6 nuScenes-style cameras displayed in web UI
- Reset button to respawn vehicle at starting point
- Start/Stop autonomous driving controls
- Real-time vehicle status and waypoint visualization

Usage:
    python vla_web_driver.py [--mock] [--target-speed SPEED] [--port PORT]

Then open http://localhost:8080 in browser.

Author: Claude Code
Date: 2025-11-25
"""

import os
import sys
import time
import math
import threading
import json
from io import BytesIO
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

# Flask for web server
try:
    from flask import Flask, Response, render_template_string, jsonify, request, make_response
except ImportError:
    print("Please install flask: pip install flask")
    sys.exit(1)

# Add CARLA path
CARLA_EGG = os.path.expanduser("~/CARLA_UE5_0.10.0/Carla-0.10.0-Linux-Shipping/PythonAPI/carla/dist/carla-0.10.0-py3.10-linux-x86_64.egg")
if os.path.exists(CARLA_EGG):
    sys.path.insert(0, CARLA_EGG)

import carla

# Local imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from waypoint_controller import WaypointVehicleController, HistoryTracker, ControlCommand


# ===============================
# Camera Configuration (nuScenes style)
# ===============================
NUSCENES_CAMERAS = {
    'CAM_FRONT': {
        'location': carla.Location(x=1.5, y=0.0, z=2.4),
        'rotation': carla.Rotation(pitch=-15, yaw=0, roll=0),
        'fov': 70,
    },
    'CAM_FRONT_LEFT': {
        'location': carla.Location(x=1.2, y=-0.5, z=2.4),
        'rotation': carla.Rotation(pitch=-15, yaw=-55, roll=0),
        'fov': 70,
    },
    'CAM_FRONT_RIGHT': {
        'location': carla.Location(x=1.2, y=0.5, z=2.4),
        'rotation': carla.Rotation(pitch=-15, yaw=55, roll=0),
        'fov': 70,
    },
    'CAM_BACK': {
        'location': carla.Location(x=-2.0, y=0.0, z=2.4),
        'rotation': carla.Rotation(pitch=-15, yaw=180, roll=0),
        'fov': 110,
    },
    'CAM_BACK_LEFT': {
        'location': carla.Location(x=-1.0, y=-0.5, z=2.4),
        'rotation': carla.Rotation(pitch=-15, yaw=-110, roll=0),
        'fov': 70,
    },
    'CAM_BACK_RIGHT': {
        'location': carla.Location(x=-1.0, y=0.5, z=2.4),
        'rotation': carla.Rotation(pitch=-15, yaw=110, roll=0),
        'fov': 70,
    },
}

CAMERA_ORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


# ===============================
# HTML Template for Web UI
# ===============================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VLA Autonomous Driving - 6 Camera View</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0a0a15;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(90deg, #16213e, #0f3460);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }
        .header h1 {
            font-size: 1.3rem;
            color: #e94560;
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .mode-badge {
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.85rem;
        }
        .mode-auto { background: #00c853; color: #000; }
        .mode-manual { background: #ff9800; color: #000; }
        .mode-stopped { background: #666; color: #fff; }

        .main-container {
            display: grid;
            grid-template-columns: 1fr 280px;
            gap: 15px;
            padding: 15px;
            height: calc(100vh - 60px);
        }

        .cameras-section {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        .combined-camera-container {
            background: #16213e;
            border-radius: 8px;
            padding: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .combined-camera-container img {
            display: block;
            width: 100%;
            max-width: 1920px;
            height: auto;
            border-radius: 4px;
        }

        .camera-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: repeat(2, 1fr);
            gap: 8px;
            flex: 1;
        }

        .camera-box {
            background: #1a1a2e;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }
        .camera-box img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .camera-label {
            position: absolute;
            top: 5px;
            left: 5px;
            background: rgba(0,0,0,0.7);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            color: #e94560;
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .panel {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
        }
        .panel h2 {
            font-size: 0.9rem;
            color: #e94560;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #0f3460;
        }

        .speed-display {
            text-align: center;
            padding: 15px;
        }
        .speed-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #00c853;
        }
        .speed-unit {
            font-size: 0.9rem;
            color: #888;
        }

        .stat-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }
        .stat-label { color: #888; }
        .stat-value { font-weight: bold; color: #fff; }

        .control-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .btn {
            padding: 12px 8px;
            border: none;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
        }
        .btn:hover { transform: scale(1.02); filter: brightness(1.1); }
        .btn:active { transform: scale(0.98); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .btn-start { background: #00c853; color: #000; }
        .btn-stop { background: #e94560; color: #fff; }
        .btn-reset { background: #2196f3; color: #fff; grid-column: span 2; }

        .gauge-container {
            display: flex;
            justify-content: space-around;
            margin-top: 10px;
        }
        .gauge {
            text-align: center;
        }
        .gauge-bar {
            width: 60px;
            height: 8px;
            background: #0a0a15;
            border-radius: 4px;
            overflow: hidden;
            margin: 5px auto;
        }
        .gauge-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.1s;
        }
        .gauge-throttle .gauge-fill { background: #00c853; }
        .gauge-brake .gauge-fill { background: #e94560; }
        .gauge-steer .gauge-fill { background: #2196f3; }
        .gauge-label { font-size: 0.7rem; color: #888; }
        .gauge-value { font-size: 0.8rem; font-weight: bold; }

        .waypoints-visual {
            height: 120px;
            background: #0a0a15;
            border-radius: 8px;
            position: relative;
            overflow: hidden;
        }
        .waypoint-dot {
            position: absolute;
            width: 6px;
            height: 6px;
            background: #e94560;
            border-radius: 50%;
            transform: translate(-50%, -50%);
        }
        .ego-marker {
            position: absolute;
            bottom: 15px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 8px solid transparent;
            border-right: 8px solid transparent;
            border-bottom: 16px solid #00c853;
        }

        .position-info {
            font-size: 0.75rem;
            color: #666;
            text-align: center;
            margin-top: 8px;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .recording { animation: pulse 1s infinite; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 VLA Autonomous Driving</h1>
        <div class="header-right">
            <span id="fps-display" style="color: #888; font-size: 0.8rem;">FPS: --</span>
            <div id="mode-badge" class="mode-badge mode-stopped">STOPPED</div>
        </div>
    </div>

    <div class="main-container">
        <div class="cameras-section">
            <!-- Single combined camera feed (uses only 1 connection instead of 6) -->
            <div class="combined-camera-container">
                <img id="combined-feed" src="/video_feed_combined" alt="6 Camera View">
            </div>
        </div>

        <div class="sidebar">
            <div class="panel speed-display">
                <div class="speed-value" id="speed-value">0.0</div>
                <div class="speed-unit">km/h</div>
            </div>

            <div class="panel">
                <h2>Controls</h2>
                <div class="control-grid">
                    <button class="btn btn-start" id="btn-start" onclick="startAuto()">▶ Start</button>
                    <button class="btn btn-stop" id="btn-stop" onclick="stopAuto()">⏹ Stop</button>
                    <button class="btn btn-reset" onclick="resetVehicle()">🔄 Reset Vehicle</button>
                </div>

                <div class="gauge-container">
                    <div class="gauge gauge-throttle">
                        <div class="gauge-label">Throttle</div>
                        <div class="gauge-bar"><div class="gauge-fill" id="throttle-bar"></div></div>
                        <div class="gauge-value" id="throttle-val">0%</div>
                    </div>
                    <div class="gauge gauge-steer">
                        <div class="gauge-label">Steer</div>
                        <div class="gauge-bar"><div class="gauge-fill" id="steer-bar"></div></div>
                        <div class="gauge-value" id="steer-val">0°</div>
                    </div>
                    <div class="gauge gauge-brake">
                        <div class="gauge-label">Brake</div>
                        <div class="gauge-bar"><div class="gauge-fill" id="brake-bar"></div></div>
                        <div class="gauge-value" id="brake-val">0%</div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <h2>VLA Model Stats</h2>
                <div class="stat-row">
                    <span class="stat-label">Inferences</span>
                    <span class="stat-value" id="inferences">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Controls</span>
                    <span class="stat-value" id="controls">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Waypoints</span>
                    <span class="stat-value" id="waypoints">0</span>
                </div>
            </div>

            <div class="panel">
                <h2>Trajectory Preview</h2>
                <div class="waypoints-visual" id="waypoints-canvas">
                    <div class="ego-marker"></div>
                </div>
                <div class="position-info" id="position-info">
                    Position: (0.0, 0.0)
                </div>
            </div>
        </div>
    </div>

    <script>
        let isAutonomous = false;

        function updateStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    // Speed
                    document.getElementById('speed-value').textContent = data.speed.toFixed(1);

                    // Gauges
                    const throttlePct = Math.abs(data.throttle) * 100;
                    const steerPct = (data.steer + 1) * 50;  // -1 to 1 -> 0 to 100
                    const brakePct = Math.abs(data.brake) * 100;

                    document.getElementById('throttle-bar').style.width = throttlePct + '%';
                    document.getElementById('throttle-val').textContent = throttlePct.toFixed(0) + '%';

                    document.getElementById('steer-bar').style.width = steerPct + '%';
                    document.getElementById('steer-val').textContent = (data.steer * 45).toFixed(0) + '°';

                    document.getElementById('brake-bar').style.width = brakePct + '%';
                    document.getElementById('brake-val').textContent = brakePct.toFixed(0) + '%';

                    // Stats
                    document.getElementById('inferences').textContent = data.inferences;
                    document.getElementById('controls').textContent = data.controls;
                    document.getElementById('waypoints').textContent = data.waypoint_count;
                    document.getElementById('fps-display').textContent = 'FPS: ' + data.fps.toFixed(1);

                    // Position
                    if (data.position) {
                        document.getElementById('position-info').textContent =
                            `Position: (${data.position.x.toFixed(1)}, ${data.position.y.toFixed(1)})`;
                    }

                    // Mode badge
                    isAutonomous = data.autonomous;
                    const badge = document.getElementById('mode-badge');
                    if (data.autonomous) {
                        badge.textContent = 'AUTONOMOUS';
                        badge.className = 'mode-badge mode-auto';
                    } else if (data.speed > 0.5) {
                        badge.textContent = 'MANUAL';
                        badge.className = 'mode-badge mode-manual';
                    } else {
                        badge.textContent = 'STOPPED';
                        badge.className = 'mode-badge mode-stopped';
                    }

                    // Update waypoints visualization
                    updateWaypoints(data.waypoints);
                })
                .catch(e => console.log('Status fetch error:', e));
        }

        function updateWaypoints(waypoints) {
            const canvas = document.getElementById('waypoints-canvas');
            canvas.querySelectorAll('.waypoint-dot').forEach(d => d.remove());

            if (!waypoints || waypoints.length === 0) return;

            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            const scale = 8;

            waypoints.forEach((wp, i) => {
                const dot = document.createElement('div');
                dot.className = 'waypoint-dot';
                const x = width/2 + wp[1] * scale;
                const y = height - 25 - wp[0] * scale;
                dot.style.left = Math.max(5, Math.min(width-5, x)) + 'px';
                dot.style.top = Math.max(5, Math.min(height-5, y)) + 'px';
                dot.style.opacity = 1 - i * 0.12;
                canvas.appendChild(dot);
            });
        }

        function sendCommand(endpoint, callback) {
            var xhr = new XMLHttpRequest();
            var url = window.location.origin + endpoint;
            console.log('Sending POST to:', url);
            xhr.open('POST', url, true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.onreadystatechange = function() {
                if (xhr.readyState === 4) {
                    console.log('Response status:', xhr.status);
                    if (xhr.status === 200) {
                        var data = JSON.parse(xhr.responseText);
                        console.log('Response:', data);
                        if (callback) callback(true, data);
                    } else {
                        console.error('Error:', xhr.status, xhr.statusText);
                        if (callback) callback(false, xhr.statusText);
                    }
                }
            };
            xhr.onerror = function() {
                console.error('XHR error');
                if (callback) callback(false, 'Network error');
            };
            xhr.send();
        }

        function startAuto() {
            console.log('=== START BUTTON CLICKED ===');
            var btn = document.getElementById('btn-start');
            btn.disabled = true;
            btn.textContent = 'Starting...';
            sendCommand('/control/start', function(success, data) {
                btn.disabled = false;
                btn.textContent = '▶ Start';
                if (success) {
                    console.log('Autonomous mode STARTED');
                } else {
                    alert('Failed to start: ' + data);
                }
            });
        }

        function stopAuto() {
            console.log('=== STOP BUTTON CLICKED ===');
            var btn = document.getElementById('btn-stop');
            btn.disabled = true;
            btn.textContent = 'Stopping...';
            sendCommand('/control/stop', function(success, data) {
                btn.disabled = false;
                btn.textContent = '⏹ Stop';
                if (success) {
                    console.log('Autonomous mode STOPPED');
                } else {
                    alert('Failed to stop: ' + data);
                }
            });
        }

        function resetVehicle() {
            console.log('=== RESET BUTTON CLICKED ===');
            document.getElementById('btn-start').disabled = true;
            document.getElementById('btn-stop').disabled = true;
            sendCommand('/control/reset', function(success, data) {
                setTimeout(function() {
                    document.getElementById('btn-start').disabled = false;
                    document.getElementById('btn-stop').disabled = false;
                }, 1000);
                if (success) {
                    console.log('Vehicle RESET');
                } else {
                    alert('Failed to reset: ' + data);
                }
            });
        }

        // Update status every 500ms
        setInterval(updateStatus, 500);

        // Initial update
        updateStatus();
    </script>
</body>
</html>
"""


# ===============================
# VLA Web Driver with 6 Cameras
# ===============================
class VLAWebDriver:
    def __init__(self, target_speed=5.0, use_mock=True):
        self.target_speed = target_speed
        self.use_mock = use_mock

        # CARLA
        self.client = None
        self.world = None
        self.vehicle = None
        self.cameras = {}
        self.spawn_point = None
        self.spawn_index = 10  # Default spawn point

        # Camera frames
        self.camera_frames = {name: None for name in CAMERA_ORDER}
        self.frame_locks = {name: threading.Lock() for name in CAMERA_ORDER}

        # State
        self.speed = 0.0
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.position = {'x': 0, 'y': 0, 'z': 0}
        self.autonomous_mode = False
        self.running = False

        # Stats
        self.inference_count = 0
        self.control_count = 0
        self.last_waypoints = []
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)

        # Controller
        self.controller = WaypointVehicleController(
            lateral_controller='pure_pursuit',
            target_speed=target_speed,
        )

        # Flask app
        self.app = Flask(__name__)

        # Enable CORS for all routes
        @self.app.after_request
        def add_cors_headers(response):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response

        # Handle OPTIONS preflight requests
        @self.app.route('/control/<action>', methods=['OPTIONS'])
        def options_handler(action):
            response = make_response()
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response

        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            response = Response(render_template_string(HTML_TEMPLATE), mimetype='text/html')
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response

        @self.app.route('/video_feed/<camera_name>')
        def video_feed(camera_name):
            if camera_name not in CAMERA_ORDER:
                return "Invalid camera", 404
            return Response(
                self._generate_frames(camera_name),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/video_feed_combined')
        def video_feed_combined():
            """Combined 6-camera feed as single stream (only uses 1 browser connection)"""
            return Response(
                self._generate_combined_feed(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/status')
        def status():
            return jsonify({
                'speed': self.speed * 3.6,
                'throttle': self.throttle,
                'steer': self.steer,
                'brake': self.brake,
                'autonomous': self.autonomous_mode,
                'inferences': self.inference_count,
                'controls': self.control_count,
                'waypoint_count': len(self.last_waypoints),
                'waypoints': self.last_waypoints,
                'fps': self.fps,
                'position': self.position,
            })

        @self.app.route('/control/start', methods=['POST'])
        def start_auto():
            self.autonomous_mode = True
            self.controller.reset()
            print(">>> Autonomous mode ENABLED")
            return jsonify({'status': 'ok'})

        @self.app.route('/control/stop', methods=['POST'])
        def stop_auto():
            self.autonomous_mode = False
            if self.vehicle:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))
            print(">>> Autonomous mode DISABLED")
            return jsonify({'status': 'ok'})

        @self.app.route('/control/reset', methods=['POST'])
        def reset_vehicle():
            print(">>> Resetting vehicle...")
            self._reset_vehicle()
            return jsonify({'status': 'ok'})

    def _generate_frames(self, camera_name):
        """Generate MJPEG frames for a specific camera"""
        while True:
            with self.frame_locks[camera_name]:
                frame = self.camera_frames.get(camera_name)
                if frame is not None:
                    frame = frame.copy()
                else:
                    # Black frame with camera name
                    frame = np.zeros((270, 480, 3), dtype=np.uint8)
                    cv2.putText(frame, camera_name, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.05)  # ~20 FPS per camera

    def _generate_combined_feed(self):
        """Generate combined 6-camera video feed as single stream (solves browser connection limit)"""
        # Resolution per camera: 640x360, total combined: 1920x720
        CAM_W, CAM_H = 640, 360

        while True:
            # Create a 2x3 grid: top row = front cameras, bottom row = back cameras
            combined = np.zeros((CAM_H * 2, CAM_W * 3, 3), dtype=np.uint8)

            # Camera positions in grid
            positions = {
                'CAM_FRONT_LEFT': (0, 0),
                'CAM_FRONT': (0, CAM_W),
                'CAM_FRONT_RIGHT': (0, CAM_W * 2),
                'CAM_BACK_LEFT': (CAM_H, 0),
                'CAM_BACK': (CAM_H, CAM_W),
                'CAM_BACK_RIGHT': (CAM_H, CAM_W * 2),
            }

            for name, (y, x) in positions.items():
                with self.frame_locks[name]:
                    frame = self.camera_frames.get(name)
                    if frame is not None:
                        # Resize to target resolution
                        if frame.shape[:2] != (CAM_H, CAM_W):
                            frame = cv2.resize(frame, (CAM_W, CAM_H))
                        combined[y:y+CAM_H, x:x+CAM_W] = frame
                    else:
                        # Draw camera label on black background
                        cv2.putText(combined, name, (x + 20, y + CAM_H//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)

            # Add camera labels with shadow for better visibility
            labels = ['FRONT LEFT', 'FRONT', 'FRONT RIGHT', 'BACK LEFT', 'BACK', 'BACK RIGHT']
            label_positions = [
                (10, CAM_H - 10), (CAM_W + 10, CAM_H - 10), (CAM_W * 2 + 10, CAM_H - 10),
                (10, CAM_H * 2 - 10), (CAM_W + 10, CAM_H * 2 - 10), (CAM_W * 2 + 10, CAM_H * 2 - 10)
            ]
            for label, (x, y) in zip(labels, label_positions):
                # Shadow
                cv2.putText(combined, label, (x + 2, y + 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                # Text
                cv2.putText(combined, label, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Encode as JPEG with good quality
            _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.05)  # ~20 FPS

    def _camera_callback(self, camera_name):
        """Create callback for a specific camera"""
        def callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]

            with self.frame_locks[camera_name]:
                self.camera_frames[camera_name] = array

            # Update FPS (only from front camera)
            if camera_name == 'CAM_FRONT':
                now = time.time()
                self.frame_times.append(now)
                if len(self.frame_times) > 1:
                    self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])

        return callback

    def connect(self, host='localhost', port=2000):
        """Connect to CARLA"""
        print("Connecting to CARLA...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        version = self.client.get_server_version()
        print(f"  ✓ Connected to CARLA {version}")

        self.world = self.client.get_world()
        print(f"  ✓ Map: {self.world.get_map().name}")

    def spawn_vehicle(self):
        """Spawn ego vehicle with 6 cameras"""
        print("Spawning vehicle...")

        # Clean up existing
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor.destroy()
        for actor in self.world.get_actors().filter('sensor.*'):
            actor.destroy()
        time.sleep(0.5)

        # Get vehicle blueprint
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.filter('vehicle.dodge.charger')[0]

        # Set color to red
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')

        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = spawn_points[self.spawn_index % len(spawn_points)]

        # Spawn vehicle
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, self.spawn_point)
        if self.vehicle is None:
            # Try other spawn points
            for i in range(len(spawn_points)):
                self.spawn_point = spawn_points[i]
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, self.spawn_point)
                if self.vehicle is not None:
                    self.spawn_index = i
                    break

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")

        print(f"  ✓ Vehicle spawned (ID: {self.vehicle.id})")

        # Spawn 6 cameras
        self._spawn_cameras()

    def _spawn_cameras(self):
        """Spawn all 6 nuScenes-style cameras"""
        print("Setting up 6 cameras...")

        bp_library = self.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        # Higher resolution for better quality (640x360 per camera, 1920x720 combined)
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '360')

        for name, config in NUSCENES_CAMERAS.items():
            # Set FOV
            camera_bp.set_attribute('fov', str(config['fov']))

            # Create transform
            transform = carla.Transform(
                config['location'],
                config['rotation']
            )

            # Spawn camera
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
            camera.listen(self._camera_callback(name))
            self.cameras[name] = camera
            print(f"  ✓ {name}")

    def _reset_vehicle(self):
        """Reset vehicle to spawn point"""
        # Stop autonomous mode
        self.autonomous_mode = False

        if self.vehicle:
            # Stop the vehicle
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))
            time.sleep(0.3)

            # Teleport to spawn point
            self.vehicle.set_transform(self.spawn_point)

            # Reset velocity
            self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))

            # Reset controls
            self.vehicle.apply_control(carla.VehicleControl())

            time.sleep(0.2)

        # Reset stats
        self.controller.reset()
        self.inference_count = 0
        self.control_count = 0
        self.last_waypoints = []
        self.speed = 0.0
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0

        print("  ✓ Vehicle reset to spawn point")

    def generate_waypoints(self, command="keep forward"):
        """Generate mock waypoints"""
        waypoints = []
        x, y = 0.0, 0.0
        for i in range(6):
            x += 2.0
            if command == "turn left":
                y += 0.3 * (i + 1)
            elif command == "turn right":
                y -= 0.3 * (i + 1)
            waypoints.append((x, y))
        return waypoints

    def control_loop(self):
        """Main control loop"""
        print("\nControl loop started")
        print(f"  Target speed: {self.target_speed * 3.6:.1f} km/h")

        last_inference_time = 0.0
        inference_interval = 0.2  # 5 Hz

        self.running = True

        while self.running:
            current_time = time.time()

            # Update vehicle state
            if self.vehicle:
                velocity = self.vehicle.get_velocity()
                self.speed = math.sqrt(velocity.x**2 + velocity.y**2)

                transform = self.vehicle.get_transform()
                self.position = {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z,
                }

            # Run inference
            if self.autonomous_mode and current_time - last_inference_time >= inference_interval:
                self.last_waypoints = self.generate_waypoints("keep forward")
                self.inference_count += 1
                last_inference_time = current_time

            # Apply control
            if self.autonomous_mode and self.vehicle and self.last_waypoints:
                cmd = self.controller.compute_control(self.last_waypoints, self.speed)

                self.throttle = cmd.throttle
                self.steer = cmd.steer
                self.brake = cmd.brake

                control = carla.VehicleControl(
                    throttle=cmd.throttle,
                    steer=cmd.steer,
                    brake=cmd.brake,
                )
                self.vehicle.apply_control(control)
                self.control_count += 1
            elif not self.autonomous_mode:
                # Clear control values when not autonomous
                self.throttle = 0.0
                self.steer = 0.0
                self.brake = 0.0

            time.sleep(0.05)  # 20 Hz

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        print("\nCleaning up...")

        # Destroy cameras
        for name, camera in self.cameras.items():
            try:
                camera.stop()
                camera.destroy()
            except:
                pass

        # Destroy vehicle
        if self.vehicle:
            try:
                self.vehicle.destroy()
            except:
                pass

        print("✓ Cleanup complete")

    def run(self, web_port=8080):
        """Run the VLA driver with web interface"""
        try:
            self.connect()
            self.spawn_vehicle()

            # Wait for cameras
            print("\nWaiting for camera data...")
            time.sleep(2.0)

            # Start control loop in thread
            control_thread = threading.Thread(target=self.control_loop, daemon=True)
            control_thread.start()

            # Get Tailscale IP
            tailscale_ip = None
            try:
                import subprocess
                result = subprocess.run(['tailscale', 'ip', '-4'], capture_output=True, text=True)
                if result.returncode == 0:
                    tailscale_ip = result.stdout.strip()
            except:
                pass

            print(f"\n{'='*50}")
            print(f"  Web interface running at:")
            print(f"    http://localhost:{web_port}")
            if tailscale_ip:
                print(f"    http://{tailscale_ip}:{web_port}  (Tailscale)")
            print(f"\n  Click 'Start' to begin autonomous driving")
            print(f"  Click 'Reset' to return to spawn point")
            print(f"{'='*50}\n")

            # Run Flask (disable reloader to avoid issues)
            self.app.run(host='0.0.0.0', port=web_port, threaded=True, use_reloader=False)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLA Autonomous Driving with Web Interface")
    parser.add_argument("--mock", action="store_true", help="Use mock VLA model")
    parser.add_argument("--target-speed", type=float, default=5.0, help="Target speed in m/s")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument("--spawn", type=int, default=10, help="Spawn point index")
    args = parser.parse_args()

    driver = VLAWebDriver(
        target_speed=args.target_speed,
        use_mock=args.mock,
    )
    driver.spawn_index = args.spawn
    driver.run(web_port=args.port)


if __name__ == "__main__":
    main()
