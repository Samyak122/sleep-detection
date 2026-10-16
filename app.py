from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import threading
import socket
import requests
from queue import Queue

app = Flask(__name__)

# Global variables
camera = None
detector = FaceMeshDetector(maxFaces=1)
sleep_detected = False
esp_connected = False
esp_ip = None  # ESP IP address
esp_port = 80  # ESP web server port
command_queue = Queue()

# Sleep detection parameters
SLEEP_TIME = 3
EYE_CLOSED_RATIO = 0.20
calibration_values = []
calibrated = False
closed_start = None
current_ear = 0.0
eyes_closed_duration = 0.0  # 👈 NEW: Track how long eyes have been closed

# ESP32 Communication (WiFi)
def send_to_esp(command):
    """Send command to ESP32 via HTTP"""
    global esp_connected, esp_ip
    
    if not esp_ip:
        print("❌ ESP IP not set")
        return False
    
    try:
        url = f"http://{esp_ip}:{esp_port}/command"
        response = requests.get(url, params={'cmd': command}, timeout=5)
        
        if response.status_code == 200:
            esp_connected = True
            print(f"📤 Sent to ESP32 ({esp_ip}): {command}")
            return True
        else:
            esp_connected = False
            print(f"❌ ESP32 HTTP Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ESP32 Error: {e}")
        esp_connected = False
        return False

def generate_frames():
    """Video streaming generator function"""
    global camera, detector, sleep_detected, calibrated
    global EYE_CLOSED_RATIO, closed_start, current_ear, calibration_values
    global eyes_closed_duration  # 👈 NEW
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, faces = detector.findFaceMesh(frame, draw=False)
            
            if faces:
                face = faces[0]
                
                try:
                    # Calculate Eye Aspect Ratio (EAR)
                    left_eye_top = face[159]
                    left_eye_bottom = face[23]
                    left_eye_left = face[130]
                    left_eye_right = face[243]
                    
                    right_eye_top = face[386]
                    right_eye_bottom = face[253]
                    right_eye_left = face[359]
                    right_eye_right = face[463]
                    
                    left_vertical = np.linalg.norm(np.array(left_eye_top) - np.array(left_eye_bottom))
                    right_vertical = np.linalg.norm(np.array(right_eye_top) - np.array(right_eye_bottom))
                    left_horizontal = np.linalg.norm(np.array(left_eye_left) - np.array(left_eye_right))
                    right_horizontal = np.linalg.norm(np.array(right_eye_left) - np.array(right_eye_right))
                    
                    left_ear = left_vertical / left_horizontal if left_horizontal > 0 else 0
                    right_ear = right_vertical / right_horizontal if right_horizontal > 0 else 0
                    current_ear = (left_ear + right_ear) / 2
                    
                    # Calibration phase
                    if not calibrated:
                        if len(calibration_values) < 30:
                            calibration_values.append(current_ear)
                            cv2.putText(frame, f"CALIBRATING... Keep Eyes OPEN! {len(calibration_values)}/30", 
                                      (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                            cv2.putText(frame, f"Current EAR: {current_ear*100:.1f}", 
                                      (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                        else:
                            open_ear_values = np.array(calibration_values)
                            avg_open_ear = np.mean(open_ear_values)
                            EYE_CLOSED_RATIO = avg_open_ear * 0.7
                            calibrated = True
                            print(f"\n✅ Calibration complete! Threshold: {EYE_CLOSED_RATIO*100:.1f}")
                    
                    # Sleep detection phase
                    if calibrated:
                        ear_display = current_ear * 100
                        threshold_display = EYE_CLOSED_RATIO * 100
                        
                        # Eye state detection
                        if current_ear < EYE_CLOSED_RATIO:
                            # 👇 EYES CLOSED
                            if closed_start is None:
                                closed_start = time.time()
                            
                            # 👇 UPDATE DURATION
                            eyes_closed_duration = time.time() - closed_start
                            status_color = (0, 0, 255)
                            status_text = "EYES CLOSED"
                            
                            # Draw duration on video
                            cv2.putText(frame, f"Closed: {eyes_closed_duration:.1f}s/{SLEEP_TIME}s", 
                                      (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                            
                            # Warning bar
                            warning_width = int((eyes_closed_duration / SLEEP_TIME) * 300)
                            cv2.rectangle(frame, (30, 200), (30 + min(warning_width, 300), 215), 
                                        (0, 0, 255), -1)
                            
                            # Sleep detection
                            if eyes_closed_duration > SLEEP_TIME and not sleep_detected:
                                sleep_detected = True
                                if esp_connected:
                                    send_to_esp("STOP")
                                cv2.putText(frame, "🚨 SLEEP DETECTED! 🚨", 
                                          (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                                print("🚨 SLEEP DETECTED!")
                        else:
                            # 👇 EYES OPEN
                            if closed_start is not None:
                                closed_start = None
                                eyes_closed_duration = 0.0  # Reset duration
                                if sleep_detected:
                                    sleep_detected = False
                                    if esp_connected:
                                        send_to_esp("RESUME")
                                    print("▶️ Resumed")
                            
                            status_color = (0, 255, 0)
                            status_text = "AWAKE"
                            cv2.putText(frame, f"Awake", 
                                      (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        
                        # Display EAR and threshold
                        cv2.putText(frame, f"EAR: {ear_display:.1f}", 
                                  (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                        cv2.putText(frame, f"Threshold: {threshold_display:.1f}", 
                                  (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                        cv2.putText(frame, status_text, 
                                  (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
                        
                        # 👇 SHOW ESP STATUS ON VIDEO
                        if esp_connected:
                            cv2.putText(frame, "ESP32: Connected", 
                                      (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        else:
                            cv2.putText(frame, "ESP32: Disconnected", 
                                      (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                
                except Exception as e:
                    print(f"Error: {e}")
            
            else:
                cv2.putText(frame, "No Face Detected", 
                          (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # FPS display
            fps = int(camera.get(cv2.CAP_PROP_FPS))
            cv2.putText(frame, f"FPS: {fps}", 
                      (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current detection status"""
    global calibrated, sleep_detected, esp_connected, current_ear, EYE_CLOSED_RATIO
    global eyes_closed_duration  # 👈 NEW
    
    status = {
        'calibrated': calibrated,
        'sleep_detected': sleep_detected,
        'esp_connected': esp_connected,
        'ear_value': float(current_ear * 100) if current_ear else 0,
        'threshold': float(EYE_CLOSED_RATIO * 100) if EYE_CLOSED_RATIO else 20,
        'eyes_closed_duration': float(eyes_closed_duration),  # 👈 NEW: Send duration to web app
        'timestamp': time.time()
    }
    return jsonify(status)

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global calibration_values, calibrated
    calibration_values = []
    calibrated = False
    return jsonify({'status': 'calibrating'})

@app.route('/api/connect_esp', methods=['POST'])
def connect_esp():
    global esp_connected, esp_ip
    data = request.get_json()
    ip = data.get('ip', '').strip()
    
    if not ip:
        return jsonify({'connected': False, 'error': 'IP address required'})
    
    esp_ip = ip
    # Test connection
    success = send_to_esp("TEST")
    return jsonify({'connected': esp_connected, 'ip': esp_ip})

@app.route('/api/send_command', methods=['POST'])
def send_command():
    command = request.json.get('command', '')
    success = send_to_esp(command)
    return jsonify({'success': success, 'command': command})

if __name__ == '__main__':
    print("🚀 Starting Sleep Detection Server...")
    print("📸 Camera initializing...")
    print("🌐 Web interface: http://127.0.0.1:5000")
    print("")
    print("📶 ESP32 WiFi: Enter IP address in web interface")
    print("")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)


# Add this function to handle ESP8266 TCP connections
def start_esp_server():
    """Start a TCP server for ESP8266 connections"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 5000))
    server.listen(5)
    
    print("📡 ESP8266 TCP Server started on port 5000")
    
    while True:
        client, addr = server.accept()
        print(f"✅ ESP8266 connected from {addr}")
        # Store client for sending commands
        esp_clients.append(client)
        
        # Handle client in separate thread
        client_thread = threading.Thread(target=handle_esp_client, args=(client,))
        client_thread.daemon = True
        client_thread.start()

# Start ESP server in background when Flask starts
esp_thread = threading.Thread(target=start_esp_server)
esp_thread.daemon = True
esp_thread.start()