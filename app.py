from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import uuid
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
GRAPH_FOLDER = "static/graphs"
LOG_FOLDER = "static/logs"
CRIME_MODEL_PATH = "models/crime_best.pt"
CROWD_MODEL_PATH = "models/crowd_best.pt"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Threshold configuration
CROWD_WARNING_THRESHOLD = 25
CROWD_CRITICAL_THRESHOLD = 50

# Create necessary directories
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, GRAPH_FOLDER, LOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Load YOLO models
try:
    crime_model = YOLO(CRIME_MODEL_PATH)
    crowd_model = YOLO(CROWD_MODEL_PATH)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    crime_model, crowd_model = None, None

# Global variables
crowd_data = []
detection_history = []
alert_history = []
video_analysis_results = {}  # Store video analysis results for UI access

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded!"})

    file = request.files["file"]
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected!"})
    
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"})
    
    # Generate unique filename to prevent overwriting
    original_filename = secure_filename(file.filename)
    file_extension = original_filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
    
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    mode = request.form.get("mode", "crime")  # Default to crime if not specified
    
    if mode == "crime":
        response = crime_detection(file_path, unique_filename, original_filename)
    elif mode == "crowd":
        if file_extension in ["mp4", "avi", "mov"]:  
            response = crowd_detection_video(file_path, unique_filename, original_filename)
        else:  
            response = crowd_detection_image(file_path, unique_filename, original_filename)
    else:
        response = jsonify({"status": "error", "message": "Invalid mode selected."})
    
    # Log the detection
    log_detection(mode, original_filename, response.json if hasattr(response, 'json') else {})
    
    return response

def crime_detection(image_path, filename, original_filename):
    try:
        if crime_model is None:
            raise Exception("Crime model not loaded correctly!")

        results = crime_model(image_path)
        confidence_threshold = 0.5  # Adjustable threshold
        detections = {}
        
        # Process detections
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = crime_model.names[class_id] if class_id < len(crime_model.names) else "Unknown"
                
                if confidence >= confidence_threshold:
                    if class_name not in detections or confidence > detections[class_name]:
                        detections[class_name] = round(confidence * 100, 2)  # Convert to percentage

        if not detections:
            return jsonify({
                "status": "success", 
                "message": "No suspicious activities detected.",
                "original_filename": original_filename,
                "detections": {}
            })

        # Save result image with bounding boxes
        result_image = results[0].plot()
        result_filename = f"crime_detected_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result_image)

        # Generate report
        detection_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detection_type": "Crime Detection",
            "filename": original_filename,
            "detections": detections
        }
        
        # Add to detection history
        detection_history.append(detection_report)
        
        # Create alert for crime detection
        alert_message = f"Crime detected: {', '.join(detections.keys())}"
        alert_level = "high"
        create_alert(alert_message, alert_level, "crime")
        
        return jsonify({
            "status": "success", 
            "image_url": f"/static/results/{result_filename}", 
            "original_filename": original_filename,
            "detections": detections,
            "report": detection_report,
            "alert": {
                "message": alert_message,
                "level": alert_level
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def crowd_detection_image(image_path, filename, original_filename):
    global crowd_data
    try:
        if crowd_model is None:
            raise Exception("Crowd model not loaded correctly!")

        results = crowd_model(image_path)
        confidence_threshold = 0.3
        
        # Count people
        people_count = sum(1 for r in results for box in r.boxes if float(box.conf) >= confidence_threshold)
        
        # Save data for trends
        timestamp = time.strftime("%H:%M:%S")
        crowd_data.append({"time": timestamp, "count": people_count})
        if len(crowd_data) > 100:  # Keep only last 100 data points
            crowd_data = crowd_data[-100:]
            
        # Generate trend graph
        graph_filename = generate_crowd_graph()

        # Draw count directly on the image
        result_image = results[0].plot()
        
        # Add people count text to image with a visible background
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"People Count: {people_count}"
        
        # Create a rectangle for text background
        (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)
        cv2.rectangle(result_image, (10, 10), (10 + text_width + 20, 10 + text_height + 20), (255, 255, 255), -1)
        
        # Add text
        cv2.putText(result_image, text, (20, 10 + text_height + 5), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        result_filename = f"crowd_detected_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result_image)

        # Generate crowd assessment
        crowd_assessment = "Normal"
        alert_level = "info"
        alert_message = None
        
        if people_count > CROWD_CRITICAL_THRESHOLD:
            crowd_assessment = "Very Crowded"
            alert_level = "high"
            alert_message = f"CRITICAL ALERT: Crowd density exceeds critical threshold ({people_count} people)"
        elif people_count > CROWD_WARNING_THRESHOLD:
            crowd_assessment = "Crowded"
            alert_level = "medium"
            alert_message = f"WARNING: Crowd density exceeds warning threshold ({people_count} people)"
        elif people_count > 10:
            crowd_assessment = "Moderate"
            
        # Create alert if threshold exceeded
        if alert_message:
            create_alert(alert_message, alert_level, "crowd")
        
        # Generate report
        crowd_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detection_type": "Crowd Detection",
            "filename": original_filename,
            "people_count": people_count,
            "assessment": crowd_assessment
        }
        
        # Add to detection history
        detection_history.append(crowd_report)

        response_data = {
            "status": "success",
            "image_url": f"/static/results/{result_filename}",
            "original_filename": original_filename,
            "people_count": people_count,
            "assessment": crowd_assessment,
            "graph_url": f"/static/graphs/{graph_filename}",
            "report": crowd_report
        }
        
        # Add alert info if applicable
        if alert_message:
            response_data["alert"] = {
                "message": alert_message,
                "level": alert_level
            }
            
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def crowd_detection_video(video_path, filename, original_filename):
    global crowd_data, video_analysis_results
    try:
        if crowd_model is None:
            raise Exception("Crowd model not loaded correctly!")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        sample_interval = 5  # Process every 5th frame
        
        total_people = 0
        frame_wise_counts = []
        max_count = 0
        timestamp_base = time.time()
        
        # Create output directory for processed frames
        frames_dir = os.path.join(RESULT_FOLDER, f"video_frames_{os.path.splitext(filename)[0]}")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Alert variables
        threshold_crossed = False
        alert_frames = []
        alert_counts = []
        max_alert_count = 0
        
        # Full frame analysis storage
        detailed_frame_analysis = []
        
        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % sample_interval != 0:
                continue
                
            results = crowd_model(frame)
            frame_people_count = sum(1 for r in results for box in r.boxes if float(box.conf) >= 0.3)
            
            frame_wise_counts.append(frame_people_count)
            total_people += frame_people_count
            max_count = max(max_count, frame_people_count)
            
            # Store detailed frame analysis
            frame_time = time.strftime("%H:%M:%S", time.localtime(timestamp_base + frame_count/fps))
            detailed_frame_analysis.append({
                "frame_number": frame_count,
                "timestamp": frame_time,
                "people_count": frame_people_count,
                "threshold_exceeded": frame_people_count > CROWD_WARNING_THRESHOLD,
                "critical_threshold_exceeded": frame_people_count > CROWD_CRITICAL_THRESHOLD
            })
            
            # Check if threshold is crossed
            if frame_people_count > CROWD_WARNING_THRESHOLD:
                threshold_crossed = True
                alert_frames.append(frame_count)
                alert_counts.append(frame_people_count)
                max_alert_count = max(max_alert_count, frame_people_count)
            
            # Generate timestamped data point
            timestamp = time.strftime("%H:%M:%S", time.localtime(timestamp_base + frame_count/fps))
            crowd_data.append({"time": timestamp, "count": frame_people_count})
            
            # Draw count directly on the frame
            result_image = results[0].plot()
            
            # Add people count text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"People Count: {frame_people_count}"
            
            # Choose color based on threshold
            text_color = (0, 0, 255)  # Red by default
            if frame_people_count > CROWD_CRITICAL_THRESHOLD:
                text_color = (0, 0, 255)  # Red for critical
                bg_color = (255, 200, 200)  # Light red background
            elif frame_people_count > CROWD_WARNING_THRESHOLD:
                text_color = (0, 0, 255)  # Red for warning
                bg_color = (255, 255, 200)  # Light yellow background
            else:
                text_color = (0, 128, 0)  # Green for normal
                bg_color = (200, 255, 200)  # Light green background
            
            # Create a rectangle for text background
            (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)
            cv2.rectangle(result_image, (10, 10), (10 + text_width + 20, 10 + text_height + 20), bg_color, -1)
            
            # Add text
            cv2.putText(result_image, text, (20, 10 + text_height + 5), font, 1, text_color, 2, cv2.LINE_AA)
            
            # Save frame
            frame_filename = f"frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(frames_dir, frame_filename), result_image)

        cap.release()
        
        # Keep only the last 100 data points
        if len(crowd_data) > 100:
            crowd_data = crowd_data[-100:]
            
        # Generate trend graph
        graph_filename = generate_crowd_graph()
        
        # Calculate average crowd size
        avg_people_count = total_people / len(frame_wise_counts) if frame_wise_counts else 0
        
        # Generate crowd assessment
        crowd_assessment = "Normal"
        alert_level = "info"
        alert_message = None
        
        if avg_people_count > CROWD_CRITICAL_THRESHOLD:
            crowd_assessment = "Very Crowded"
            alert_level = "high"
            alert_message = f"CRITICAL ALERT: Average crowd density exceeds critical threshold ({round(avg_people_count, 1)} people)"
        elif avg_people_count > CROWD_WARNING_THRESHOLD:
            crowd_assessment = "Crowded"
            alert_level = "medium"
            alert_message = f"WARNING: Average crowd density exceeds warning threshold ({round(avg_people_count, 1)} people)"
        elif threshold_crossed:
            crowd_assessment = "Periodically Crowded"
            alert_level = "medium"
            alert_message = f"WARNING: Crowd threshold crossed at {len(alert_frames)} points in the video (max: {max_alert_count} people)"
        elif avg_people_count > 10:
            crowd_assessment = "Moderate"
            
        # Create alert if threshold exceeded
        if alert_message:
            create_alert(alert_message, alert_level, "crowd")
        
        # Get paths of all processed frames
        all_frame_paths = []
        for frame_num in range(0, frame_count, sample_interval):
            frame_path = os.path.join(frames_dir, f"frame_{frame_num}.jpg")
            if os.path.exists(frame_path):
                all_frame_paths.append(f"/static/results/video_frames_{os.path.splitext(filename)[0]}/frame_{frame_num}.jpg")
        
        # Generate report
        video_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detection_type": "Video Crowd Analysis",
            "filename": original_filename,
            "duration_seconds": frame_count / fps if fps else 0,
            "avg_people_count": round(avg_people_count, 1),
            "max_people_count": max_count,
            "assessment": crowd_assessment,
            "frame_count": frame_count,
            "threshold_crossed": threshold_crossed,
            "alert_frames": alert_frames
        }
        
        # Add to detection history
        detection_history.append(video_report)

        # Store the complete analysis results for UI access
        analysis_id = f"video_{os.path.splitext(filename)[0]}"
        video_analysis_results[analysis_id] = {
            "analysis_id": analysis_id,
            "frame_wise_counts": frame_wise_counts,
            "detailed_frame_analysis": detailed_frame_analysis,
            "all_frame_paths": all_frame_paths,
            "processed_frames_dir": f"/static/results/video_frames_{os.path.splitext(filename)[0]}/",
            "sample_interval": sample_interval,
            "fps": fps,
            "frame_count": frame_count,
            "original_filename": original_filename,
            "avg_people_count": round(avg_people_count, 1),
            "max_people_count": max_count,
            "assessment": crowd_assessment
        }

        response_data = {
            "status": "success",
            "original_filename": original_filename,
            "avg_people_count": round(avg_people_count, 1),
            "max_people_count": max_count,
            "assessment": crowd_assessment,
            "sample_frames": all_frame_paths[:5],  # First 5 frames for preview
            "graph_url": f"/static/graphs/{graph_filename}",
            "report": video_report,
            "frames_count": len(frame_wise_counts),
            "processed_frames_dir": f"/static/results/video_frames_{os.path.splitext(filename)[0]}/",
            "frame_wise_counts": frame_wise_counts,  # Send all counts for display
            "analysis_id": analysis_id,  # Key to access full analysis
            "threshold_crossed": threshold_crossed,
            "alert_frames": alert_frames if alert_frames else []
        }
        
        # Add alert info if applicable
        if alert_message:
            response_data["alert"] = {
                "message": alert_message,
                "level": alert_level
            }
            
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def generate_crowd_graph():
    plt.figure(figsize=(10, 6))
    
    # Extract times and counts from crowd_data
    times = [data_point["time"] for data_point in crowd_data]
    counts = [data_point["count"] for data_point in crowd_data]
    
    # Plot with improved styling
    plt.plot(range(len(counts)), counts, marker="o", markersize=5, linestyle="-", linewidth=2, color="blue", label="People Count")
    
    # Add thresholds
    plt.axhline(y=CROWD_WARNING_THRESHOLD, color="orange", linestyle="--", linewidth=1.5, label=f"Warning Threshold ({CROWD_WARNING_THRESHOLD})")
    plt.axhline(y=CROWD_CRITICAL_THRESHOLD, color="red", linestyle="--", linewidth=1.5, label=f"Critical Threshold ({CROWD_CRITICAL_THRESHOLD})")
    
    # Color code regions
    plt.fill_between(range(len(counts)), CROWD_CRITICAL_THRESHOLD, max(counts)+5 if counts else CROWD_CRITICAL_THRESHOLD+10, color="red", alpha=0.1)
    plt.fill_between(range(len(counts)), CROWD_WARNING_THRESHOLD, CROWD_CRITICAL_THRESHOLD, color="yellow", alpha=0.1)
    plt.fill_between(range(len(counts)), 0, CROWD_WARNING_THRESHOLD, color="green", alpha=0.1)
    
    # Improve axes and labels
    plt.xlabel("Time Points", fontsize=12)
    plt.ylabel("People Count", fontsize=12)
    plt.title("Crowd Monitoring Trend", fontsize=14, fontweight="bold")
    
    # Show grid and set background
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().set_facecolor("#f8f9fa")
    
    # Improve x-axis ticks (show timestamps at intervals)
    num_ticks = min(10, len(times))
    if num_ticks > 0:
        tick_positions = np.linspace(0, len(times)-1, num_ticks, dtype=int)
        plt.xticks(tick_positions, [times[i] for i in tick_positions], rotation=45)
    
    # Add annotations for maximum value
    if counts:
        max_idx = counts.index(max(counts))
        plt.annotate(f"Max: {max(counts)}", 
                    xy=(max_idx, max(counts)),
                    xytext=(max_idx, max(counts)+5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=10,
                    fontweight='bold')
    
    # Add legend and make it more visible
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")
    
    # Tight layout and improved padding
    plt.tight_layout()
    
    # Generate unique filename
    graph_filename = f"crowd_graph_{int(time.time())}.png"
    plt.savefig(os.path.join(GRAPH_FOLDER, graph_filename), dpi=100, bbox_inches="tight")
    plt.close()
    
    return graph_filename

def create_alert(message, level, alert_type):
    """Create an alert and add it to the alert history"""
    global alert_history
    
    alert = {
        "id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        "level": level,
        "type": alert_type,
        "read": False
    }
    
    alert_history.append(alert)
    # Keep only the last 100 alerts
    if len(alert_history) > 100:
        alert_history = alert_history[-100:]
        
    return alert

def log_detection(mode, filename, result):
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "filename": filename,
        "result": result
    }
    
    log_file = os.path.join(LOG_FOLDER, "detection_log.json")
    
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
            
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Error logging detection: {e}")

@app.route("/alerts", methods=["GET"])
def get_alerts():
    unread_only = request.args.get('unread', 'false').lower() == 'true'
    alert_type = request.args.get('type', None)
    limit = request.args.get('limit', default=10, type=int)
    
    filtered_alerts = alert_history
    
    if unread_only:
        filtered_alerts = [alert for alert in filtered_alerts if not alert["read"]]
        
    if alert_type:
        filtered_alerts = [alert for alert in filtered_alerts if alert["type"] == alert_type]
    
    return jsonify({
        "status": "success",
        "alerts": filtered_alerts[-limit:]
    })

@app.route("/mark_alert_read", methods=["POST"])
def mark_alert_read():
    alert_id = request.json.get("alert_id")
    
    if not alert_id:
        return jsonify({"status": "error", "message": "No alert ID provided"})
    
    for alert in alert_history:
        if alert["id"] == alert_id:
            alert["read"] = True
            break
    
    return jsonify({"status": "success", "message": "Alert marked as read"})

@app.route("/history", methods=["GET"])
def get_history():
    limit = request.args.get('limit', default=10, type=int)
    mode = request.args.get('mode', default=None)
    
    filtered_history = detection_history
    if mode:
        filtered_history = [entry for entry in detection_history if entry.get("detection_type", "").lower().startswith(mode.lower())]
    
    return jsonify({
        "status": "success",
        "history": filtered_history[-limit:]
    })

@app.route("/clear_data", methods=["POST"])
def clear_data():
    global crowd_data, detection_history, alert_history, video_analysis_results
    crowd_data = []
    detection_history = []
    alert_history = []
    video_analysis_results = {}
    
    return jsonify({
        "status": "success",
        "message": "All detection data cleared successfully"
    })

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")

@app.route("/real_time_data", methods=["GET"])
def get_real_time_data():
    """Endpoint for real-time crowd data updates"""
    if not crowd_data:
        return jsonify({
            "status": "success",
            "count": 0,
            "trend": []
        })
        
    return jsonify({
        "status": "success",
        "count": crowd_data[-1]["count"] if crowd_data else 0,
        "trend": crowd_data[-10:]
    })

@app.route("/video_analysis/<analysis_id>", methods=["GET"])
def get_video_analysis(analysis_id):
    """Endpoint to get detailed video analysis data"""
    if analysis_id not in video_analysis_results:
        return jsonify({
            "status": "error",
            "message": "Analysis not found"
        })
        
    # Get frame range from query parameters
    start_frame = request.args.get('start_frame', default=0, type=int)
    end_frame = request.args.get('end_frame', default=None, type=int)
    
    analysis = video_analysis_results[analysis_id]
    
    # Filter frame data based on range
    detailed_frame_analysis = analysis["detailed_frame_analysis"]
    if end_frame is not None:
        detailed_frame_analysis = detailed_frame_analysis[start_frame:end_frame]
    elif start_frame > 0:
        detailed_frame_analysis = detailed_frame_analysis[start_frame:]
    
    # Get frame paths within the range
    frame_paths = []
    for frame_data in detailed_frame_analysis:
        frame_num = frame_data["frame_number"]
        frame_path = f"{analysis['processed_frames_dir']}frame_{frame_num}.jpg"
        if os.path.exists(os.path.join("static", frame_path.replace("/static/", ""))):
            frame_paths.append(frame_path)
    
    response_data = {
        "status": "success",
        "analysis_id": analysis_id,
        "original_filename": analysis["original_filename"],
        "avg_people_count": analysis["avg_people_count"],
        "max_people_count": analysis["max_people_count"],
        "assessment": analysis["assessment"],
        "frame_details": detailed_frame_analysis,
        "frame_paths": frame_paths,
        "total_frames": len(analysis["detailed_frame_analysis"])
    }
    
    return jsonify(response_data)

@app.route("/frame_analysis", methods=["GET"])
def get_frame_analysis():
    """Endpoint to get all available frame analysis data"""
    analysis_ids = list(video_analysis_results.keys())
    
    analyses = []
    for analysis_id in analysis_ids:
        analysis = video_analysis_results[analysis_id]
        analyses.append({
            "analysis_id": analysis_id,
            "original_filename": analysis["original_filename"],
            "avg_people_count": analysis["avg_people_count"],
            "max_people_count": analysis["max_people_count"],
            "assessment": analysis["assessment"],
            "frame_count": len(analysis["detailed_frame_analysis"]),
            "created_at": analysis["detailed_frame_analysis"][0]["timestamp"] if analysis["detailed_frame_analysis"] else ""
        })
    
    return jsonify({
        "status": "success",
        "analyses": analyses
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    # Calculate stats
    crime_detections = [entry for entry in detection_history if entry.get("detection_type") == "Crime Detection"]
    crowd_detections = [entry for entry in detection_history if entry.get("detection_type") in ["Crowd Detection", "Video Crowd Analysis"]]
    
    crime_count = len(crime_detections)
    crowd_count = len(crowd_detections)
    
    # Most common crime types
    crime_types = {}
    for entry in crime_detections:
        for crime_type, confidence in entry.get("detections", {}).items():
            if crime_type not in crime_types:
                crime_types[crime_type] = 0
            crime_types[crime_type] += 1
    
    # Average crowd size
    avg_crowd_size = 0
    if crowd_detections:
        total_people = sum(entry.get("people_count", entry.get("avg_people_count", 0)) for entry in crowd_detections)
        avg_crowd_size = total_people / len(crowd_detections)
    
    # Get crowd trend data
    crowd_trend = [{"time": entry["time"], "count": entry["count"]} for entry in crowd_data[-20:]]
    
    # Get unread alert count
    unread_alerts = len([alert for alert in alert_history if not alert["read"]])
    
    return jsonify({
        "status": "success",
        "total_detections": len(detection_history),
        "crime_detections": crime_count,
        "crowd_detections": crowd_count,
        "most_common_crimes": dict(sorted(crime_types.items(), key=lambda x: x[1], reverse=True)[:5]),
        "avg_crowd_size": round(avg_crowd_size, 1),
        "max_crowd_size": max([entry["count"] for entry in crowd_data]) if crowd_data else 0,
        "crowd_trend": crowd_trend,
        "unread_alerts": unread_alerts,
        "video_analyses_count": len(video_analysis_results)
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)