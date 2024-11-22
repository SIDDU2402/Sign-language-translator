from flask import Flask, render_template, Response, jsonify
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('final_sign_language_model_v3.h5')

# Dynamically fetch the class labels (update this based on your actual model)
labels = {0: 'Done', 1: 'HowareYou', 2: 'Hungry', 3: 'MakeFavour', 4: 'Notwell', 5: 'Thanks'}

cap = None  

def predict_sign(frame):
    """Preprocess the frame and predict the class."""
    img = cv2.resize(frame, (128, 128))  
    img = np.expand_dims(img, axis=0) / 255.0  
    predictions = model.predict(img)  
    class_idx = np.argmax(predictions) 
    return labels[class_idx]  

def gen_frames():
    """Generate frames from the webcam."""
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)  

    while True:
        if cap is None or not cap.isOpened():
            break  

        success, frame = cap.read()
        if not success:
            break

        
        sign = predict_sign(frame)
        cv2.putText(frame, sign, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/start_video')
def start_video():
    """Start the video feed."""
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)  
        if not cap.isOpened():
            return jsonify({"status": "error", "message": "Failed to start video capture"}), 500
    return jsonify({"status": "started"})

@app.route('/stop_video')
def stop_video():
    """Stop the video feed."""
    global cap
    if cap and cap.isOpened():
        cap.release()
        cap = None
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    """Stream video frames to the client."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
