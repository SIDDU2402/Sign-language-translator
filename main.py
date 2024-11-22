import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify
import threading


app = Flask(__name__)


model = tf.keras.models.load_model('final_sign_language_model_v3.h5')


labels = {0: 'Done', 1: 'HowareYou', 2: 'Hungry', 3: 'MakeFavour', 4: 'Notwell', 5: 'Thanks'}

cap = None  
metrics = {
    "accuracy": [],
    "loss": []
}

def predict_sign(frame):
    """Preprocess the frame and predict the class."""
    img = cv2.resize(frame, (128, 128))  
    img = np.expand_dims(img, axis=0) / 255.0  
    predictions = model.predict(img)  
    class_idx = np.argmax(predictions)
    return labels[class_idx], predictions[0][class_idx]

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

        sign, confidence = predict_sign(frame)
        metrics["accuracy"].append(confidence)
        metrics["loss"].append(1 - confidence)  # Fake loss for illustration

        cv2.putText(frame, f'{sign} ({confidence*100:.2f}%)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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

def plot_metrics():
    """Create and update plots for accuracy and loss."""
    root = tk.Tk()
    root.title("Sign Language Model Metrics")

    def update_plots():
        """Update the graphs with the latest accuracy and loss data."""
        if metrics["accuracy"] and metrics["loss"]:
            ax1.clear()
            ax2.clear()
            ax1.plot(metrics["accuracy"], label="Accuracy", color="green")
            ax2.plot(metrics["loss"], label="Loss", color="red")
            ax1.set_title("Accuracy Over Time")
            ax2.set_title("Loss Over Time")
            ax1.set_xlabel("Frames")
            ax2.set_xlabel("Frames")
            ax1.set_ylabel("Accuracy")
            ax2.set_ylabel("Loss")
            ax1.legend()
            ax2.legend()
            canvas.draw()
        
        root.after(1000, update_plots)  # Update plots every 1 second

   
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    update_plots()

    root.mainloop()

def start_flask():
    """Run Flask app in a separate thread."""
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.start()

    
    plot_metrics()
