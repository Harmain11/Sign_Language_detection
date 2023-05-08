from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from function import *
from keras.utils import to_categorical
from keras.models import model_from_json

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

app = Flask(__name__)

cap = cv2.VideoCapture(0)

def gen_frames():
        # 1. New detection variables
    sequence = []
    sentence = []
    accuracy=[]
    predictions = []
    threshold = 0.8 

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break
            cropframe = frame[40:400, 0:300]
            frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
            image, results = mediapipe_detection(cropframe, hands)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-14:]
            try:
                if len(sequence) == 14:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    accuracy.append(str(res[np.argmax(res)]*100))
                            else:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)]*100))
                    if len(sentence) > 1:
                        sentence = sentence[-1:]
                        accuracy = accuracy[-1:]
                cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)
                cv2.putText(frame,"Output: -"+' '.join(sentence)+''.join(accuracy), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
