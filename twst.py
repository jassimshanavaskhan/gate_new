from flask import Flask, render_template, Response, jsonify
import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
from scipy.spatial import distance as dist

# app = Flask(__name__)
app = Flask(__name__, static_folder='static')


class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] < D.shape[1]:
                for col in unusedCols:
                    self.register(inputCentroids[col])
            else:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

        return self.objects

total_persons = 0

def generate_frames():
    model_weights = 'MobileNetSSD_deploy.caffemodel'
    model_config = 'MobileNetSSD_deploy.prototxt'
    net = cv2.dnn.readNetFromCaffe(model_config, model_weights)

    tracker = CentroidTracker(maxDisappeared=20)

    cap = cv2.VideoCapture('vid2.mp4')

    crossed_persons = set()

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        cv2.line(frame, (0, 150), (W, 150), (255, 0, 0), 2)
        cv2.line(frame, (0, 240), (W, 240), (255, 0, 0), 2)

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.1:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append((startX, startY, endX, endY))

        objects = tracker.update(boxes)

        for (objectID, centroid) in objects.items():
            text = "Person {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            if centroid[1] >= 150 and centroid[1] <= 240 and objectID not in crossed_persons:
                global total_persons
                total_persons += 1
                crossed_persons.add(objectID)

        cv2.putText(frame, "Total Persons: {}".format(total_persons), (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html', total_persons=total_persons)


@app.route('/get_total_persons')
def get_total_persons():
    return jsonify({'total_persons': total_persons})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
