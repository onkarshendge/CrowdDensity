import cv2
import numpy as np
import pafy

BLOB_SCALE_FACTOR = 0.00932
USE_YOUTUBE = True

if __name__ == '__main__':
    net = cv2.dnn.readNet("cfg\yolov3.weights", "cfg\yolov3.cfg")
    classes = []
    with open("cfg\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

#mRe-514tGMg
    if USE_YOUTUBE:
        url = 'https://youtu.be/mRe-514tGMg'
        video = pafy.new(url)
        # for s in video.streams:
        #     print(s.resolution, s.url)
        best = video.streams[3]
        print(best.url)
        # best = video.getbest()
        # print(best.resolution)
        camera = cv2.VideoCapture(best.url)
    else:
        camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        frame = cv2.resize(frame, None, fx=1, fy=1)
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, BLOB_SCALE_FACTOR, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        confidences = []
        boxes = []
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    boxes.append([center_x, center_y, 5, 5, str(classes[class_id])])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h, label = boxes[i]
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
                cv2.putText(frame, label, (x, y-30), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        cv2.imshow("Image", frame)

        if cv2.waitKey(40) == 27:
            break

    cv2.destroyAllWindows()
