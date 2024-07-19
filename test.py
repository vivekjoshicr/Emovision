import cv2
from keras.models import model_from_json
import numpy as np
import os

current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

json_file_path = r"C:\Users\HP\project\Face_Emotion_Recognition_Machine_Learning-main\facialemotionmodel.json"
weights_file_path = r"C:\Users\HP\project\Face_Emotion_Recognition_Machine_Learning-main\facialemotionmodel.h5"
if not os.path.exists(json_file_path):
    print(f"Error: The file {json_file_path} does not exist in the current directory.")
    exit()

if not os.path.exists(weights_file_path):
    print(f"Error: The file {weights_file_path} does not exist in the current directory.")
    exit()
    
    
json_file = open(json_file_path, "r")
model_json = json_file.read()
json_file.close()


model = model_from_json(model_json)
model.load_weights(weights_file_path)

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:

    ret, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
          
            face_image = gray[q:q + s, p:p + r]

            face_image = cv2.resize(face_image, (48, 48))

            img = extract_features(face_image)
            pred = model.predict(img)

            prediction_label = labels[pred.argmax()]

            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)

            cv2.putText(im, prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        cv2.imshow("Emotion Detection", im)


        if cv2.waitKey(1) == 27:
            break

    except cv2.error as e:
        print(f"Error: {e}")
        pass

webcam.release()
cv2.destroyAllWindows()