from django.shortcuts import render
import cv2
import joblib

# load model
model = joblib.load('../gesture_model.pkl')

def index(request):
    return render(request, 'index.html')

def start_camera(request):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(gray, (42, 42))

        img = img.flatten().reshape(1, -1)

        prediction = model.predict(img)

        print("Prediction:", prediction)

        cv2.putText(frame, str(prediction[0]), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'index.html')
