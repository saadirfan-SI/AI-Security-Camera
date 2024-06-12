import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
detection = False

# When a person leaves the frame then the algorithm will stop the video
detection_stopped_time = None

timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

# Frame size of the video needs to be the same as the video capture size
frame_size = (int(cap.get(3)), int(cap.get(4)))

# Now it is needed to set a 4 character code to set the video format. The '*' decomposes it into 4 arguments
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = None  # Initialize out variable

while True:
    _, frame = cap.read()

    # This is to convert frame into a greyscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 'Gray' is the image, '1.3' is the scale factor which determines the accuracy and speed of the algorithm
    # And '5' is the minimum number of neighbours. The higher the number is the less number of faces detected
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    # Logic: If a body or face is available
    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

            # This will make an output stream, 20 is the frame rate
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started Recording!")

    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print('Stop Recording!')
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    # Draw rectangles around faces
    for (x, y, width, height) in faces:
        # Where frame is the image you want to draw, '255' will draw a blue rectangle.
        # Where '3' is line thickness
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

    cv2.imshow("Input", frame)

    # This is to make sure that the loop ends
    if cv2.waitKey(1) == ord('e'):
        break

# This is to release the resource
if out is not None:
    out.release()

cap.release()
cv2.destroyAllWindows()
