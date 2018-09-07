import cv2

cap = cv2.VideoCapture(0)
first_frame = None

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    frameDelta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=5)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    for c in cnts:
        if cv2.contourArea(c) < 5000:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Mouvements", frame)
    cv2.imshow("Threshold", thresh)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        first_frame = gray
        print("Calibrate")
    if key == ord("q"):
        break

cv2.destroyAllWindows()
