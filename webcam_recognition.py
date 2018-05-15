import cv2
import data_utils as d
from face_recognition import get_face


class DetectWindow:
    align = d.Align()
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    def start_window(self):
        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            print('{cap.get(3)}, {cap.get(4)}')
            rs = 8
            frame1 = cv2.resize(frame, (int(cap.get(3)/rs), int(cap.get(4)/rs)))
            faces = [self.align.align(frame1)]

            # Draw a rectangle around the faces
            # for rect in faces:
            # cv2.rectangle(frame, (rect.left() * rs, rect.top() * rs), (rect.right() * rs, rect.bottom() * rs),
            #               (255, 255, 255), 2)
            print(get_face(d.get_embeddings(faces)))

            # Display the resulting frame
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


det = DetectWindow()
det.start_window()
