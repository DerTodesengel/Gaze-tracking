import cv2
import dlib
import time
import numpy as np

class GazeTracker:
    def __init__(self, video_path, predictor_path, threshold=300):  # Установим порог в 300 пикселей
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.video_path = video_path
        self.threshold = threshold

        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.screen_center = (self.frame_width // 2, self.frame_height // 2)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # Получение FPS видео

        self.total_time_looking_at_screen = 0
        self.start_time = None
        self.is_looking = False

    def midpoint(self, p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    def get_eye_landmarks(self, landmarks, left=True):
        if left:
            return [landmarks.part(i) for i in range(36, 42)]
        else:
            return [landmarks.part(i) for i in range(42, 48)]

    def eye_center(self, eye_points):
        x = sum([p.x for p in eye_points]) / 6
        y = sum([p.y for p in eye_points]) / 6
        return int(x), int(y)

    def is_looking_at_screen(self, left_eye_center, right_eye_center):
        dist_left = np.linalg.norm(np.array(left_eye_center) - np.array(self.screen_center))
        dist_right = np.linalg.norm(np.array(right_eye_center) - np.array(self.screen_center))
        avg_dist = (dist_left + dist_right) / 2
        return avg_dist < self.threshold

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        eyes_detected = False

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            landmarks = self.predictor(gray, face)

            left_eye = self.get_eye_landmarks(landmarks, left=True)
            right_eye = self.get_eye_landmarks(landmarks, left=False)

            left_eye_center = self.eye_center(left_eye)
            right_eye_center = self.eye_center(right_eye)

            cv2.drawMarker(frame, left_eye_center, (0, 0, 255), cv2.MARKER_CROSS)
            cv2.drawMarker(frame, right_eye_center, (0, 0, 255), cv2.MARKER_CROSS)

            if self.is_looking_at_screen(left_eye_center, right_eye_center):
                eyes_detected = True

        return frame, eyes_detected

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, eyes_detected = self.process_frame(frame)
            cv2.imshow("Frame", frame)

            if eyes_detected:
                if not self.is_looking:
                    self.start_time = time.time()
                    self.is_looking = True
                # Увеличиваем общее время только если глаза распознаны
                self.total_time_looking_at_screen += 1 / self.fps
            else:
                self.is_looking = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        print(f"Total time looking at screen: {self.total_time_looking_at_screen:.2f} seconds")

if __name__ == "__main__":
    video_path = "your_video2.mp4"
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    gaze_tracker = GazeTracker(video_path, predictor_path)
    gaze_tracker.run()
