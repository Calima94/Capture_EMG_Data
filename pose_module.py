import cv2
import mediapipe as mp
import math
from collections import Counter


class PoseDetector:  # Class to detect people

    # Pose parameters
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, teste=2, var=10):
        self.lmList = []
        self.results = None
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)
        self.teste = teste
        self.var = var

    def find_pose(self, img, draw=True):
        """
        Finds the body of the person and return the processed images
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def find_position(self, img, draw=True):
        """
        Find the x and y values of each landmark point and put it in a list
        """
        self.lmList = []
        if self.results.pose_landmarks:
            for id_, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # Calc the pixel value in each point
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id_, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
        return self.lmList

    @staticmethod
    def draw_in_the_screen(img, x1, y1, x2, y2, x3, y3, angle, correct):
        if correct:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (0, 0, 255), 3)
            # Draw 2(two) circles between the selected landmarks in orange
            cv2.circle(img, (x1, y1), 4, (50, 200, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 8, (50, 200, 255), 2)
            cv2.circle(img, (x2, y2), 4, (50, 200, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (50, 200, 255), 2)
            cv2.circle(img, (x3, y3), 4, (50, 200, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 8, (50, 200, 255), 2)
            cv2.putText(img, str(int(angle)), (500, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
        else:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.line(img, (x3, y3), (x2, y2), (0, 255, 0), 3)
            # Draw 2(two) circles between the selected landmarks in greeen
            cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 8, (255, 0, 0), 2)
            cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 0), 2)
            cv2.circle(img, (x3, y3), 4, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 8, (255, 0, 0), 2)
            cv2.putText(img, str(int(angle)), (500, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)

    @staticmethod
    def test_cases_to_use(tests_init, angle, var, angles_to_test, categories_already_trained):
        """
        Test if the angle that is being measuring is valid to write At that moment in the csv file, if its true return
        "True" and its category,
        Otherwise return false and zero
        """
        for i in range(tests_init):
            if i not in categories_already_trained:
                if angles_to_test[i] + var > angle > angles_to_test[i] - var:
                    return True, i + 1
        return False, 0

    @staticmethod
    def check_if_is_correct_angle(test_init, list_of_angles, angle, var, categories_already_trained):
        """
        Check if the angle measure is in the list_of_angles +- (more or less) the variation and if its already complete
        the number of samples defined. If its True the function returns "True" else "False"
        """
        for i in range(test_init):
            if i not in categories_already_trained:
                if list_of_angles[i] + var > angle > list_of_angles[i] - var:
                    return True
        return False

        for i in range(test):
            if list_of_angles[i] - var < angle < list_of_angles[i] + var:
                return True
        return False

    def draw_circles_and_lines(self, img, x1, y1, x2, y2, x3, y3, test, list_of_angles, angle, var,
                               categories_already_trained):
        """
        Check if the angle is correct to write in database and draw the points and lines in each case,
        "Green" if it's not correct and orange if it's correct.
        """
        correct = self.check_if_is_correct_angle(test_init=test, list_of_angles=list_of_angles, angle=angle, var=var,
                                                 categories_already_trained=categories_already_trained)
        self.draw_in_the_screen(img=img, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, angle=angle, correct=correct)

    def find_angle(self, img, p1, p2, p3, test, var, list_of_angles, categories_already_trained, draw=True):
        """
        Find the angle of the Elbow using the respective landmarks and return the angle found
        """
        # Add the values of the markers points in the variables (x1, x2 and x3) and (y1, y2 and y3)
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle in degrees
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        # Transform the angle in positive angle if it's negative
        if angle < 0:
            angle += 360
        # Draw the circles and the lines respectively
        if draw:
            self.draw_circles_and_lines(img=img, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, test=test,
                                        list_of_angles=list_of_angles, angle=angle, var=var,
                                        categories_already_trained=categories_already_trained, )
        return angle

    @staticmethod
    def check_if_num_samples_is_complete(emg_table, list_of_angles, num_of_samples_in_each_class, test,
                                         categories_already_trained):
        """
        Check if the angles in a category is complete, if True subtract the number of the tests and add the category
        that have already been trained
        """
        emg_mod = [i[-1] for i in emg_table]
        emg_mod.sort()
        values = Counter(emg_mod).values()
        for i, j in enumerate(values):
            if i not in categories_already_trained:
                if j >= num_of_samples_in_each_class:
                    categories_already_trained.append(i)
                    test -= 1
                    print(f"Finished the category {list_of_angles[i]}")
        return test, categories_already_trained
