from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2


class DontCrushDuckieTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

        self.env = generated_task['env']
        self.side = 1

    def go(self, speed=1):
        img, _, _, _ = self.env.step([speed, 0])
        return img

    def rotate_45(self, side=1):
        img, _, _, _ = self.env.step([0, side * 30])
        return img

    def rotate_90(self, side=1):
        self.rotate_45(side)
        img = self.rotate_45(side)
        return img

    @staticmethod
    def is_obstacle(img, threshold):
        # height, width, _ = img.shape
        # gray_img = cv2.cvtColor(img[height//2:], cv2.COLOR_RGB2GRAY)
        # ret, thresh = cv2.threshold(gray_img, 100, 1, 0)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        thresh = np.ones(hsv.shape[:2])
        thresh[np.logical_or(hsv[:, :, 0] < 20, hsv[:, :, 0] > 30)] = 0
        thresh[hsv[:, :, 1] < 200] = 0

        return thresh.mean() > threshold

    def go_forward(self, threshold, max_steps=np.inf):
        img = self.go(0)
        steps = 0

        while not self.is_obstacle(img, threshold) and steps < max_steps:
            img = self.go()
            steps += 1

    def change_side(self):
        self.rotate_90(self.side)

        for i in range(4):
            self.go()

        self.side *= -1
        self.rotate_90(self.side)

    def try_change_side(self, threshold):
        flag = True
        while flag:
            img = self.rotate_45(self.side)
            flag = self.is_obstacle(img, threshold)
            self.rotate_45(-self.side)

            self.go()

    def solve(self):
        self.go_forward(threshold=0.04)
        self.change_side()
        self.try_change_side(threshold=0.001)
        for i in range(5):
            self.go()
        self.change_side()
        self.go_forward(threshold=0.04, max_steps=5)
    
        self.env.step([0, 0])
