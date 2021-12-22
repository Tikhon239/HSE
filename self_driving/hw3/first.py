from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2


class DontCrushDuckieTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']
        # getting the initial picture
        img, _, _, _ = env.step([0,0])
        
        condition = True
        while condition:
            img, reward, done, info = env.step([1, 0])

            height, width, _ = img.shape
            gray_img = cv2.cvtColor(img[height//2:], cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_img, 100, 1, 0)
            condition = thresh.mean() < 0.2

            env.render()
        env.step([0, 0])