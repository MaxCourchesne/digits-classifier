import pygame
import numpy as np
import random
import pickle as pkl
import sklearn
import joblib


# screen init
class WINDOW():
    num_squares_x = 28
    num_squares_y = 28
    square_size = float("inf")
    margin = 2
    grid_limits = [(0, 0), (0, 0)]
    def __init__(self, width, height, color):
        self.width = width
        self.height = height
        self.color = color

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.grid = [[0 for _ in range(WINDOW.num_squares_x)] for _ in range(WINDOW.num_squares_y)]

    def update(self):
        pygame.display.update()
    def fill(self, color):
        self.color = color
        self.screen.fill(self.color)
    def draw_grid(self):
        margin = WINDOW.margin
        square_size = max(int((self.width-((WINDOW.num_squares_x-1)*margin))/(WINDOW.num_squares_x+2)), int((self.height-((WINDOW.num_squares_y-1)*margin))/(WINDOW.num_squares_y+2)))
        WINDOW.square_size = square_size
        #square_pos_int = (square_size, square_size)
        square_pos_int = ((self.width-((WINDOW.num_squares_x-1)*margin + WINDOW.num_squares_x*square_size))/2, (self.height-((WINDOW.num_squares_y-1)*margin + WINDOW.num_squares_y*square_size))/2)
        WINDOW.grid_limits[0] = (square_pos_int[0], square_pos_int[0]+(WINDOW.num_squares_x-1)*WINDOW.margin+(WINDOW.num_squares_x)*WINDOW.square_size)
        WINDOW.grid_limits[1] = (square_pos_int[1], square_pos_int[1] + (WINDOW.num_squares_y - 1) * WINDOW.margin + (WINDOW.num_squares_y) * WINDOW.square_size)

        x_pos = square_pos_int[0]
        y_pos = square_pos_int[1]


        for row in self.grid:
            for square in row:
                # pygame.draw.rect(self.screen, (255-square, 255-square, 255-square), ((x_pos, y_pos), (square_size, square_size)))
                pygame.draw.rect(self.screen, (255-square/4, 255-square, 255-square/4), ((x_pos, y_pos), (square_size, square_size)))
                x_pos += margin+square_size
            x_pos = square_pos_int[0]
            y_pos += margin+square_size
    def draw_on_grid(self):
        x_pos, y_pos = pygame.mouse.get_pos()
        if WINDOW.grid_limits[0][0]+WINDOW.square_size+WINDOW.margin < x_pos < WINDOW.grid_limits[0][1]-WINDOW.square_size-WINDOW.margin and WINDOW.grid_limits[1][0]+WINDOW.square_size+WINDOW.margin < y_pos < WINDOW.grid_limits[1][1]-WINDOW.square_size-WINDOW.margin:
            x_coordinate, y_coordinate = int((x_pos-WINDOW.grid_limits[0][0])/(WINDOW.square_size+WINDOW.margin)), WINDOW.num_squares_y-1+int((y_pos-WINDOW.grid_limits[1][1])/(WINDOW.square_size+WINDOW.margin))
            self.grid[y_coordinate][x_coordinate] = 253
            if self.grid[y_coordinate+1][x_coordinate] == 0:
                self.grid[y_coordinate+1][x_coordinate] = random.random() * 0 + 170
                pass
            if self.grid[y_coordinate - 1][x_coordinate] == 0:
                self.grid[y_coordinate - 1][x_coordinate] = random.random() * 0 + 170
                pass
            if self.grid[y_coordinate][x_coordinate+1] == 0:
                self.grid[y_coordinate][x_coordinate+1] = random.random() * 0 + 170
                pass
            if self.grid[y_coordinate][x_coordinate-1] == 0:
                self.grid[y_coordinate][x_coordinate-1] = random.random() * 0 + 170
                pass
    def print(self, array):
        # WIN = WINDOW(800, 800, (100, 100, 100))
        self.grid = array
        self.fill((120, 70, 120))
        self.draw_grid()
        self.update()
        running = True
        while running:
            if detectClose():
                running = False


pygame.init()
WIN = WINDOW(800, 800, (100, 100, 100))

# functions
def detectClose():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True

def mouseIsPressed():
    keys = pygame.mouse.get_pressed()
    if keys[0]:
        return True

# main loop
def main(WIN):
    running = True
    while running:
        if detectClose():
            running = False
        if mouseIsPressed():
            WIN.draw_on_grid()

        WIN.fill((120, 70, 120))
        WIN.draw_grid()
        WIN.update()



main(WIN)
image = list(np.array(WIN.grid).reshape(-1))
# print(image)



# ML model's predictions
forest_0_clf = joblib.load("0_detector_forests.pkl")
forest_1_clf = joblib.load("1_detector_forests.pkl")
forest_2_clf = joblib.load("2_detector_forests.pkl")
forest_3_clf = joblib.load("3_detector_forests.pkl")
forest_4_clf = joblib.load("4_detector_forests.pkl")
forest_5_clf = joblib.load("5_detector_forests.pkl")
forest_6_clf = joblib.load("6_detector_forests.pkl")
forest_7_clf = joblib.load("7_detector_forests.pkl")
forest_8_clf = joblib.load("8_detector_forests.pkl")
forest_9_clf = joblib.load("9_detector_forests.pkl")

probs_0 = forest_0_clf.predict_proba((np.array(image).reshape(1, -1)))
probs_1 = forest_1_clf.predict_proba((np.array(image).reshape(1, -1)))
probs_2 = forest_2_clf.predict_proba((np.array(image).reshape(1, -1)))
probs_3 = forest_3_clf.predict_proba((np.array(image).reshape(1, -1)))
probs_4 = forest_4_clf.predict_proba((np.array(image).reshape(1, -1)))
probs_5 = forest_5_clf.predict_proba((np.array(image).reshape(1, -1)))
probs_6 = forest_6_clf.predict_proba((np.array(image).reshape(1, -1)))
probs_7 = forest_7_clf.predict_proba((np.array(image).reshape(1, -1)))
probs_8 = forest_8_clf.predict_proba((np.array(image).reshape(1, -1)))
probs_9 = forest_9_clf.predict_proba((np.array(image).reshape(1, -1)))

# probs_to_number = {"probs_1": "1", "probs_2": "2", "probs_3": "3", "probs_4": "4", "probs_5": "5", "probs_6": "6", "probs_7": "7", "probs_8": "8", "probs_9": "9"}
probs = [probs_0, probs_1, probs_2, probs_3, probs_4, probs_5, probs_6, probs_7, probs_8, probs_9]

highestProb = 0
highestProbNum = "IDK"
index = 0
for prob in probs:
    #print("loop")
    if prob[0, 1] >= highestProb:
        highestProb = prob[0][1]
        highestProbNum = index
    print(f"probs {index} = {prob}")
    index += 1

print(f"\n\n\n\nI think this is a {highestProbNum}")



"""WIN.print([[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 253.,
        253., 253., 253., 253., 253., 253., 253., 253.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253., 253.,
        253., 253., 253., 253., 253., 253., 253., 253., 253.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0., 253., 253., 253., 253., 253., 253.,
        253., 253., 253.,   0., 253.,   0., 253., 253., 253., 253.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0., 253., 253., 253., 253.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253., 253.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253., 253.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0., 253., 253., 253., 253.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0., 253., 253., 253., 253., 253.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0., 253., 253., 253., 253., 253.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
        253., 253., 253., 253., 253., 253., 253., 253.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253.,
        253., 253., 253., 253., 253., 253., 253., 253., 253.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253.,
        253., 253., 253.,   0.,   0.,   0., 253., 253., 253.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253.,
          0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253., 253.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0., 253., 253., 253., 253.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253., 253.,
          0., 253., 253., 253., 253., 253., 253., 253.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253., 253.,
        253., 253., 253., 253., 253., 253., 253.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 253., 253.,
        253., 253., 253., 253., 253.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.]])
"""

