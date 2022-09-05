import pygame
import numpy as np
import random
import pandas as pd
import pickle as pkl
import sklearn
from sklearn.datasets import fetch_openml
import joblib
import time

# variables
appending_to_training = False

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
                self.grid[y_coordinate+1][x_coordinate] = random.random() * 0 + 253
                pass
            if self.grid[y_coordinate - 1][x_coordinate] == 0:
                #self.grid[y_coordinate - 1][x_coordinate] = random.random() * 0 + 253
                pass
            if self.grid[y_coordinate][x_coordinate+1] == 0:
                self.grid[y_coordinate][x_coordinate+1] = random.random() * 0 + 253
                pass
            if self.grid[y_coordinate][x_coordinate-1] == 0:
                #self.grid[y_coordinate][x_coordinate-1] = random.random() * 0 + 253
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
#print(image)

'''forest_clf_all = joblib.load("forest_clf_all.pkl")
print(forest_clf_all.predict((np.array(image).reshape(1, -1))))

svc_clf_all = joblib.load("svc_clf_all.pkl")
print(svc_clf_all.predict((np.array(image).reshape(1, -1))))'''

oneVsOne_clf_all = joblib.load("oneVsOne_clf_all.pkl")
guessed_num = oneVsOne_clf_all.predict((np.array(image).reshape(1, -1)))[0]
#print(image)
print(f"I think this is a {guessed_num}")


if appending_to_training:
    guess_is_right = ""
    while guess_is_right != "y" and guess_is_right != "n":
        guess_is_right = input(f"was it a {guessed_num}? (y/n): ")


    if guess_is_right == "y":
        drawn_num = guessed_num
    elif guess_is_right == "n":
        drawn_num = "poulet"
        while drawn_num not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            drawn_num = input(f"The number was acctually a: ")


    print(f"drawn_num = {drawn_num}")
    df_X = pd.read_csv("training_from drawing_X")
    df_y = pd.read_csv("training_from drawing_y")
    print(df_X.iloc[0])
    print(df_y.iloc[0])

    to_append_X = pd.DataFrame({"data": image})
    to_append_y = pd.DataFrame({"target": str(drawn_num)}, index=[len(df_y)])
    print(f"y index = {len(df_y)}")

    df_X.append(to_append_X)
    df_y.append(to_append_y)

    df_X.to_csv("training_from drawing_X", index=False)
    df_y.to_csv("training_from drawing_y", index=False)

    '''
    except Exception as e:
        print("didn't append")
        print(e)
        pass
        mnist = fetch_openml("mnist_784", version=1)
        X, y = mnist["data"], mnist["target"]
        df_X = pd.DataFrame(X)
        df_y = pd.DataFrame(y)
        df_X.to_csv("training_from drawing_X")
        df_y.to_csv("training_from drawing_y")
        print(f"df_X = {df_X}")
        print(f"df_y = {df_y}")
    '''

time.sleep(5)