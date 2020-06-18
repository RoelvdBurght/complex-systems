import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
import PySimpleGUI as sg
import city_class_test as city_class
from city_class_test import Empty, Activity


class GuiCell(object):
    def __init__(self, graph, top_left, size, model_cell):
        self.graph = graph
        self.top_left = top_left
        self.bottom_right = (top_left[0] + size, top_left[1] + size)
        self.color_dict = {1:'Green', 2:'Red', 3:'Grey'}
        self.model_cell = model_cell
        self.black = False
        self.colored = False

    def draw(self):
        # Draws the rectangles on the screen in the right color. Extra checks are present as a speed up 
        if issubclass(type(self.model_cell), Activity) and self.colored == False:
            self.graph.DrawRectangle(self.top_left, self.bottom_right, fill_color=self.color_dict[self.model_cell.value])
            self.black = False
            self.colored = True

        elif type(self.model_cell) == Empty and self.black is False:
            self.graph.DrawRectangle(self.top_left, self.bottom_right, fill_color="Black")
            self.black = True
            self.colored = False

class GUI(object):
    def __init__(self, city):
        self.city = city
        self.height, self.width = 800, 800
        self.linecolor = "white"
        self.linewidth = 0.5
        self.cell_grid = np.empty((self.city.n, self.city.n), dtype='object')

        sg.theme('DarkAmber')   # Add a touch of color 
        layout = [
            [sg.Graph(canvas_size=(self.width, self.height), graph_bottom_left=(0,0), graph_top_right=(self.width, self.height), key='graph')],
            [sg.T('Actions'), sg.Button('Start sim'), sg.Input('# iterations', key='num_iter', size=(10, 1), focus=True), sg.Button('Single step'), sg.Button('Initialise cells')]      
            ]
        window = sg.Window('Graph test', layout)
        self.graph = window['graph']
        window.Finalize()

        # Draw borders
        self.graph.DrawLine((1, self.height-0.5), (self.width-0.1, self.height-0.5), color=self.linecolor, width=0.1)
        self.graph.DrawLine((1, self.height-0.1), (1, 0), color=self.linecolor, width=0.1)
        self.graph.DrawLine((self.width-0.1, self.height-0.1), (self.width-0.1, 0), color=self.linecolor, width=0.1)

        # Draw gridlines
        for x in range(self.city.n):
            self.graph.DrawLine((0, self.height/self.city.n * x), (self.width, self.height/self.city.n * x), color=self.linecolor, width=self.linewidth)
            self.graph.DrawLine((self.width/self.city.n * x, 0), (self.width/self.city.n * x, self.height), color=self.linecolor, width=self.linewidth)

        # Fill grid with GuiCell objects
        self.x_co = np.linspace(0, self.width, self.city.n + 1)
        self.y_co = np.linspace(0, self.height, self.city.n + 1)
        for x, model_row in zip(self.x_co[:-1], self.city.grid):
            for y, model_cell in zip(self.y_co[:-1], model_row):
                self.cell_grid[model_cell.pos[0], model_cell.pos[1]] = GuiCell(self.graph, (x, y), self.height/self.city.n, model_cell)

        while True:
            event, values = window.read()
            if event == None or event == 'Cancel': # if user closes window or clicks cancel
                break

            if event == 'Single step':
                self.update()

            if event == 'Start sim':
                for i in range(int(values['num_iter'])):
                    self.update()
                    window.Refresh()
                # pass

        window.close()

    def update(self):

        # Do one iteration of the model
        self.city.step()

        # Draws all cells on the grid
        for row, model_row in zip(self.cell_grid, self.city.grid):
            for cell, model_cell in zip(row, model_row):
                cell.draw()
                cell.model_cell = model_cell


    def one_step(self):
        # self.city.step()
        self.update()




if __name__ == "__main__":

    c = city_class.City(n=100)
    gui = GUI(c)
