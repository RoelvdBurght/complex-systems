import numpy as np
import PySimpleGUI as sg
import city_model as city_class
import pyscreenshot as ImageGrab



class GuiCell(object):
    # Object which stores the type of activity in the gui grid. Used for visualisation of the activities
    def __init__(self, graph, top_left, size, model_cell):
        self.graph = graph
        self.size = size
        self.top_left = top_left
        self.bottom_right = (top_left[0] + size, top_left[1] + size)
        self.color_dict = {0: 'Black', 1:'Dark green', 2:'Yellow', 3:'Blue', 4:'Grey', 5:'Grey'}
        self.model_cell = model_cell
        self.black = False
        self.colored = False

    def draw(self):
        # Draws the activities on the screen in the right color.
        if self.model_cell.value == 0: # and self.black is False:
            self.graph.DrawCircle((self.top_left[0]-1/2*self.size, self.top_left[1]-1/2*self.size), self.size/2,
                                  fill_color=self.color_dict[self.model_cell.value])
        else:
            self.graph.DrawCircle((self.top_left[0] - 1 / 2 * self.size, self.top_left[1] - 1 / 2 * self.size),
                                  self.size / 2, fill_color=self.color_dict[self.model_cell.value])



class GUI(object):
    # Gui class. Creates and shows the GUI, handles user input and updates the model.
    def __init__(self, city):
        self.city = city
        self.height, self.width = 800, 800
        self.linecolor = "white"
        self.linewidth = 0.5
        self.cell_grid = np.empty((self.city.n, self.city.n), dtype='object')
        self.screenshot_counter = 1

        sg.theme('DarkAmber')   # Add a touch of color 
        layout = [
            [sg.Graph(canvas_size=(self.width, self.height), graph_bottom_left=(0,0), graph_top_right=(self.width, self.height), key='graph')],
            [sg.T('Actions'), sg.Button('Multiple steps'), sg.Text('Number of iterations to run:'), sg.Input('25', key='num_iter', size=(10, 1), focus=True), sg.Button('Single step')]
            ]
        self.window = sg.Window('Graph test', layout)
        self.graph = self.window['graph']
        self.window.Finalize()

        # Fill grid with GuiCell objects
        self.x_co = np.linspace(0, self.width, self.city.n + 1)
        self.y_co = np.linspace(0, self.height, self.city.n + 1)
        for x, model_row in zip(self.x_co[:-1], self.city.grid):
            for y, model_cell in zip(self.y_co[:-1], model_row):
                self.cell_grid[model_cell.pos[0], model_cell.pos[1]] = GuiCell(self.graph, (x, y), self.height/self.city.n, model_cell)
        self.window.Refresh()
        # self.run_and_save(1000, 25)

        while True:
            event, values = self.window.read()
            if event is None or event == 'Cancel': # if user closes window or clicks cancel
                break
            if event == 'Single step':
                self.city.step()
                self.update()

            if event == 'Multiple steps':
                self.window.Refresh()
                for i in range(int(values['num_iter'])):
                    self.city.step()
                self.update()
                self.window.Refresh()
                # self.screenshot()

        self.window.close()

    def run_and_save(self, iterations, step_size):
        # Runs model for given number of iterations, taking and saving a screenshot at given intervals (step_size)
        for i in range(int(iterations/step_size)):
            print('iterations: {} to {} out of {}'.format(step_size*i, step_size*(i+1),
                                                          iterations))
            for steps in range(step_size):
                self.city.step()
            self.update()
            self.window.Refresh()
            # self.screenshot()

    def screenshot(self):
        # Create and save screenshot of the gui
        w, h = self.window.size
        upper_left = self.window.current_location()
        bottom_right = (upper_left[0] + w, upper_left[1] + h)
        image = ImageGrab.grab(bbox=(upper_left[0], upper_left[1], bottom_right[0], bottom_right[1]))
        image.save('screenshots/{}.png'.format(self.screenshot_counter))
        self.screenshot_counter += 1

    def update(self):
        # Draws all cells on the grid
        for row, model_row in zip(self.cell_grid, self.city.grid):
            for cell, model_cell in zip(row, model_row):
                cell.draw()
                cell.model_cell = model_cell

    def one_step(self):
        # Perform one iteration of the model and update gui
        self.update()
        self.screenshot()

if __name__ == "__main__":

    c = city_class.City(n=100)
    gui = GUI(c)

