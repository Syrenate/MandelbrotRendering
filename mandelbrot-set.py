import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import time
import math


class ComplexNumber:
    """Class to represent a single complex number x + iy, where i^2 = -1."""
    def __init__(self, x : float, y : float):
        self.x = x
        self.y = y

    def absolute_square(self):
        """Returns squared values of called ComplexNumber."""
        return ComplexNumber(self.x ** 2, self.y ** 2)

    def __str__(self):
        return f"{round(self.x, 3)} + i * {round(self.y, 3)}"

    def __add__(c1, c2):
        """Method to add two instanced of given instances of ComplexNumber, c1 and c2."""
        return ComplexNumber(c1.x + c2.x, c1.y + c2.y)
    
class ComplexPlane:
    """Class to represent a plane of complex numbers, with a real/imaginary axis represented by the x/y axis respectively."""
    def __init__(self, dim_x, dim_y, x_min, y_min, x_max, y_max, step_sizes, transform, iteration_max, is_generated, file_contents):
        """Constructor, initialises the plane over each pixel, each of which stores the transform value
                        under the given transform function."""
        self.iteration_max = iteration_max                   
        # Initialise array used for Histogram colouring; a colouring technique which distributes colour based on the number pixels that reach a certain intensity.
        # To do this, the array iterations_per_pixel stores the number of pixels that reach each iteration count (represented by the index of iterations_per_pixel).
        self.iterations_per_pixel = np.zeros(iteration_max)  
        self.metadata = (dim_x, dim_y, x_min, y_min, x_max, y_max)
        self.construction_data = (is_generated, file_contents)
        self.step_sizes = step_sizes

        # The array grid will be a 2D array of each pixel, storing the transform of the coordinate corresponding to the index, from the top right of the image. 
        self.grid = []
        self.escape_time_algorithm(transform)

    def escape_time_algorithm(self, transform):
        """Method implementing the simplest plotting algorithm; one which iterates through each pixel on the image to evaluate the iterations taken to grow to infinity.
           For loops scan each pixel of a specified image size, evaluating the escape time of each pixel's corresponding coordinate in the real-imaginary plane."""
        dim_x, dim_y = self.metadata[:2]
        x_min, y_min = self.metadata[2:4]
        x_max, y_max = self.metadata[4:]
        is_generated, file_contents = self.construction_data

        step_sizes = get_step_size(x_min, x_max, y_min, y_max)
        print(step_sizes)
        print(x_min, x_max, y_min, y_max)

        prev_perc = 0 # For displaying completion percentage.
        start_time = time.time()
        for j in range(dim_y): # This loop scans all pixels 
            row = []
            for i in range(dim_x):
                iterations = 0
                if is_generated: # If this config has been found in plane_data.txt, just read from the contents.
                    iterations = file_contents[j][i]
                else:
                    new_num = ComplexNumber(i * step_sizes[0] + x_min, - j * step_sizes[1] + y_max)
                    iterations = transform(new_num, self.iteration_max)
                # Here, if iterations reaches the iteration_max, then we colour differently to easily show the 'border' of the set.
                if iterations != self.iteration_max: self.iterations_per_pixel[iterations - 1] += 1 
                row.append(iterations)
            self.grid.append(row)

            if not is_generated: # For displaying the percentage completion of set generation.
                percent_skip = 25 # Display percentages every percent_skip (please use factors of 100 :) ).
                percentage = round(100 * j / dim_y)
                if round(100 * j / dim_y) != prev_perc and percentage % percent_skip == 0 and percentage != 100:
                    print(f"Generation is {round(100 * j / dim_y)}% complete in {round(time.time() - start_time, 3)}s...")
                    prev_perc = round(100 * j / dim_y)

    def rectangle_checking_algorithm(self, transform):
        """Method implementing a rectangle checking, which iterates through perimeters rather than a large area.
           By dividing the grid into four and filling in if all border pixels share the same colour (otherwise subdividing again), 
               we use the fact that the Mandelbrot set is full to only check the borders."""
        dim_x, dim_y = self.metadata[:2]
        x_min, y_min = self.metadata[2:4]
        x_max, y_max = self.metadata[4:]
        is_generated, file_contents = self.construction_data
        self.grid = np.full((dim_y, dim_x), fill_value=0)

        def split_grid(_grid, top_left_pixel):
            """Method that returns four subgrids of a grid."""
            _dim_x, _dim_y = len(_grid[0]) if len(_grid) > 0 else 0, len(_grid)
            if _dim_y <= 1 or _dim_x <= 1: return _grid

            _half_index_x = round(_dim_x / 2)
            _half_index_y = round(_dim_y / 2)
            _grid_div_1, _grid_div_2 = _grid[:_half_index_y], _grid[_half_index_y:] # Splits grid horizontally into upper and lower subgrids.
            _grid_1, _grid_2, _grid_3, _grid_4 = [], [], [], []
            for j in range(_dim_y): # Loop that splits each row of _grid_div in half to create 4 subgrids.
                if j < _half_index_y: 
                    _grid_1.append(_grid_div_1[j][:_half_index_x]) # Top left subgrid
                    _grid_2.append(_grid_div_1[j][_half_index_x:]) # Top right subgrid
                else: 
                    _grid_3.append(_grid_div_2[j - _half_index_y][:_half_index_x]) # Bottom left subgrid
                    _grid_4.append(_grid_div_2[j - _half_index_y][_half_index_x:]) # Bottom right subgrid
            # Return each grid and the top left pixel coordinate to maintain a sense of location.
            return [(_grid_1, top_left_pixel), (_grid_2, (top_left_pixel[0] + _half_index_x, top_left_pixel[1])), 
                    (_grid_3, (top_left_pixel[0], top_left_pixel[1] + _half_index_y)), 
                    (_grid_4, (top_left_pixel[0] + _half_index_x, top_left_pixel[1] + _half_index_y))]
        
        def get_border_iteration(_grid, top_left_pixel):
            """Method to determine if a given subgrid has border pixels that are all equal, so we know when to keep subdividing or stop and fill the grid."""
            _dim_x, _dim_y = len(_grid[0]) if len(_grid) > 0 else 0, len(_grid)
            
            top_left = ComplexNumber(top_left_pixel[0] * self.step_sizes[0] + x_min, top_left_pixel[1] * self.step_sizes[1] + y_min)
            _comparison = transform(top_left, self.iteration_max) # Variable used to store first result of transform, and check against every other pixel in the border.
            
            for i in range(_dim_x): # Loop that scans the top and bottom row of the subgrid. Either reads from file or passes coordinate to the transform function.
                top_num = ComplexNumber((i + top_left_pixel[0]) * self.step_sizes[0] + x_min, top_left_pixel[1] * self.step_sizes[1] + y_min)
                top_num_transform = transform(top_num, self.iteration_max)#file_contents[i + top_left_pixel[0]][top_left_pixel[1]] if is_generated else transform(top_num, self.iteration_max)

                bottom_num = ComplexNumber((i + top_left_pixel[0]) * self.step_sizes[0] + x_min, (_dim_y - 1 + top_left_pixel[1]) * self.step_sizes[1] + y_min)
                bottom_num_transform = transform(bottom_num, self.iteration_max)#file_contents[i + top_left_pixel[0]][_dim_y - 1 + top_left_pixel[1]] if is_generated else transform(bottom_num, self.iteration_max)
                if (i != 0 and top_num_transform != _comparison) or bottom_num_transform != _comparison: return -1

            for j in range(_dim_y): # Loop that scans the left and right columns of the subgrid.
                left_num = ComplexNumber(top_left_pixel[0] * self.step_sizes[0] + x_min, (j + top_left_pixel[1]) * self.step_sizes[1] + y_min)
                left_num_transform = transform(left_num, self.iteration_max)#file_contents[top_left_pixel[0]][j + top_left_pixel[1]] if is_generated else transform(left_num, self.iteration_max)
                right_num = ComplexNumber((_dim_x - 1 + top_left_pixel[0]) * self.step_sizes[0] + x_min, (j + top_left_pixel[1]) * self.step_sizes[1] + y_min)
                right_num_transform = transform(right_num, self.iteration_max)#file_contents[_dim_x - 1 + top_left_pixel[0]][j + top_left_pixel[1]] if is_generated else transform(right_num, self.iteration_max)
                if (j != 0 and left_num_transform != _comparison) or right_num_transform != _comparison: return -1

            return _comparison
        
        def evaluate_grid(_grid, top_left_pixel):
            if len(_grid) != 0 and len(_grid[0]) != 0: 
                border_iteration = get_border_iteration(_grid, top_left_pixel)
                if border_iteration == -1:    # If the border pixels are not equal, split and evaluate subgrids. 
                    grids = split_grid(_grid, top_left_pixel)
                    for i in range(len(grids)):
                        evaluate_grid(grids[i][0], grids[i][1])
                else: # Otherwise, fill in the main grid with the iteration value at the border of the subgrid.
                    if border_iteration != self.iteration_max: self.iterations_per_pixel[border_iteration] += (len(_grid) * len(_grid[0]))
                    for j in range(len(_grid)):
                        for i in range(len(_grid[j])):
                            self.grid[top_left_pixel[1] + j][top_left_pixel[0] + i] = border_iteration #if i != 0 and j != 0 else 0
                

        evaluate_grid(self.grid, (0,0))

    def render(self):
        render_start = time.time()

        prev_perc = 0 # For displaying completion percentage.
        total_pixels = 0
        for i in range(self.iteration_max - 1): 
            total_pixels += self.iterations_per_pixel[i]
        hue_grid = np.full((len(self.grid), len(self.grid[0]), 3), fill_value=(0,0,0))

        for col in range(len(self.grid[0])):
            for row in range(len(self.grid)):
                hue = 0
                current_iteration = self.grid[col][row]
                if current_iteration == self.iteration_max:
                    hue = 0
                else: 
                    for i in range(current_iteration):
                        hue += float(self.iterations_per_pixel[i] / total_pixels)
                        
                ## Hue is between 0 and 1, so a greater power for any one of the RGB channels means less significance at low hue.
                hue_grid[col][row] = ((hue**1.5) * 255, (hue**3) * 255, (hue) * 255)

                # For displaying the percentage completion of rendering.
                percent_skip = 25 # Display percentages every percent_skip.
                percentage = round(100 * col / dim_y)
                if percentage != prev_perc and percentage % percent_skip == 0 and percentage != 100:
                    print(f"Rendering is {round(100 * col / dim_y)}% complete in {round(time.time() - render_start, 3)}s...")
                    prev_perc = round(100 * col / dim_y)
        
        plt.imshow(hue_grid)
        plt.show()
        print(f"Render complete! Time to render: {round(time.time() - render_start, 3)}s.\n")

## Debug functions to edit rendering without recalculating.
def read_plane(x_min, y_min, x_max, y_max, dim_x, dim_y, iteration_max):
    file = open("plane_data.txt").read()
    file_lines = file.split("\n")
    for line in file_lines:
        metadata = line.split("|")
        if f"{x_min}, {x_max}, {y_min}, {y_max}, {dim_x}, {dim_y}, {iteration_max}" == metadata[0]:
            line_split = metadata[1].split(", ")   
            output = []
            for j in range(dim_y):
                row = []
                for i in range(dim_x):
                    row.append(int(line_split[j * dim_y + i]))
                output.append(row)
            return output
    return []

def write_plane(grid, x_min, y_min, x_max, y_max, dim_x, dim_y, iteration_max):
    file = open("plane_data.txt", "a")
    contents = f"{x_min}, {x_max}, {y_min}, {y_max}, {dim_x}, {dim_y}, {iteration_max}|"
    index = 0
    for row in grid:
        for elem in row:
            if index == 0: contents += str(elem)
            else: contents += f", {str(elem)}"
            index += 1
    file.write(contents + "\n")

if __name__ == "__main__":
    iteration_limit = 255

    def mandelbrot_transform(c, limit):
        """Function returning the number of iterations taken to blow up to infinity in the mandelbrot set, if ever (limited by limit)."""
        iteration_count = 0
        # Optimisation: bulb checking, check if the current pixel is in the main or period-2 bulb, and skip calculation if so.
        p = math.sqrt((c.x - 1/4)**2 + c.y**2)
        if c.x <= p - 2*p**2 + 1/4 or (c.x+1)**2 + c.y**2 <= 1/16:
            return iteration_limit
        
        z = ComplexNumber(0, 0)
        z_abs_square = ComplexNumber(0, 0)

        while z_abs_square.x + z_abs_square.y < 4 and iteration_count < limit:
            z = ComplexNumber(z_abs_square.x - z_abs_square.y, 2 * z.x * z.y) + c
            z_abs_square = z.absolute_square()
            iteration_count += 1
        return iteration_count
    
    # Fixed initial parameters of the program.
    dim_x = 512
    dim_y = 512

    x_min = -2.023
    y_min = -1.125
    x_max = 0.6
    y_max = 1.125

    # Get distance between each adjacent coordinate, parallel to real/imaginary axis.
    def get_step_size(_x_min, _x_max, _y_min, _y_max):
        return (_x_max - _x_min) / dim_x, (_y_max - _y_min) / dim_y

    zoom_scale = 3
    def on_click(event, figure, _x_min, _x_max, _y_min, _y_max, step_sizes, click_id):
        """Method which awaits a left button press, and calculates the new generation parameters based on the click position."""
        if event.button is MouseButton.LEFT and event.inaxes:
            click_pos = (round(event.xdata), round(event.ydata))

            click_coord_x = _x_min + click_pos[0] * step_sizes[0]
            click_coord_y = _y_min + (dim_y - click_pos[1]) * step_sizes[1]
            
            smallest_wall_distance = min([click_coord_x - _x_min, _x_max - click_coord_x,
                                          click_coord_y - _y_min, _y_max - click_coord_y]) / zoom_scale

            print(f"Zooming on coordinate ({round(click_coord_x, 3)}, {round(click_coord_y, 3)})...")

            # Connect the button release_event, which subsequently calls _main to generate around the click position.
            release_id = figure.canvas.mpl_connect('button_release_event', 
                lambda event: on_release(event, click_coord_x, click_coord_y, smallest_wall_distance, release_id))
            plt.disconnect(click_id)

    def on_release(event, click_coord_x, click_coord_y, smallest_wall_distance, release_id):
        """Method which disconnects the previous on_click connection, and begins construction of the desired zoom area."""
        if event.button is MouseButton.LEFT:
            plt.disconnect(release_id)  # Disconnects the click event 
            plt.close()
            _main(click_coord_x - smallest_wall_distance, click_coord_x + smallest_wall_distance, 
                  click_coord_y - smallest_wall_distance, click_coord_y + smallest_wall_distance, True)

    def _main(_x_min, _x_max, _y_min, _y_max, _is_zoom):
        """Method which runs the main loop of simulation."""
        fig, ax = plt.subplots()

        step_sizes = get_step_size(_x_min, _x_max, _y_min, _y_max)
        # Connects the on_click method to the event of a button click with most recent generation parameters.
        click_id = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, fig, _x_min, _x_max, _y_min, _y_max, step_sizes, click_id))

        # Attempt to find precalculated data with current metadata, given this is a full generation (not zooming).
        plane_data = [] if _is_zoom else read_plane(_x_min, _y_min, _x_max, _y_max, dim_x, dim_y, iteration_limit)
        if plane_data == []:
            generation_start = time.time()
            plane = ComplexPlane(dim_x, dim_y, _x_min, _y_min, _x_max, _y_max, step_sizes, mandelbrot_transform, iteration_limit, False, [])
            print(f"Generation complete! Time to generate: {round(time.time() - generation_start, 3)}s.")

            # Write calculated iteration data for fast debugging when using the same metadata (disregarding zooms).
            if not _is_zoom: write_plane(plane.grid, _x_min, _y_min, _x_max, _y_max, dim_x, dim_y, iteration_limit)
            plane.render()
        else:
            print("Importing found data...")
            plane = ComplexPlane(dim_x, dim_y, _x_min, _y_min, _x_max, _y_max, step_sizes, mandelbrot_transform, iteration_limit, True, plane_data)
            plane.render()

    _main(x_min, x_max, y_min, y_max, False)
