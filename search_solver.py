import itertools
import random
import time

import numpy as np
import puzzle_gen

def solve(puzzle):

    grid_size = len(puzzle[0])
    top_visibility = puzzle[0]
    right_visibility = puzzle[1]
    bottom_visibility = puzzle[2]
    left_visibility = puzzle[3]

    # Start by getting all possible permutations of the 5 numbers.
    permutations = map(list, itertools.permutations(range(1, grid_size + 1)))
   
    row_dict = dict()
    for p in permutations:
        left_visible = puzzle_gen.get_visibility_number(p)
        right_visible = puzzle_gen.get_visibility_number(p[::-1])

        key = (left_visible, right_visible)
        current = row_dict.get(key, [])
        current.append(p)
        row_dict[key] = current

    # Permute the list to make distribute search times better
    for key in row_dict:
        random.shuffle(row_dict[key])

    grid = np.zeros((grid_size, grid_size))

    def recurse(row_number):

        if row_number == grid_size: # Filled all rows
            return puzzle_gen.validate_solution(grid, puzzle)

        # When we need to fill a row, look at all of the possibilities for the
        # row by considering 
        left_visible = left_visibility[row_number]
        right_visible = right_visibility[row_number]
        key = (left_visible, right_visible)

        for row in row_dict[key]:

            # Apply the row.
            grid[row_number] = row

            # Recurse
            if recurse(row_number + 1):
                return True

    recurse(0)

    return grid


def get_random_puzzle(size):
    filename = "%d_square_board_data.npz" % size
    loaded = np.load(filename)

    boards_array = loaded['boards_array']
    index = random.randint(0, len(boards_array) - 1) # endpoints included
    print("INDEX:", index)

    board = boards_array[index]
    board_size = int(len(board) ** 0.5)
    board = board.reshape((board_size, board_size))
    puzzle = puzzle_gen.get_visibility_numbers_from_grid(board)
    
    print(board)
    return puzzle


def main():
    times = []

    num_trials = 20
    for i in range(num_trials):
        puzzle = get_random_puzzle(5)
        start = time.time()
        soln = solve(puzzle)
        end = time.time()
        times.append(end - start)

    print("Average time per solution:", sum(times) / num_trials)


if __name__ == "__main__":
    main()
