import numpy as np
import itertools
import pprint
import time

def get_visibility_number(towers):
    # Towers is a numpy array, with towers from height 1 to n.
    visible = 0
    tallest = 0
    for height in towers:
        if height > tallest:
            tallest = height
            visible += 1
    return visible

def test_get_visibility_number():
    grid = np.array([[3, 5, 1, 2, 4],
                     [2, 1, 4, 5, 3],
                     [5, 4, 2, 3, 1],
                     [1, 2, 3, 4, 5],
                     [4, 3, 5, 1, 2]])

    # Visibility numbers given as Top, Right, Bottom, Left, always right/down
    top = (2, 1, 3, 2, 2)
    right = (2, 2, 4, 1, 2)
    bottom = (2, 3, 1, 3, 2)
    left = (2, 3, 1, 5, 2)

    # Test each of the numbers 
    for i, visibility in enumerate(top):
        skyline = grid[:, i] 
        assert(get_visibility_number(skyline) == visibility)

    for i, visibility in enumerate(right):
        skyline = grid[i, :][::-1]
        assert(get_visibility_number(skyline) == visibility)
    
    for i, visibility in enumerate(bottom):
        skyline = grid[:, i][::-1]
        assert(get_visibility_number(skyline) == visibility)
    
    for i, visibility in enumerate(left):
        skyline = grid[i, :]
        assert(get_visibility_number(skyline) == visibility)


def validate_solution(grid, visibility):
    # grid is the numpy array
    # Visibility is a python tuple of Top, Right, Bottom, Left constraints

    grid_size = len(grid)
    top_visibility = visibility[0]
    right_visibility = visibility[1]
    bottom_visibility = visibility[2]
    left_visibility = visibility[3]


    for i in range(grid_size):
        top_skyline = grid[:, i]
        right_skyline = grid[i, :][::-1]
        bottom_skyline = grid[:, i][::-1]
        left_skyline = grid[i, :]
        
        top_visible = get_visibility_number(top_skyline)
        right_visible = get_visibility_number(right_skyline)
        bottom_visible = get_visibility_number(bottom_skyline)
        left_visible = get_visibility_number(left_skyline)

        # Fails if any of the constraints don't work
        if (top_visible != top_visibility[i] or
            right_visible != right_visibility[i] or
            bottom_visible != bottom_visibility[i] or
            left_visible != left_visibility[i]):
            return False

    return True
        

def test_validate_solution():

    grid = np.array([[3, 5, 1, 2, 4],
                     [2, 1, 4, 5, 3],
                     [5, 4, 2, 3, 1],
                     [1, 2, 3, 4, 5],
                     [4, 3, 5, 1, 2]])

    top = (2, 1, 3, 2, 2)
    right = (2, 2, 4, 1, 2)
    bottom = (2, 3, 1, 3, 2)
    left = (2, 3, 1, 5, 2)

    visibility = (top, right, bottom, left)

    assert(validate_solution(grid, visibility))


def get_visibility_numbers_from_grid(grid):
    grid_size = len(grid)
    
    top_visibility = [get_visibility_number(grid[:, i]) for i in range(grid_size)]
    right_visibility = [get_visibility_number(grid[i, :][::-1]) for i in range(grid_size)]
    bottom_visibility = [get_visibility_number(grid[:, i][::-1]) for i in range(grid_size)]
    left_visibility = [get_visibility_number(grid[i, :]) for i in range(grid_size)]

    visibility = (tuple(top_visibility), tuple(right_visibility),
                  tuple(bottom_visibility), tuple(left_visibility))
    return visibility


def test_get_visibility_numbers_from_grid():
    grid = np.array([[3, 5, 1, 2, 4],
                     [2, 1, 4, 5, 3],
                     [5, 4, 2, 3, 1],
                     [1, 2, 3, 4, 5],
                     [4, 3, 5, 1, 2]])
    
    top = (2, 1, 3, 2, 2)
    right = (2, 2, 4, 1, 2)
    bottom = (2, 3, 1, 3, 2)
    left = (2, 3, 1, 5, 2)

    visibility = (top, right, bottom, left)
    assert(visibility == get_visibility_numbers_from_grid(grid))


def validate_grid(grid):
    # Check that the grid is in fact a 2d square
    is_square = len(grid.shape) == 2 and grid.shape[0] == grid.shape[1]
    if not is_square:
        return False
    grid_size = len(grid)

    # Check that each row / col contains 1 of each number
    expected = np.array(range(1, grid_size + 1))
    for i in range(grid_size):
        row = grid[i, :]
        col = grid[:, i]
    
        if (np.sort(row) != expected).any() or (np.sort(col) != expected).any():
            return False

    return True


def test_validate_grid():
    grid = np.array([[3, 5, 1, 2, 4],
                     [2, 1, 4, 5, 3],
                     [5, 4, 2, 3, 1],
                     [1, 2, 3, 4, 5],
                     [4, 3, 5, 1, 2]])
    assert(validate_grid(grid) == True)
    
    grid = np.array([[3, 5, 1, 2, 4],
                     [2, 1, 4, 5, 3],
                     [5, 4, 2, 3, 1],
                     [1, 2, 3, 4, 5],
                     [4, 3, 5, 2, 2]])
    assert(validate_grid(grid) == False)
    
    grid = np.array([[3, 5, 1, 2, 4],
                     [2, 1, 4, 5, 3],
                     [5, 4, 2, 3, 1],
                     [1, 2, 3, 4, 5],
                     [4, 3, 6, 1, 2]])
    assert(validate_grid(grid) == False)


def base_counter(bits, base):

    counter = 0
    while counter < base ** bits:
        temp = counter
        value = np.zeros(bits)
        for bit in range(bits):
            value[bit] = temp % base
            temp //= base

        yield tuple(value[::-1])
        counter += 1


def generate_all_grids_with_permutations(size = 5):

    # Choose to generate all permutations up front, at least for now.
    perms = itertools.permutations(range(1, size + 1))

    row_indices = np.ones(size) * (len(perms) - 1)
    while row_indices.sum() >= 0:
        pass


def generate_boards(size):

    #  all_boards = dict()
    all_boards = []

    board = np.zeros((size, size))

    def get_possible_moves(row_idx, col_idx):
        all_possible = set(range(1, size+1))
        row = board[row_idx, :col_idx]
        col = board[:row_idx, col_idx]
        remaining = all_possible - set(row) - set(col)
        return tuple(remaining)


    def generate(row_idx, col_idx):

        # If we're at the last cell, print the board
        if row_idx == size:
            is_valid = validate_grid(board)
            if not is_valid:
                print("Found board that isn't valid!")
            else:
                all_boards.append(board.copy())

            return

        # Otherwise, figure out all valid moves for the current cell
        possible = get_possible_moves(row_idx, col_idx)
    
        # And then make those moves, and fill in the next cell.
        next_col_idx = col_idx + 1
        next_row_idx = row_idx + int(next_col_idx == size)
        next_col_idx %= size
        #  print("Moving from (%d, %d) to (%d, %d)" % (row_idx, col_idx,
        #      next_row_idx, next_col_idx))

        for move in possible:
            board[row_idx, col_idx] = move
            generate(next_row_idx, next_col_idx)
        
    generate(0, 0) # Recursively generate boards
    return all_boards



def generate_boards_brute(size):

    all_boards = dict()
    board = np.zeros((size, size))

    def generate(row_idx, col_idx):
        if row_idx == size:
            if validate_grid(board):
                vis = get_visibility_numbers_from_grid(board)
                all_boards[vis] = all_boards.get(vis, []) + [board.copy()]
            return

        next_col_idx = col_idx + 1
        next_row_idx = row_idx + int(next_col_idx == size)
        next_col_idx %= size
        for move in range(1, size+1):
            board[row_idx, col_idx] = move
            generate(next_row_idx, next_col_idx)

    generate(0, 0)
    return all_boards


test_get_visibility_number()
test_validate_solution()
test_get_visibility_numbers_from_grid()
test_validate_grid()


def get_vis_groups_from_boards(boards):
    # Returns a dictionary mapping a constraint to all of its solutions
    vis_groups = dict()
    for board in boards:
        vis = get_visibility_numbers_from_grid(board)
        if vis in vis_groups:
            vis_groups[vis].append(board.copy())
        else:
            vis_groups[vis] = [board.copy()]

    return vis_groups


def get_vis_counts_from_vis_groups(vis_groups):
    return {vis:len(solns) for (vis, solns) in vis_groups.items()}


def get_arrays_from_vis_counts(vis_counts):

    rows = []
    counts = []

    for (vis, count) in vis_counts.items():
        rows.append(np.array(vis).reshape(-1))
        counts.append(count)

    return np.vstack(tuple(rows)), np.vstack(tuple(counts))


def get_arrays_from_boards(boards, vis_counts):

    rows = []
    counts = []

    for board in boards:
        vis = get_visibility_numbers_from_grid(board)
        count = vis_counts[vis]

        rows.append(np.array(board).reshape(-1))
        counts.append(count)

    return np.vstack(tuple(rows)), np.vstack(tuple(counts))


for i in range(1, 6):
    start = time.time()
    boards = generate_boards(i)
    end = time.time()

    # First, sort the boards by their constraints
    vis_groups = get_vis_groups_from_boards(boards)
    vis_counts = get_vis_counts_from_vis_groups(vis_groups)

    # Next, generate some arrays for data visualization later
    vis_array, vis_counts_array = get_arrays_from_vis_counts(vis_counts)
    boards_array, boards_counts_array = get_arrays_from_boards(boards, vis_counts)
    np.savez("%d_square_board_data.npz" % i,
             vis_array=vis_array,
             vis_counts_array=vis_counts_array,
             boards_array=boards_array,
             boards_counts_array=boards_counts_array,
            )

    # Determine some statistics:
    unique_constraints = len(vis_groups)

    possible_puzzles = 0
    for key in vis_groups:
        possible_puzzles += len(vis_groups[key])

    solution_counts = dict()
    for key in vis_groups:
        solutions = len(vis_groups[key])
        solution_counts[solutions] = solution_counts.get(solutions, 0) + 1

    print("\tUnique Constraints:", unique_constraints)
    print("\tPossible Puzzles:  ", possible_puzzles)
    print("\tSolution Counts:   ", solution_counts)

    print("Took %f seconds to generate %d size board" % (end - start, i))
