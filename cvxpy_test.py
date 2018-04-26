import numpy as np
import cvxpy as cp

def solve_tower(puzzle):
    # Puzzles are in the format:
    # Row 1: Top row
    # Row 2: Left columns
    # Row 3: Right columns
    # Row 4: Bottom row

    top = puzzle[0]
    left = puzzle[1]
    right = puzzle[2]
    bottom = puzzle[3]

    size = len(top)

    # Vector of 1-n constant
    nums = np.arange(size)

    # Create the 5 by 5 integer value
    X = []
    for i in range(size):
        X_row = []
        for j in range(size):
            X_row.append(cp.Int(size))
        X.append(X_row)

    # Create the objective to solve
    obj = cp.Minimize(sum([sum(cp.max_elemwise(x)) for x in X]))

    # Create the constraints
    constraints = []

    # Create the between zero and one constraint
    for i in range(size):
        for j in range(size):
            constraints.append(X[i][j][:] <= 1)

    for i in range(size):
        for j in range(size):
            constraints.append(X[i][j][:] >= 0)

    # Create the only one assigned number constraint
    for i in range(size):
        for j in range(size):
            constraints.append(cp.sum_entries(X[i][j]) == 1)

    # Create the each number constraint
    # For columns
    for j in range(size):
        constraints.append(sum([X[i][j] for i in range(size)]) == 1)

    # For rows
    for i in range(size):
        constraints.append(sum([X[i][j] for j in range(size)]) == 1)

    # Create the top row constraints
    for i in range(size):
        # if the value is either 1 or 5, value has to be 5 or 1
        if(top[i] == 1):
            constraints.append(X[0][i][5] == 1)
        elif(top[i] == 5):
            constraints.append(X[0][i][1] == 1)
        # if the value is 2, then create the constraints that the value closest
        # to the edge has to be greater than all the values between it and the
        # 5 value
