import numpy as np
import cvxpy as cp
from itertools import combinations
import heapq
from scipy.special import comb as nCr
import time
import random


def main():
    #puzzle = [[4,1,4,2,3],[3,2,1,3,3],[2,2,1,2,3],[2,2,2,1,3]]
    puzzle = [[2,2,2,1,3],[2,2,3,3,1],[2,2,3,3,1],[2,2,2,1,3]]
    start_time = time.clock()
    output = solve_tower(puzzle)
    end_time = time.clock()
    for line in output[0]:
        print(line)
    print(output[1])
    print(output[2])
    print(end_time - start_time, "seconds")
    print("Solved:",validate_solution(np.array(output[0]),puzzle))

def get_visibility_number(towers):
    # Towers is a numpy array, with towers from height 1 to n.
    visible = 0
    tallest = 0
    for height in towers:
        if height > tallest:
            tallest = height
            visible += 1
    return visible

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
            left_visible != left_visibility[i]
            ):
            return False

    return True

def solve_tower(puzzle):
    # Puzzles are in the format:
    # Row 1: Top row
    # Row 2: Right columns
    # Row 3: Bottom columns
    # Row 4: Left row

    top = puzzle[0]
    right = puzzle[1]
    bottom = puzzle[2]
    left = puzzle[3]

    size = len(top)

    # Vector of 1-n constant
    nums = np.transpose(np.arange(1,size+1))

    # Create the n by n by n boolean values
    X = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(cp.Variable(size))
        X.append(row)

    Y = []

    # Create the objective to solve
    #obj = cp.Minimize(sum([sum(cp.max_elemwise(x)) for x in X]))
    obj = cp.Minimize(0)

    # Create the constraints
    constraints = []

    # Create the between zero and one constraint
    for i in range(size):
        for j in range(size):
                constraints.append(X[i][j][:] <= 1)
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
    for j in range(size):
        # Find the combinations of the sets of inequalities
        comb = list(combinations([i for i in range(1,size)],top[j] - 1))

        # Create the boolean variables to account for the values
        Y.append(cp.Variable(len(comb)))
        # Create the between 0 and 1 constraint
        for k in range(len(comb)):
            constraints.append(Y[j][k] <= 1)
            constraints.append(Y[j][k] >= 0)
        # Create the only one on constraint
        constraints.append(cp.sum_entries(Y[j]) == 1)

        for comb_index in range(len(comb)):
            pos = comb[comb_index]
            # Create the set of inequalities, starting with the first element
            # Check if there is a next largest value:
            if len(pos) != 0:
                # Create the constraint that the value is less than the next largest value
                constraints.append(nums * X[0][j] < nums * X[pos[0]][j] + (size + 1) * (1-Y[j][comb_index]))

                # Create the constraint that the current value is larger than
                # the values between it and the next largest value
                for i in range(1,pos[0]):
                    constraints.append((size + 1) * (1-Y[j][comb_index]) + nums * X[0][j] > nums * X[i][j])

                # Create the rest of the inequalities
                for p in range(len(pos)):
                    index = pos[p]
                    # Check if we're at the last largest value
                    if p == len(pos)-1:
                        # Create the constraint that the current value is greater than the rest of the values, aka, max value
                        #for i in range(index+1,size):
                        #    constraints.append((size + 1) * (1-Y[j][comb_index]) + nums * X[index][j] > nums * X[i][j])
                        constraints.append((size + 1) * (1-Y[j][comb_index]) + nums * X[index][j] >= size)

                    else:
                        # Create the constraint that the current value is less than the next largest value
                        constraints.append(nums * X[index][j] < (size + 1) * (1-Y[j][comb_index]) + nums * X[pos[p+1]][j])
                        # Create the constraint that the current value is larger than
                        # the values between it and the next largest value
                        for i in range(index+1,pos[p+1]):
                            constraints.append((size + 1) * (1-Y[j][comb_index]) + nums * X[index][j] > nums * X[i][j])
            else:
                # If there are no next larger numbers, first number is max
                constraints.append(X[0][j][size-1] == 1)

    # Create the right col constraints
    for i in range(size):
        # Find the combinations of the sets of inequalities
        comb = list(combinations([j for j in range(1,size)],right[i] - 1))
        # Create the boolean variables to account for the values
        Y.append(cp.Variable(len(comb)))
        # Create the between 0 and 1 constraint
        for k in range(len(comb)):
            constraints.append(Y[5+i][k] <= 1)
            constraints.append(Y[5+i][k] >= 0)
        # Create the only one on constraint
        constraints.append(cp.sum_entries(Y[5+i]) == 1)

        for comb_index in range(len(comb)):
            pos = comb[comb_index]
            # Create the set of inequalities, starting with the first element
            # Check if there is a next largest value:
            if len(pos) != 0:
                # Create the constraint that the value is less than the next largest value
                constraints.append(nums * X[i][-1] < nums * X[i][-1*pos[0]-1] + (size + 1) * (1-Y[5+i][comb_index]))

                # Create the constraint that the current value is larger than
                # the values between it and the next largest value
                for j in range(1,pos[0]):
                    constraints.append((size + 1) * (1-Y[5+i][comb_index]) + nums * X[i][-1] > nums * X[i][-1*j-1])

                # Create the rest of the inequalities
                for p in range(len(pos)):
                    index = pos[p]
                    # Check if we're at the last largest value
                    if p == len(pos)-1:
                        # Create the constraint that the current value is
                        # greater than the rest of the values left, aka max
                        #for j in range(index+1,size):
                        #    constraints.append((size + 1) * (1-Y[5+i][comb_index]) + nums * X[i][-1*index-1] > nums * X[i][-1*j-1])
                        constraints.append((size + 1) * (1-Y[5+i][comb_index]) + nums * X[i][-1*index-1] >= size)
                    else:
                        # Create the constraint that the current value is less than the next largest value
                        constraints.append(nums * X[i][-1*index-1] < (size + 1) * (1-Y[5+i][comb_index]) + nums * X[i][-1*pos[p+1]-1])
                        # Create the constraint that the current value is larger than
                        # the values between it and the next largest value
                        for j in range(index+1,pos[p+1]):
                            constraints.append((size + 1) * (1-Y[5+i][comb_index]) + nums * X[i][-1*index-1] > nums * X[i][-1*j-1])
            else:
                # If there are no next larger numbers, first right most number is max
                constraints.append(X[i][-1][size-1] == 1)

    # Create the bottom row constraints
    for j in range(size):
        # Find the combinations of the sets of inequalities
        comb = list(combinations([i for i in range(1,size)],bottom[j] - 1))

        # Create the boolean variables to account for the values
        Y.append(cp.Variable(len(comb)))
        # Create the between 0 and 1 constraint
        for k in range(len(comb)):
            constraints.append(Y[10+j][k] <= 1)
            constraints.append(Y[10+j][k] >= 0)
        # Create the only one on constraint
        constraints.append(cp.sum_entries(Y[10+j]) == 1)

        for comb_index in range(len(comb)):
            pos = comb[comb_index]
            # Create the set of inequalities, starting with the first element
            # Check if there is a next largest value:
            if len(pos) != 0:
                # Create the constraint that the value is less than the next largest value
                constraints.append(nums * X[-1][j] < nums * X[-1*pos[0]-1][j] + (size + 1) * (1-Y[10+j][comb_index]))

                # Create the constraint that the current value is larger than
                # the values between it and the next largest value
                for i in range(1,pos[0]):
                    constraints.append((size + 1) * (1-Y[10+j][comb_index]) + nums * X[-1][j] > nums * X[-1*i-1][j])

                # Create the rest of the inequalities
                for p in range(len(pos)):
                    index = pos[p]
                    # Check if we're at the last largest value
                    if p == len(pos)-1:
                        # Create the constraint that the current value is
                        # greater than the rest of the values left, aka max
                        #for i in range(index+1,size):
                        #    constraints.append((size + 1) * (1-Y[10+j][comb_index]) + nums * X[-1*index-1][j] > nums * X[-1*i-1][j])
                        constraints.append((size + 1) * (1-Y[10+j][comb_index]) + nums * X[-1*index-1][j] >= size)
                    else:
                        # Create the constraint that the current value is less than the next largest value
                        constraints.append(nums * X[-1*index-1][j] < (size + 1) * (1-Y[10+j][comb_index]) + nums * X[-1*pos[p+1]-1][j])
                        # Create the constraint that the current value is larger than
                        # the values between it and the next largest value
                        for i in range(index+1,pos[p+1]):
                            constraints.append((size + 1) * (1-Y[10+j][comb_index]) + nums * X[-1*index-1][j] > nums * X[-1*i-1][j])
            else:
                # If there are no next larger numbers, first right most number is max
                #for i in range(1,size):
                #    constraints.append((size + 1) * (1-Y[10+j][comb_index]) + nums * X[-1][j] > nums * X[-1*i-1][j])
                constraints.append(X[-1][j][size-1] == 1)

    # Create the left col constraints
    for i in range(size):
        # Find the combinations of the sets of inequalities
        comb = list(combinations([j for j in range(1,size)],left[i] - 1))

        # Create the boolean variables to account for the values
        Y.append(cp.Variable(len(comb)))
        # Create the between 0 and 1 constraint
        for k in range(len(comb)):
            constraints.append(Y[15+i][k] <= 1)
            constraints.append(Y[15+i][k] >= 0)
        # Create the only one on constraint
        constraints.append(cp.sum_entries(Y[15+i]) == 1)

        for comb_index in range(len(comb)):
            pos = comb[comb_index]
            # Create the set of inequalities, starting with the first element
            # Check if there is a next largest value:
            if len(pos) != 0:
                # Create the constraint that the value is less than the next largest value
                constraints.append(nums * X[i][0] < nums * X[i][pos[0]] + (size + 1) * (1-Y[15+i][comb_index]))

                # Create the constraint that the current value is larger than
                # the values between it and the next largest value
                for j in range(1,pos[0]):
                    constraints.append((size + 1) * (1-Y[15+i][comb_index]) + nums * X[i][0] > nums * X[i][j])

                # Create the rest of the inequalities
                for p in range(len(pos)):
                    index = pos[p]
                    # Check if we're at the last largest value
                    if p == len(pos)-1:
                        # Create the constraint that the current value is
                        # greater than the rest of the values, aka max
                        #for j in range(index+1,size):
                        #    constraints.append((size + 1) * (1-Y[15+i][comb_index]) + nums * X[i][index] > nums * X[i][j])
                        constraints.append((size + 1) * (1-Y[15+i][comb_index]) + nums * X[i][index] >= size)
                    else:
                        # Create the constraint that the current value is less than the next largest value
                        constraints.append(nums * X[i][index] < (size + 1) * (1-Y[15+i][comb_index]) + nums * X[i][pos[p+1]])
                        # Create the constraint that the current value is larger than
                        # the values between it and the next largest value
                        for j in range(index+1,pos[p+1]):
                            constraints.append((size + 1) * (1-Y[15+i][comb_index]) + nums * X[i][index] > nums * X[i][j])
            else:
                # If there are no next larger numbers, first number is max
                #for j in range(1,size):
                #    constraints.append((size + 1) * (1-Y[15+i][comb_index]) + nums * X[i][0] > nums * X[i][j])
                constraints.append(X[i][0][size-1] == 1)

    prob = cp.Problem(obj,constraints)
    prob.solve()

    # Branch and Bound
    # Create the frontier as a heapquene
    frontier = []

    # Check if the problem was infeasible
    if not(prob.status == 'infeasible'):
        # Turn the solution into a numpy array
        output = []
        for i in range(size):
            output_row = []
            for j in range(size):
                output_row.append(X[i][j].value)
            output.append(output_row)
        x_star = np.array(output)

        max_y_size = np.int_(nCr(size-1,round((size-1)/2)))
        output = []
        for i in range(size*4):
            Y_val = np.array(Y[i].value,ndmin=2, copy=False)
            output.append(np.append(Y_val,np.array(np.zeros((max_y_size-Y_val.shape[0],1)),ndmin=2,copy=False),axis=0))
        y_star = np.array(output)

        # Round values
        x_star[np.logical_and(x_star >= -.005, x_star <= .005)] = 0
        x_star[np.logical_and(x_star >= 1-.005, x_star <= 1+.005)] = 1

        y_star[np.logical_and(y_star >= -.005, y_star <= .005)] = 0
        y_star[np.logical_and(y_star >= 1-.005, y_star <= 1+.005)] = 1

        # Count the number of ones in x and y
        total_count = np.count_nonzero(x_star == 1)
        total_count += np.count_nonzero(y_star == 1)

        # Push the current solution onto the heapquene
        heapq.heappush(frontier, (random.random()-total_count,0, x_star, [], y_star, []))
    else:
        heapq.heappush(frontier, (random.random()-total_count,0, None, [], None, []))
    iters = 0
    while(len(frontier) > 0):
        iters += 1
        # Get the lowest cost solution
        item = heapq.heappop(frontier)

        # Check if the lowest cost solution is integer valued and exists
        if not(item[2] is None):

            x_done = np.all(np.logical_or(item[2] == 0, item[2] == 1))
            y_done = np.all(np.logical_or(item[4] == 0, item[4] == 1))
            if(x_done and y_done):
                # If it is, format an solution array, and then return it.
                solution = []
                for i in range(size):
                    solution_row = []
                    for j in range(size):
                        solution_row.append(np.nonzero(item[2][i][j])[0][0] + 1)
                    solution.append(solution_row)
                constraints_int = item[3]
                return (solution, constraints_int,iters)

            # If no integer valued solution, find variable closest to .5
            closestX = (np.abs(item[2]-.5))
            closestY = (np.abs(item[4]-.5))
            x_constraint = False
            closest = closestY
            if(closestX.min() < closestY.min()):
                x_constraint = True
                closest = closestX
            closest_index = np.where(closest == closest.min())
            new_i = closest_index[0][0]
            new_j = closest_index[1][0]
            new_k = closest_index[2][0]

            # Get the current constraint set
            new_constraints = []
            for con in item[3]:
                new_constraints.append(X[con[0]-1][con[1]-1][con[2]-1] == con[3])
            for con in item[5]:
                new_constraints.append(Y[con[0]][con[1]] == con[2])

            # Solve the 0 and 1 constraint problems, and add them to the frontier
            for c in range(2):
                if(x_constraint):
                    prob = cp.Problem(obj,constraints + new_constraints + [X[new_i][new_j][new_k] == c])
                else:
                    prob = cp.Problem(obj,constraints + new_constraints + [Y[new_i][new_j] == c])
                prob.solve(solver=cp.ECOS)

                # Check if the problem was infeasible
                if not(prob.status.find('infeasible') > -1):
                    # Turn the solution into a numpy array
                    output = []
                    for i in range(size):
                        output_row = []
                        for j in range(size):
                            output_row.append(X[i][j].value)
                        output.append(output_row)
                    x_star = np.array(output)

                    max_y_size = np.int_(nCr(size-1,round((size-1)/2)))
                    output = []
                    for i in range(size*4):
                        Y_val = np.array(Y[i].value,ndmin=2, copy=False)
                        output.append(np.append(Y_val,np.array(np.zeros((max_y_size-Y_val.shape[0],1)),ndmin=2,copy=False),axis=0))
                    y_star = np.array(output)

                    # Round values
                    x_star[np.logical_and(x_star >= -.005, x_star <= .005)] = 0
                    x_star[np.logical_and(x_star >= 1-.005, x_star <= 1+.005)] = 1

                    y_star[np.logical_and(y_star >= -.005, y_star <= .005)] = 0
                    y_star[np.logical_and(y_star >= 1-.005, y_star <= 1+.005)] = 1

                    # Count the number of ones in x and y
                    total_count = np.count_nonzero(x_star == 1)
                    total_count += np.count_nonzero(y_star == 1)

                    # Count the number of 1 constraints in x and y
                    for con in item[3]:
                        total_count += con[3]
                    for con in item[5]:
                        total_count += con[2]
                    total_count += c

                    output = []
                    for i in range(size):
                        row = []
                        for j in range(size):
                            if(np.all(np.logical_or(x_star[i][j] == 0, x_star[i][j] == 1))):
                                row.append(np.nonzero(x_star[i][j])[0][0] + 1)
                            else:
                                row.append(0)
                        output.append(row)
                    x_solution = np.array(output)

                    x_done = np.all(np.logical_or(x_star == 0, x_star == 1))
                    valid_x = False
                    if(x_done):
                        valid_x = validate_solution(x_solution,puzzle)

                    if not (x_done and not valid_x):
                        if(x_constraint):
                            heapq.heappush(frontier,(random.random()-total_count,item[1]+1,x_star,item[3] + [(new_i+1, new_j+1, new_k+1, c)],y_star,item[5]))
                        else:
                            heapq.heappush(frontier,(random.random()-total_count,item[1]+1,x_star,item[3],y_star,item[5] + [(new_i,new_j,c)]))
                else:
                    # Push the current solution onto the heapquene
                    if(x_constraint):
                        heapq.heappush(frontier,(random.random()-c,item[1]+1,None,item[3] + [(new_i+1, new_j+1, new_k+1, c)],None,item[5]))
                    else:
                        heapq.heappush(frontier,(random.random()-c,item[1]+1,None,item[3],None,item[5] + [(new_i,new_j,c)]))


if __name__ == '__main__':
    main()
