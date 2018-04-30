import numpy as np
import cvxpy as cp
from itertools import combinations
import heapq
import random
from scipy.special import comb as nCr


def main():
    puzzle = [[2,2,2,1,3],[2,2,3,3,1],[2,2,3,3,1],[2,2,2,1,3]]
    print(cp.installed_solvers())
    solve_tower(puzzle)

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
    obj = cp.Minimize(sum([sum(cp.max_elemwise(x)) for x in X]))

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
                # 0 checking:
                if pos[0] - 1 != 0:
                    for i in range(1,pos[0]):
                        constraints.append((size + 1) * (1-Y[j][comb_index]) + nums * X[0][j] > nums * X[i][j])

                # Create the rest of the inequalities
                for p in range(len(pos)):
                    index = pos[p]
                    # Check if we're at the last largest value
                    if p == len(pos)-1:
                        # Create the constraint that the current value is greater than the rest of the values
                        for i in range(index+1,size):
                            constraints.append((size + 1) * (1-Y[j][comb_index]) + nums * X[index][j] > nums * X[i][j])
                    else:
                        # Create the constraint that the current value is less than the next largest value
                        constraints.append(nums * X[index][j] < (size + 1) * (1-Y[j][comb_index]) + nums * X[pos[p+1]][j])
                        # Create the constraint that the current value is larger than
                        # the values between it and the next largest value
                        # 0 checking:
                        for i in range(index+1,pos[p+1]):
                            constraints.append((size + 1) * (1-Y[j][comb_index]) + nums * X[index][j] > nums * X[i][j])
            else:
                # If there are no next larger numbers, first number is max
                for i in range(1,size):
                    constraints.append((size + 1) * (1-Y[j][comb_index]) + nums * X[0][j] > nums * X[i][j])

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
                constraints.append(cp.sum_entries(cp.mul_elemwise(nums, X[i][-1])) < cp.sum_entries(cp.mul_elemwise(nums, X[i][-1*pos[0]-1])) + (size + 1) * (1-Y[5+i][comb_index]))

                # Create the constraint that the current value is larger than
                # the values between it and the next largest value
                # 0 checking:
                if pos[0] - 1 != 0:
                    for j in range(1,pos[0]):
                        constraints.append((size + 1) * (1-Y[5+i][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[i][-1])) > cp.sum_entries(cp.mul_elemwise(nums, X[i][-1*j-1])))

                # Create the rest of the inequalities
                for p in range(len(pos)):
                    index = pos[p]
                    # Check if we're at the last largest value
                    if p == len(pos)-1:
                        # Create the constraint that the current value is greater than the rest of the values left
                        # 0 checking:
                        if size - index - 1 != 0:
                            for j in range(index+1,size):
                                constraints.append((size + 1) * (1-Y[5+i][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[i][-1*index-1])) > cp.sum_entries(cp.mul_elemwise(nums, X[i][-1*j-1])))
                    else:
                        # Create the constraint that the current value is less than the next largest value
                        constraints.append(cp.sum_entries(cp.mul_elemwise(nums, X[i][-1*index-1])) < (size + 1) * (1-Y[5+i][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[i][-1*pos[p+1]-1])))
                        # Create the constraint that the current value is larger than
                        # the values between it and the next largest value
                        # 0 checking:
                        if pos[p+1] - index - 1 != 0:
                            for j in range(index+1,pos[p+1]):
                                constraints.append((size + 1) * (1-Y[5+i][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[i][-1*index-1])) > cp.sum_entries(cp.mul_elemwise(nums, X[i][-1*j-1])))
            else:
                # If there are no next larger numbers, first right most number is max
                for j in range(1,size):
                    constraints.append((size + 1) * (1-Y[5+i][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[i][-1])) > cp.sum_entries(cp.mul_elemwise(nums, X[i][-1*j-1])))

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
                constraints.append(cp.sum_entries(cp.mul_elemwise(nums, X[-1][j])) < cp.sum_entries(cp.mul_elemwise(nums, X[-1*pos[0]-1][j])) + (size + 1) * (1-Y[10+j][comb_index]))

                # Create the constraint that the current value is larger than
                # the values between it and the next largest value
                # 0 checking:
                if pos[0] - 1 != 0:
                    for i in range(1,pos[0]):
                        constraints.append((size + 1) * (1-Y[10+j][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[-1][j])) > cp.sum_entries(cp.mul_elemwise(nums, X[-1*i-1][j])))

                # Create the rest of the inequalities
                for p in range(len(pos)):
                    index = pos[p]
                    # Check if we're at the last largest value
                    if p == len(pos)-1:
                        # Create the constraint that the current value is greater than the rest of the values left
                        # 0 checking:
                        if size - index - 1 != 0:
                            for i in range(index+1,size):
                                constraints.append((size + 1) * (1-Y[10+j][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[-1*index-1][j])) > cp.sum_entries(cp.mul_elemwise(nums, X[-1*i-1][j])))
                    else:
                        # Create the constraint that the current value is less than the next largest value
                        constraints.append(cp.sum_entries(cp.mul_elemwise(nums, X[-1*index-1][j])) < (size + 1) * (1-Y[10+j][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[-1*pos[p+1]-1][j])))
                        # Create the constraint that the current value is larger than
                        # the values between it and the next largest value
                        # 0 checking:
                        if pos[p+1] - index - 1 != 0:
                            for i in range(index+1,pos[p+1]):
                                constraints.append((size + 1) * (1-Y[10+j][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[-1*index-1][j])) > cp.sum_entries(cp.mul_elemwise(nums, X[-1*i-1][j])))
            else:
                # If there are no next larger numbers, first right most number is max
                for i in range(1,size):
                    constraints.append((size + 1) * (1-Y[10+j][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[-1][j])) > cp.sum_entries(cp.mul_elemwise(nums, X[-1*i-1][j])))

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
                constraints.append(cp.sum_entries(cp.mul_elemwise(nums, X[i][0])) < cp.sum_entries(cp.mul_elemwise(nums, X[i][pos[0]])) + (size + 1) * (1-Y[15+i][comb_index]))

                # Create the constraint that the current value is larger than
                # the values between it and the next largest value
                # 0 checking:
                if pos[0] - 1 != 0:
                    for j in range(1,pos[0]):
                        constraints.append((size + 1) * (1-Y[15+i][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[i][0])) > cp.sum_entries(cp.mul_elemwise(nums, X[i][j])))

                # Create the rest of the inequalities
                for p in range(len(pos)):
                    index = pos[p]
                    # Check if we're at the last largest value
                    if p == len(pos)-1:
                        # Create the constraint that the current value is greater than the rest of the values
                        # 0 checking:
                        if size - index - 1 != 0:
                            for j in range(index+1,size):
                                constraints.append((size + 1) * (1-Y[15+i][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[i][index])) > cp.sum_entries(cp.mul_elemwise(nums, X[i][j])))
                    else:
                        # Create the constraint that the current value is less than the next largest value
                        constraints.append(cp.sum_entries(cp.mul_elemwise(nums, X[i][index])) < (size + 1) * (1-Y[15+i][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[i][pos[p+1]])))
                        # Create the constraint that the current value is larger than
                        # the values between it and the next largest value
                        # 0 checking:
                        if pos[p+1] - index - 1 != 0:
                            for j in range(index+1,pos[p+1]):
                                constraints.append((size + 1) * (1-Y[15+i][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[i][index])) > cp.sum_entries(cp.mul_elemwise(nums, X[i][j])))
            else:
                # If there are no next larger numbers, first number is max
                for j in range(1,size):
                    constraints.append((size + 1) * (1-Y[15+i][comb_index]) + cp.sum_entries(cp.mul_elemwise(nums, X[i][0])) > cp.sum_entries(cp.mul_elemwise(nums, X[i][j])))

    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.ECOS,max_iters=10)

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
        for i in range(size):
            Y_val = np.array(Y[i].value,ndmin=2, copy=False)
            output.append(np.append(Y_val,np.array(np.zeros((max_y_size-Y_val.shape[0],1)),ndmin=2,copy=False),axis=0))
        y_star = np.array(output)

        # Round values
        x_star[np.logical_and(x_star >= -.005, x_star <= .005)] = 0
        x_star[np.logical_and(x_star >= 1-.005, x_star <= 1+.005)] = 1

        y_star[np.logical_and(y_star >= -.005, y_star <= .005)] = 0
        y_star[np.logical_and(y_star >= 1-.005, y_star <= 1+.005)] = 1

        # Push the current solution onto the heapquene
        heapq.heappush(frontier, (prob.value+random.random(), x_star, [], y_star, []))
    else:
        heapq.heappush(frontier, (prob.value+random.random(), None, [], None, []))

    while(len(frontier) > 0):
        # Get the lowest cost solution
        item = heapq.heappop(frontier)

        print("")
        for i in range(size):
            for j in range(size):
                found1 = False
                for k in range(size):
                    if item[1][i][j][k] == 1:
                        print(k+1,end=" ")
                        found1 = True
                if not found1:
                    print(0,end=" ")
            print("")
        print("")


        # Check if the lowest cost solution is integer valued and exists
        if not(item[1] is None):
            x_done = np.all(np.logical_or(item[1] == 0, item[1] == 1))
            y_done = np.all(np.logical_or(item[3] == 0, item[3] == 1))
            if(x_done and y_done):
                # If it is, format an solution array, and then return it.
                solution = []
                for i in range(size):
                    solution_row = []
                    for j in range(size):
                        solution_row.append(np.nonzero(item[1][i][j])[0][0] + 1)
                    solution.append(solution_row)
                constraints_int = item[2]
                return (solution, constraints_int)

        # If no integer valued solution, find variable closest to .5
        closestX = (np.abs(item[1]-.5))
        closestY = (np.abs(item[3]-.5))
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
        for con in item[2]:
            new_constraints.append(X[con[0]-1][con[1]-1][con[2]-1] == con[3])
        for con in item[4]:
            new_constraints.append(Y[con[0]][con[1]] == con[2])

        # Solve the 0 and 1 constraint problems, and add them to the frontier
        for c in range(2):
            if(x_constraint):
                prob = cp.Problem(obj,constraints + new_constraints + [X[new_i][new_j][new_k] == c])
            else:
                prob = cp.Problem(obj,constraints + new_constraints + [Y[new_i][new_j] == c])
            prob.solve(solver=cp.ECOS)
            print(prob.status)

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
                for i in range(size):
                    Y_val = np.array(Y[i].value,ndmin=2, copy=False)
                    output.append(np.append(Y_val,np.array(np.zeros((max_y_size-Y_val.shape[0],1)),ndmin=2,copy=False),axis=0))
                y_star = np.array(output)

                # Round values
                x_star[np.logical_and(x_star >= -.005, x_star <= .005)] = 0
                x_star[np.logical_and(x_star >= 1-.005, x_star <= 1+.005)] = 1

                y_star[np.logical_and(y_star >= -.005, y_star <= .005)] = 0
                y_star[np.logical_and(y_star >= 1-.005, y_star <= 1+.005)] = 1

                # Push the current solution onto the heapquene
                if(x_constraint):
                    heapq.heappush(frontier,(prob.value+random.random(),x_star,item[2] + [(new_i+1, new_j+1, new_k+1, c)],y_star,item[4]))
                else:
                    heapq.heappush(frontier,(prob.value+random.random(),x_star,item[2],y_star,item[4] + [(new_i,new_j,c)]))
            else:
                # Push the current solution onto the heapquene
                if(x_constraint):
                    heapq.heappush(frontier,(prob.value+random.random(),None,item[2] + [(new_i+1, new_j+1, new_k+1, c)],None,item[4]))
                else:
                    heapq.heappush(frontier,(prob.value+random.random(),None,item[2],None,item[4] + [(new_i,new_j,c)]))


if __name__ == '__main__':
    main()
