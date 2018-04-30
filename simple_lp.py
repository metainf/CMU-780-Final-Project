import numpy as np
import cvxpy as cp
from itertools import combinations
import heapq


def main():
    puzzle = [2,2,5]
    solve_tower(puzzle)

def solve_tower(puzzle):
    # Puzzles are in the format:
    # Row 1: Top row
    # Row 2: Right columns
    # Row 3: Bottom columns
    # Row 4: Left row

    left = puzzle[0]
    right = puzzle[1]

    size = puzzle[2]

    # Vector of 1-n constant
    nums = np.transpose(np.arange(1,size+1))

    # Create the 1 by n by n boolean values
    X = []
    for i in range(size):
        X.append(cp.Variable(size))

    X_hat = []
    for i in range(size):
        X_hat.append(cp.Variable())

    # Create the objective to solve
    obj = cp.Minimize(sum([sum(x) for x in X]))

    # Create the constraints
    constraints = []

    # Create the between zero and one constraint
    for i in range(size):
        constraints.append(X[i][:] <= 1)
        constraints.append(X[i][:] >= 0)

    # Create the only one assigned number constraint
    for i in range(size):
        constraints.append(cp.sum_entries(X[i]) == 1)

    constraints.append(sum([X[i] for i in range(size)]) == 1)

    # Link X and X-X_hat
    for i in range(size):
        constraints.append(X_hat[i] == cp.sum_entries(cp.mul_elemwise(nums, X[i])))

    # Create the left vision number constraint
    comb = list(combinations([i for i in range(1,size)],left - 1))
    print(comb)
    # Create the boolean variables to account for the values
    Y = cp.Variable(len(comb))
    # Create the between 0 and 1 constraint
    for k in range(len(comb)):
        constraints.append(Y[k] <= 1)
        constraints.append(Y[k] >= 0)

    # Create the only one on constraint
    constraints.append(cp.sum_entries(Y) == 1)

    for comb_index in range(len(comb)):
        pos = comb[comb_index]
        print(pos)
        # Create the set of inequalities, starting with the first element
        # Check if there is a next largest value:
        if len(pos) != 0:
            # Create the constraint that the value is less than the next largest value
            constraints.append(X_hat[0] < X_hat[pos[0]] + (size + 1) * (1-Y[comb_index]))

            # Create the constraint that the current value is larger than
            # the values between it and the next largest value
            for i in range(1,pos[0]):
                constraints.append((size + 1) * (1-Y[comb_index]) + X_hat[0] > X_hat[i])

            # Create the rest of the inequalities
            for p in range(len(pos)):
                index = pos[p]
                # Check if we're at the last largest value
                if p == len(pos)-1:
                    # Create the constraint that the current value is greater than the rest of the values
                    # 0 checking:
                    for i in range(index+1,size):
                        constraints.append((size + 1) * (1-Y[comb_index]) + X_hat[index] > X_hat[i])
                else:
                    # Create the constraint that the current value is less than the next largest value
                    constraints.append(X_hat[index] < (size + 1) * (1-Y[comb_index]) + X_hat[pos[p+1]])
                    # Create the constraint that the current value is larger than
                    # the values between it and the next largest value
                    for i in range(index+1,pos[p+1]):
                        constraints.append((size + 1) * (1-Y[comb_index]) + X_hat[index] > X_hat[i])
        else:
            # If there are no next larger numbers, first number is max
            for i in range(1,size):
                constraints.append((size + 1) * (1-Y[comb_index]) + X_hat[0] > X_hat[i])

    prob = cp.Problem(obj,constraints)
    prob.solve()

    # Create the frontier as a heapquene
    frontier = []

    # Check if the problem was infeasible
    if not(prob.status == 'infeasible'):
        # Turn the solution into a numpy array
        output = []
        for i in range(size):
            output.append(X[i].value)
        x_star = np.array(output)

        y_star = np.array(Y.value)

        # Round values
        x_star[np.logical_and(x_star >= -.005, x_star <= .005)] = 0
        x_star[np.logical_and(x_star >= 1-.005, x_star <= 1+.005)] = 1

        # Round values
        y_star[np.logical_and(y_star >= -.005, y_star <= .005)] = 0
        y_star[np.logical_and(y_star >= 1-.005, y_star <= 1+.005)] = 1

        # Push the current solution onto the heapquene
        heapq.heappush(frontier, (prob.value, x_star, [], y_star,[]))
    else:
        heapq.heappush(frontier, (prob.value, None, [], None,[]))

    while(len(frontier) > 0):
        # Get the lowest cost solution
        item = heapq.heappop(frontier)

        # Check if the lowest cost solution is integer valued and exists
        if not(item[1] is None):
            if(np.all(np.logical_or(item[1] == 0, item[1] == 1)) and np.all(np.logical_or(item[3] == 0, item[3] == 1))):
                # If it is, format an solution array, and then return it.
                for i in range(5):
                    for j in range(5):
                        if item[1][i][j] == 1:
                            print(j+1)
                print(Y.value)
                exit()

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

        # Get the current constraint set
        new_constraints = []
        for con in item[2]:
            new_constraints.append(X[con[0]-1][con[1]-1] == con[2])
        for con in item[4]:
            new_constraints.append(Y[con[0]] == con[1])

        # Solve the 0 and 1 constraint problems, and add them to the frontier
        for c in range(2):
            if(x_constraint):
                prob = cp.Problem(obj, constraints + new_constraints + [X[new_i][new_j] == c])
            else:
                prob = cp.Problem(obj, constraints + new_constraints + [Y[new_i] == c])
            prob.solve(solver=cp.ECOS,max_iters=10)

            # Check if the problem was infeasible
            if not(prob.status == 'infeasible'):
                # Turn the solution into a numpy array
                output = []
                for i in range(size):
                    output.append(X[i].value)
                x_star = np.array(output)
                y_star = np.array(Y.value)

                # Round values
                x_star[np.logical_and(x_star >= -.005, x_star <= .005)] = 0
                x_star[np.logical_and(x_star >= 1-.005, x_star <= 1+.005)] = 1

                # Round values
                y_star[np.logical_and(y_star >= -.005, y_star <= .005)] = 0
                y_star[np.logical_and(y_star >= 1-.005, y_star <= 1+.005)] = 1

                print(x_star)
                print("")

                # Push the current solution onto the heapquene
                if(x_constraint):
                    heapq.heappush(frontier, (prob.value, x_star, item[2] + [(new_i+1, new_j+1, c)],y_star,item[4]))
                else:
                    heapq.heappush(frontier, (prob.value, x_star, item[2],y_star,item[4] + [(new_i,c)]))
            else:
                if(x_constraint):
                    heapq.heappush(frontier, (prob.value, None, item[2] + [(new_i+1, new_j+1, c)],None,item[4]))
                else:
                    heapq.heappush(frontier, (prob.value, None, item[2],None,item[4] + [(new_i,c)]))


if __name__ == '__main__':
    main()
