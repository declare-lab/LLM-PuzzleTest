import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from queue import Queue
import matplotlib.pyplot as plt


def create_maze(x_dim, y_dim):
    # Create a grid filled with walls
    maze = np.ones((x_dim*2+1, y_dim*2+1))
    # maze = np.ones((x_dim, y_dim))

    # Define the starting point
    x, y = (0, 0)
    maze[2*x+1, 2*y+1] = 0

    # Initialize the stack with the starting point
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < x_dim and ny < y_dim and maze[2*nx+1, 2*ny+1] == 1:
                maze[2*nx+1, 2*ny+1] = 0
                maze[2*x+1+dx, 2*y+1+dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()
            
    # Create an entrance and an exit
    maze[1, 0] = 0
    if random.uniform(0, 1) < 0.5:
        maze[-2, -1] = 0
    else:
        maze[-1, -2] = 0

    return maze


def find_path(maze):
    # BFS algorithm to find the shortest path
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = (1, 1)
    end = (maze.shape[0]-2, maze.shape[1]-2)
        
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))
    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0]+dx, node[1]+dy)
            if (next_node == end):
                return path + [next_node]
            if (next_node[0] >= 0 and next_node[1] >= 0 and 
                next_node[0] < maze.shape[0] and next_node[1] < maze.shape[1] and 
                maze[next_node] == 0 and not visited[next_node]):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))


def draw_line(ax, start, end):
    # Function to draw a single grid line
    ax.plot([start[1], end[1]], [start[0], end[0]], color="grey", linewidth=0.12)


def draw_maze(maze, fig, ax, path=None):
    # Set the border color to white
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Set the border color of the ax.imshow to be white
    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    
    # Iterate over the maze array and draw grid lines between cells with value 0
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 0:
                # Draw top border
                if i == 0 or maze[i-1, j] == 0:
                    draw_line(ax, (i-0.5, j-0.5), (i-0.5, j+0.5))
                # Draw left border
                if j == 0 or maze[i, j-1] == 0:
                    draw_line(ax, (i-0.5, j-0.5), (i+0.5, j-0.5))
                # Draw bottom border
                if i == maze.shape[0]-1 or maze[i+1, j] == 0:
                    draw_line(ax, (i+0.5, j-0.5), (i+0.5, j+0.5))
                # Draw right border
                if j == maze.shape[1]-1 or maze[i, j+1] == 0:
                    draw_line(ax, (i-0.5, j+0.5), (i+0.5, j+0.5))

        
    # Draw the solution path if it exists
    if path is not None:
        x_coords = [x[1] for x in path]
        y_coords = [y[0] for y in path]
        ax.plot(x_coords, y_coords, color='red', linewidth=2)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw entry and exit arrows
    ax.arrow(-0.5, 1, .5, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)

    if maze[-2, -1] == 0:
        ax.arrow(
            maze.shape[1] - 1.4, maze.shape[0] - 2, 0.55, 0, fc='blue', ec='blue', head_width=0.3, head_length=0.3
        )
    elif maze[-1, -2] == 0:
        ax.arrow(
            maze.shape[1] - 1.99, maze.shape[0]-0.3  - 1, 0.0, 0.5, fc='blue', ec='blue', head_width=0.3, head_length=0.3
        )
    
    # plt.show()


def count_turns(path):
    # Directions represented as (row_change, column_change)
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    # Mapping of current direction to next direction for left and right turns
    left_turns_map = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}
    right_turns_map = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}

    def get_direction(p1, p2):
        # Determine the direction of movement between two points
        row_change = p2[0] - p1[0]
        column_change = p2[1] - p1[1]
        return (row_change, column_change)

    left_turns = 0
    right_turns = 0
    current_direction = None
    all_turns = []

    for i in range(len(path) - 1):
        next_direction = get_direction(path[i], path[i+1])

        if current_direction:
            # Check for left turn
            if left_turns_map[current_direction] == next_direction:
                left_turns += 1
                all_turns.append("left")
            # Check for right turn
            elif right_turns_map[current_direction] == next_direction:
                right_turns += 1
                all_turns.append("right")
        
        current_direction = next_direction

    return left_turns, right_turns, all_turns


if __name__ == "__main__":

    os.makedirs("data/images/maze")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))
    solution_set = []

    context = f"The empty cells are coloured white and the obstacle cells " + \
            f"are coloured black. From an empty cell, you can only move up, down, left, or right to another adjacent empty " + \
            f"cell. You cannot move diagonally between two empty cells and cannot step into a cell with an obstacle. " + \
            f"The entry cell of the maze is shown with the green arrow. The exit cell of the maze is shown with the " + \
            f"blue arrow. Suppose you have found the most optimal path in the maze between the entrance and exit, " + \
            f"where you need to go through the least number of empty cells and you need to make the least number of " + \
            f"left and right turns."

    while question_index < num_instances:
        x, y = random.choice([3, 4, 5, 6]), random.choice([3, 4, 5, 6])
        maze = create_maze(x, y)
        path = find_path(maze)
        path = [(1, 0), (1, 1)] + path
        if maze[-2, -1] == 0:
            path += [(len(maze) - 2, len(maze[0]) - 1)]
        elif maze[-1, -2] == 0:
            path += [(len(maze) - 1, len(maze[0]) - 2)]
            
        turns = count_turns(path)
        z = random.uniform(0, 1)
        
        if z < 0.2:
            query = "What is the total number of left turns do you need to make in this optimal path?"
            answer = turns[0]
        elif z < 0.4:
            query = "What is the total number of right turns do you need to make in this optimal path?"
            answer = turns[1]
        elif z < 0.6:
            query = "What is the combined number of left and right turns do you need to make in this optimal path?"
            answer = turns[0] + turns[1]
        else:
            query = "How many cells do you need to visit in this optimal path including the entrance and exit cells?"
            answer = len(path)

        question = f"This is maze having {2*x+1} * {2*y+1} cells. {context} {query}"

        solution = {
            "maze": maze.astype(int).tolist(),
            "optimal_path": path, "optimal_path_length": len(path),
            "left_turns": turns[0], "right_turns": turns[1],
            "total_turns": sum(turns[:2])
        }

        if solution not in solution_set:
            solution_set.append(solution)

            fig, ax = plt.subplots(figsize=(10, 10))
            draw_maze(maze, fig, ax)
            fname = f"data/images/maze/maze_{question_index:04}.jpg"
            plt.savefig(fname, dpi=300)
            plt.close(fig)

            example = {
                "image": fname[5:], "question": question, "answer": answer,
                "solution": solution
            }

            data.append(example)
            question_index += 1
            progress_bar.update(1)


    with open("data/maze.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

