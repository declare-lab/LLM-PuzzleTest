import os
import json
import random
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d


# Some functions adopted from: https://stackoverflow.com/questions/42863543/applying-the-4-color-theorem-to-list-of-neighbor-polygons-stocked-in-a-graph-arr

# Four colour theorem
def solve(X, Y, solution):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        Xc = list(X[c])

        for r in Xc:
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()


def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


# Invert subset collection
def exact_cover(X, Y):
    newX = dict((j, set()) for j in X)
    for i, row in Y.items():
        for j in row:
            newX[j].add(i)
    return newX


def colour_map(nodes, edges, ncolours=4, fixed=2):
    colours = range(ncolours)

    #The edges that meet each node
    node_edges = dict((n, set()) for n in nodes)
    for e in edges:
        n0, n1 = e
        node_edges[n0].add(e)
        node_edges[n1].add(e)

    for n in nodes:
        node_edges[n] = list(node_edges[n])

    # Set to cover
    coloured_edges = list(product(colours, edges))
    X = nodes + coloured_edges

    # Subsets to cover X with
    Y = dict()
    # Primary rows
    for n in nodes:
        ne = node_edges[n]
        for c in colours:
            Y[(n, c)] = [n] + [(c, e) for e in ne]

    # Dummy rows
    for i, ce in enumerate(coloured_edges):
        Y[i] = [ce]

    X = exact_cover(X, Y)

    # Set first few nodes
    assert fixed <= 4
    partial = [(nodes[k], k) for k in range(fixed)]

    for s in partial:
        select(X, Y, s)

    for s in solve(X, Y, []):
        s = partial + [u for u in s if not isinstance(u, int)]
        s.sort()
        yield s


# Find Region Adjacency
def create_edges(region):
    edges = set()
    # Ensure the region forms a closed loop by connecting the last and first points
    points = region + [region[0]]
    for i in range(len(region)):
        # Create an edge as a sorted tuple to avoid directional issues
        edge = tuple(sorted((points[i], points[i+1])))
        edges.add(edge)
    return edges


def find_adjacencies(regions):
    # Create a list to hold the sets of edges for each region
    region_edges = [create_edges(region) for region in regions]
    
    adjacency_list = {}
    for i, edges1 in enumerate(region_edges):
        # Initialize the adjacency list for region i
        adjacency_list[i] = []
        for j, edges2 in enumerate(region_edges):
            if i != j and not edges1.isdisjoint(edges2):
                # If regions i and j share at least one edge, they are adjacent
                adjacency_list[i].append(j)
    return adjacency_list


# Clips a polygon defined by 'vertices' against a rectangular clipping window.
def clip_polygon(vertices, x_min, x_max, y_min, y_max):

    def inside(p, bound):
        if bound == 'left':
            return p[0] >= x_min
        if bound == 'right':
            return p[0] <= x_max
        if bound == 'bottom':
            return p[1] >= y_min
        if bound == 'top':
            return p[1] <= y_max

    def compute_intersection(p1, p2, bound):
        if bound in ['left', 'right']:
            x_bound = x_min if bound == 'left' else x_max
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            y = p1[1] + slope * (x_bound - p1[0])
            return [x_bound, y]
        elif bound in ['bottom', 'top']:
            y_bound = y_min if bound == 'bottom' else y_max
            slope = (p2[0] - p1[0]) / (p2[1] - p1[1])
            x = p1[0] + slope * (y_bound - p1[1])
            return [x, y_bound]

    def clip_against_bound(input_vertices, bound):
        output_vertices = []
        for i in range(len(input_vertices)):
            p1 = input_vertices[i]
            p2 = input_vertices[(i + 1) % len(input_vertices)]

            if inside(p2, bound):
                if not inside(p1, bound):
                    output_vertices.append(compute_intersection(p1, p2, bound))
                output_vertices.append(p2)
            elif inside(p1, bound):
                output_vertices.append(compute_intersection(p1, p2, bound))

        return output_vertices

    clipped_vertices = vertices[:]
    for boundary in ['left', 'right', 'bottom', 'top']:
        clipped_vertices = clip_against_bound(clipped_vertices, boundary)

    return np.array(clipped_vertices).round(5)


def choose_prefix(solution_ids, masked=4):
    prefix_counts = {}
    for item in solution_ids:
        prefix = item[:-(2*masked-1)]
        if prefix not in prefix_counts:
            count = 0
            for sol in solution_ids:
                if sol.startswith(prefix):
                    count += 1
            prefix_counts[prefix] = count

    inv_prefix_counts = {}
    for key, val in prefix_counts.items():
        if val in inv_prefix_counts:
            inv_prefix_counts[val].append(key)
        else:
            inv_prefix_counts[val] = [key]
    
    answer = -1

    if len(set(inv_prefix_counts.keys()).intersection(set([1, 2, 3, 4, 6, 8]))) == 0:
        raise ValueError
    
    while answer not in inv_prefix_counts:
        answer = random.choice([1, 2, 3, 4, 6, 8])
    
    scheme = random.choice(inv_prefix_counts[answer])
    unmasked_map = {}
    for k, region_colour in enumerate(scheme[:-1].split("-")):
        unmasked_map[k] = int(region_colour)

    masked_region_colours = []
    for item in solution_ids:
        prefix, suffix = item[:-(2*masked-1)], item[-(2*masked-1):]
        if prefix == scheme:
            masked_region_colours.append([int(id_) for id_ in suffix.split("-")])

    assert answer == len(masked_region_colours)

    return scheme, answer, masked_region_colours, unmasked_map


all_colours = [
    [1, 0.3, 0.3, 1],
    [0, 0.6, 0, 1],
    [0, 0.7, 1, 1],
    [1, 0.8, 0.1, 1],
    [1, 1, 1, 1],
]

colour_index = {
    0: "Red", 1: "Green", 2: "Blue", 3: "Yellow"
}

def create_instance(fig, ax):
    # Data points and Voronoi tesselation
    num_points = random.choice(list(range(15, 21)))
    points = np.random.rand(num_points, 2)
    points = np.append(points, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0)
    vor = Voronoi(points)
    
    # Find regions considering bounded Voronoi with (0, 1) x and y limits
    all_regions, all_vertices = [], {}
    for region in vor.regions:
        if not -1 in region and region != []:
            original_vertices = np.array([vor.vertices[i] for i in region])
            new_vertices = clip_polygon(original_vertices, 0, 1, 0, 1)
            new_region = []
            for vertex in new_vertices:
                if tuple(vertex) not in all_vertices:
                    all_vertices[tuple(vertex)] = len(all_vertices)
                new_region.append(all_vertices[tuple(vertex)])
            all_regions.append((region, new_region, new_vertices, original_vertices))
    
    random.shuffle(all_regions)
    
    # Find the adjacency list
    adjacency_list = find_adjacencies([item[1] for item in all_regions])
    adjacent_polygons_graph = []
    
    for region, adjacent_regions in adjacency_list.items():
        for neighbor in sorted(adjacent_regions):
            if (region, neighbor) not in adjacent_polygons_graph:
                adjacent_polygons_graph.append((region, neighbor))
    
    num_polygons = len(all_regions)
    input_polygons = [(k, None) for k in range(num_polygons)]
    
    # Extract the nodes list 
    nodes = [t[0] for t in input_polygons]
    
    # Solution of colour mapping
    # if num_points > 20:
    #    all_solutions = colour_map(nodes, adjacent_polygons_graph, ncolours=4, fixed=4)
    # else:
    all_solutions = colour_map(nodes, adjacent_polygons_graph, ncolours=4, fixed=3)
    
    solution_ids = []
    for count, solution in enumerate(all_solutions, start=1):
        solution.sort(key=lambda x: x[0])
        solution_ids.append("-".join([str(item[1]) for item in solution]))
    
    # Create masked instance
    num_masked = random.choice([2, 3, 4, 5, 6])
    scheme, answer, masked_region_colours, unmasked_map = choose_prefix(solution_ids, masked=num_masked)
    
    # Plot and Colourize
    voronoi_plot_2d(vor, ax, show_vertices=False, show_points=False, line_width=0.25)
    index, masked_polygons = 0, []
    for k, (_, _, polygon_vertices, _) in enumerate(all_regions):
        
        if k < num_polygons - len(masked_region_colours[0]):
            plt.fill(*zip(*polygon_vertices), facecolor=all_colours[unmasked_map[index]], linewidth=0.01)
        else:
            plt.fill(*zip(*polygon_vertices), facecolor=all_colours[4], linewidth=0.01)
    
        polygon = Polygon(polygon_vertices)
        centroid = polygon.centroid.coords[0]
        plt.text(centroid[0], centroid[1], k+1, ha='center', va='center')
        index += 1
        
    
    # Fix the range of axes
    plt.xlim([0,1]), plt.ylim([0,1])
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    start = max(unmasked_map.keys()) + 1
    possible_colours_of_unknown = []
    for sol in masked_region_colours:
        possible_colours_of_unknown.append({start+k+1: colour_index[sol[k]] for k in range(len(sol))})
        
    solution = {
        "polygon_vertices": {
            k+1: all_regions[k][2].tolist() for k in range(len(all_regions))
        },
        "polygon_adjacency_list": [(item[0] + 1, item[1] + 1) for item in adjacent_polygons_graph],
        "known_colours": {
            k+1: colour_index[v] for k, v in unmasked_map.items()
        },
        "possible_colours_of_unknown": possible_colours_of_unknown,
        "unique_maps": len(masked_region_colours),
        "total_regions": len(all_regions),
        "known_regions": [1 + k for k in sorted(list(unmasked_map.keys()))],
        "unknown_regions": sorted(list(possible_colours_of_unknown[0].keys()))
    }

    return solution


def create_context(solution):

    context = f"You are given an incomplete map of a country having {solution['total_regions']} different regions. " + \
            f"The objective is to colour the regions of the map using only the four available colours: red, green, blue and yellow, " + \
            f"such that no two adjacent regions have the same colour. Adjacent regions are defined as two regions that share a common " + \
            f"boundary of non-zero length. The regions indicated by numbers 1 to {solution['known_regions'][-1]} have already been coloured, " + \
            f"as shown in the image. The regions indicated by numbers {solution['unknown_regions'][0]} to {solution['unknown_regions'][-1]} " + \
            f"are shown in white as they are yet to be coloured. You need to assign colours to these regions in a way such that it doesn't " + \
            f"violate the objective. Each unique colour combination of the regions would result in a unique complete map. How many unique " + \
            f"complete maps can be created by colouring all the white regions starting from the given incomplete map?"

    return context


# In[5]:

if __name__ == "__main__":

    os.makedirs("data/images/map")

    data, question_index, num_instances = [], 0, 100
    progress_bar = tqdm(range(num_instances))
    solution_set = []

    while question_index < num_instances:
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            solution = create_instance(fig, ax)

            if solution not in solution_set:
                solution_set.append(solution)

                question = create_context(solution)

                fname = f"data/images/map/map_{question_index:04}.jpg"
                plt.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close(fig)

                example = {
                    "image": fname[5:], "question": question, "answer": solution["unique_maps"],
                    "solution": solution
                }
            
                data.append(example)
                question_index += 1
                progress_bar.update(1)

        except:
            continue


    with open("data/map.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

