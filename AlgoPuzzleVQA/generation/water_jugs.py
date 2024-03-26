import os
import json
import copy
import math
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.puzzle import Puzzle
import matplotlib.pyplot as plt


class JugFill(Puzzle):

    def __init__(self, pos, capacity, goal):
        self.pos = pos
        self.capacity = capacity
        self.goal = goal

    def __iter__(self):
        for i in range(len(self.pos)):
            for j in range(len(self.pos)):
                if i == j: 
                    continue
                qty = min(self.pos[i], self.capacity[j] - self.pos[j])
                if not qty: 
                    continue
                dup = list(self.pos)
                dup[i] -= qty
                dup[j] += qty

                yield JugFill(tuple(dup), self.capacity, self.goal)


def combination_sum(nums, target, k):
    result = []

    def dfs(i, current, total):
        if total == target and len(current) == k:
            result.append(current[:][::-1])
            return

        if i >= len(nums) or total > target:
            return

        current.append(nums[i])
        dfs(i, current, total + nums[i])
        current.pop()

        dfs(i + 1, current, total)

    dfs(0, [], 0)

    return result


def valid_configuration(quantity, total_jugs, capacities):
    assert total_jugs == len(capacities)
    nums = list(range(1, quantity + 1))
    results = []
    for num_jugs in range(total_jugs, 0, -1):
        combinations = combination_sum(nums, quantity, num_jugs)
        combinations = [c + [0] * (total_jugs - num_jugs) for c in combinations]
        results += combinations

    valid = []
    for result in results:
        for item in list(set(itertools.permutations(result))):
            decision = [q <= c for q, c in zip(item, capacities)]
            if False not in decision:
                valid.append(item)

    return valid


def draw_jugs(fig, ax, capacity, initial):
    
    num_jugs = len(capacity)
    max_capacity = max(capacity)

    # Draw each jug with a single y-axis
    for i in range(num_jugs):
        # Determine the position for each jug
        jug_x = 1 + 2 * i

        # Bottom, left and right borders (bold)
        ax.plot([jug_x - 0.8, jug_x + 0.8], [0, 0], color="black", lw=4)
        ax.plot([jug_x - 0.8, jug_x - 0.8], [0, capacity[i]], color="black", lw=2)
        ax.plot([jug_x + 0.8, jug_x + 0.8], [0, capacity[i]], color="black", lw=2)

        # Fill the jug with water
        ax.add_patch(plt.Rectangle((jug_x - 0.8, 0), 1.6, initial[i], color="blue", alpha=0.35))

        # Setting the labels for each jug
        ax.text(jug_x, max_capacity + 0.5, f"Jug {i+1}", horizontalalignment="center", fontsize=12)

    # Setting the limits, labels, and y-axis scale for the whole figure
    ax.set_xlim(0, 2 * num_jugs + 0.25)
    ax.set_ylim(0, max_capacity + 1)
    ax.set_ylabel("Quantity (litres)", fontsize=12)
    ax.set_yticks(range(0, max_capacity+1))
    ax.set_xticks([])  # Hide x-axis ticks

    ax.grid(linewidth=0.15)
    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":

    os.makedirs("data/images/water_jugs")

    instance_pool, instance_pool_set = [], set()

    print ("Generating puzzle candidates.")
    for _ in tqdm(range(20)):
        num_jugs = random.choice([3, 4, 5])
        capacities = random.sample(list(range(1, 15)), num_jugs)
        capacities.sort(reverse=True)

        largest_jug_quatity = capacities[0]
        total_jug_quantities = sum(capacities) 
        quantity = random.choice(list(range(largest_jug_quatity, total_jug_quantities + 1)))

        possible_starts = valid_configuration(quantity, num_jugs, capacities)
        possible_goals = valid_configuration(quantity, num_jugs, capacities)

        if len(possible_starts) > 20:
            possible_starts = random.sample(possible_starts, 20)
            possible_goals = random.sample(possible_goals, 20)

        for start in possible_starts:
            for goal in possible_goals:
                if start != goal:
                    try:
                        out = JugFill(start, capacities, goal).solve()
                        if tuple(out) not in instance_pool_set and len(out) <= 5:
                            instance_pool.append(
                                {
                                    "capacities": capacities,
                                    "start": start, "goal": goal,
                                    "steps": [state.pos for state in out]
                                }
                            )
                            instance_pool_set.add(tuple(out))
                    except:
                        continue

    data, question_index, num_instances = [], 0, 100

    pool5_len = num_instances // 3
    pool4_len = num_instances // 3
    pool3_len = num_instances - pool5_len - pool4_len

    random.shuffle(instance_pool)
    pool3, pool4, pool5 = [], [], []
    
    for item in instance_pool:
        if len(item["capacities"]) == 3 and len(pool3) < pool3_len:
            pool3.append(item)
        elif len(item["capacities"]) == 4 and len(pool4) < pool4_len:
            pool4.append(item)
        elif len(item["capacities"]) == 5 and len(pool5) < pool5_len:
            pool5.append(item)

    final_pool = pool3 + pool4 + pool5

    for j in tqdm(range(num_instances)):
        item = final_pool[j]

        jugs, capacities, start, goal = len(item["start"]), item["capacities"], list(item["start"]), list(item["goal"])
        answer = len(item["steps"]) - 1
        
        fig, ax = plt.subplots(figsize=(6, 6))
        fig, ax = draw_jugs(fig, ax, capacities, start)

        fname = f"data/images/water_jugs/water_jugs_{question_index:04}.jpg"
        plt.savefig(fname, bbox_inches="tight", dpi=250)
        plt.close(fig)

        question = f"You are given {jugs} jugs of capacities {', '.join([str(c) for c in capacities])} litres. Initially, the amount of water " + \
                   f"that is contained in each jar is shown in the image. A single step of water pouring from one jug to another is constrained " + \
                   f"by the following rules: i) take a non-empty jug and pour water from it to another non-full jug until the first one " + \
                   f"becomes empty or the second one becomes full; and ii) no water can be spilt while pouring. The objective is to reach " + \
                   f"the amounts of {', '.join([str(c) for c in goal])} litres of water in the jugs from left to right, respectively. What is " + \
                   f"the minimum number of water pouring steps required to achieve the objective?"

        assert sum(start) == sum(goal)

        example = {
            "image": fname[5:], "question": question, "answer": answer,
            "solution": {
                "num_jugs": jugs, "capacities": capacities, "total_water": sum(start),
                "start": start, "end": goal, "steps": item["steps"], "num_steps": answer
            }
        }

        data.append(example)
        question_index += 1


    with open("data/water_jugs.json", "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

