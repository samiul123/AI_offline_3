import itertools
from random import randint
import random
import timeit
import sys
import math
from boltons.setutils import IndexedSet
from collections import defaultdict

MIN = 0
MAX = 500


def generate_input(n, filename):
    with open(filename, "w") as f:
        f.write("{n}\n".format(n=n))
        for _ in itertools.repeat(None, n):
            f.write("{p0} {p1}\n".format(p0=randint(MIN, MAX), p1=randint(MIN, MAX)))


# if __name__ == "__main__":
#     # Write the input files for the optimal solution first.
#     n = 6
#     generate_input(n, "tsp6.txt")
#     n = 9
#     generate_input(n, "tsp9.txt")
#     n = 10
#     generate_input(n, "tsp10.txt")
#     # Then write the input files for nearest neighbor.
#     n = 1500
#     generate_input(n, "tsp1500.txt")
#     n = 3000
#     generate_input(n, "tsp3000.txt")
#     n = 9000
#     generate_input(n, "tsp9000.txt")

number_of_nodes = 0
sol_nn, sol_ni, sol_ci = [], [], []


def get(filename):
    data = []
    global number_of_nodes
    with open(filename, "r") as f:
        number_of_nodes = f.readline()
        for line in f.readlines():
            point = line.split()
            tuple = (point[0], point[1])
            # print(tuple)
            data.append(tuple)
        return data


def closest_point(point, route):
    d_min = float("inf")
    for p in route:
        d = math.sqrt((int(point[0]) - int(p[0]))**2 + (int(point[1]) - int(p[1]))**2)
        if d < d_min:
            d_min = d
            closest = p
    return closest, d_min


def nearest_neighbour(filename):
    point, *route = get(filename)
    print("Starting point: " + str(point))
    path = [point]
    sum = 0
    while len(route) >= 1:
        closest, dist = closest_point(path[-1], route)
        path.append(closest)
        route.remove(closest)
        sum += dist
    # Go back the the beginning when done.
    closest, dist = closest_point(path[-1], [point])
    path.append(closest)
    sum += dist
    global sol_nn
    sol_nn = path
    print("Optimal route using Nearest Neighbour:", path)
    print("Length:", sum)


def distances(given_node, remaining_nodes):
    distance_dict = {}
    from_to = {}
    for node in remaining_nodes:
        distance = math.sqrt((int(given_node[0]) - int(node[0]))**2 + (int(given_node[1]) - int(node[1]))**2)
        distance_dict[node] = distance
    from_to[given_node] = distance_dict
    return from_to


def find_insert_node(partial_tour, nodes):
    remaining_nodes = IndexedSet(nodes) - IndexedSet(partial_tour)
    min_node = defaultdict(dict)
    d_dict = {}
    for i in range(len(partial_tour)):
        d_dict.update(distances(partial_tour[i], remaining_nodes))
        node = min(d_dict[partial_tour[i]], key=d_dict[partial_tour[i]].get)
        min_node[partial_tour[i]]["node"] = node
        min_node[partial_tour[i]]["value"] = d_dict[partial_tour[i]][node]
    dist = []
    for i in range(len(partial_tour)):
        dist.append(min_node[partial_tour[i]]["value"])
    min_dist = min(dist)
    final_min_node = tuple()
    closest_to = tuple()
    for k, v in min_node.items():
        if v["value"] == min_dist:
            final_min_node = v["node"]
            closest_to = k
            break
    print("closest_to: " + str(closest_to))
    return final_min_node, min_dist


def distance(first_node, second_node):
    return math.sqrt((int(first_node[0]) - int(second_node[0]))**2 + (int(first_node[1]) - int(second_node[1]))**2)


def get_insertion_metric(first_node, middle_node, last_node):
    return distance(first_node, middle_node) + distance(middle_node, last_node) - distance(first_node, last_node)


insert_index = 1


def insert(partial_tour, node_to_inserted):
    global insert_index
    metrics = []
    print(partial_tour)
    minimized_metric = get_insertion_metric(partial_tour[0], node_to_inserted, partial_tour[1])
    print(str(partial_tour[0]) + " " + str(partial_tour[1]))
    print("initial metric: " + str(minimized_metric))
    print("node to be inserted: " + str(node_to_inserted))
    for i in range(0, len(partial_tour) - 1):
        metric = get_insertion_metric(partial_tour[i], node_to_inserted, partial_tour[i + 1])
        print("metric: " + str(metric))
        metrics.append(metric)
        if metric < minimized_metric:
            print("kom")
            minimized_metric = metric
            insert_index = i + 1
    partial_tour.insert(insert_index, node_to_inserted)
    print(partial_tour)
    return partial_tour

# def ni_initialize():


def nearest_insertion(filename):
    nodes = get(filename)
    # print("nodes: " + str(nodes))
    # starting_node = random.choice(tuple(nodes))
    starting_node = nodes[0]
    # print("starting_node: " + str(starting_node))
    partial_tour = [starting_node]
    # print("partial_tour: " + str(partial_tour))
    remaining_nodes = IndexedSet(nodes) - IndexedSet(partial_tour)
    # print("remaining_nodes: " + str(remaining_nodes))
    distances_from_given_node = distances(starting_node, remaining_nodes)
    # print("distances : " + str(distances_from_given_node))
    next_node = min(distances_from_given_node[starting_node], key=distances_from_given_node[starting_node].get)
    # print("next_node: " + str(next_node))
    partial_tour.append(next_node)
    # partial_tour.append(starting_node)

    # print("After adding next node partial tour: " + str(partial_tour))
    while len(partial_tour) < len(nodes):
        min_node, min_dist = find_insert_node(partial_tour, nodes)
        # print("min_dist: " + str(min_dist))
        print("min_node: " + str(min_node))
        partial_tour = insert(partial_tour, min_node)
        # print("Insert_index: " + str(insert_index))
        # print("After inserting " + str(min_node) + " partial_tour: " + str(partial_tour))
    solution = partial_tour + [partial_tour[0]]
    print("Optimal route using Nearest Insertion: " + str(solution))
    global sol_ni
    sol_ni = solution
    length = 0
    for i in range(len(solution) - 1):
        current_node = solution[i]
        next_node = solution[i + 1]
        length += distance(current_node, next_node)
    print("Length: " + str(length))


def cheapest_insertion(filename):
    nodes = get(filename)
    # print("nodes: " + str(nodes))
    # starting_node = random.choice(tuple(nodes))
    starting_node = nodes[0]
    # print("starting_node: " + str(starting_node))
    partial_tour = [starting_node]
    # print("partial_tour: " + str(partial_tour))
    remaining_nodes = IndexedSet(nodes) - IndexedSet(partial_tour)
    # print("remaining_nodes: " + str(remaining_nodes))
    distances_from_given_node = distances(starting_node, remaining_nodes)
    # print("distances : " + str(distances_from_given_node))
    next_node = min(distances_from_given_node[starting_node], key=distances_from_given_node[starting_node].get)
    # print("next_node: " + str(next_node))
    partial_tour.append(next_node)
    insert_index = 1
    while len(partial_tour) < len(nodes):
        for node in IndexedSet(nodes) - IndexedSet(partial_tour):
            minimized_metric = get_insertion_metric(partial_tour[0], node, partial_tour[1])
            for i in range(0, len(partial_tour) - 1):
                metric = get_insertion_metric(partial_tour[i], node, partial_tour[i + 1])
                if metric < minimized_metric:
                    minimized_metric = metric
                    insert_index += 1
            partial_tour.insert(insert_index, node)

    solution = partial_tour + [partial_tour[0]]
    print("Optimal route using Cheapest Insertion: " + str(solution))
    global sol_ci
    sol_ci = solution
    length = 0
    for i in range(len(solution) - 1):
        current_node = solution[i]
        next_node = solution[i + 1]
        length += distance(current_node, next_node)
    print("Length: " + str(length))


# def optimize2opt(solution, number_of_nodes, string_):
#     best = 0
#     best_move = None
#     for ci in range(0, number_of_nodes):
#         for xi in range(0, number_of_nodes):
#             yi = (ci + 1) % number_of_nodes
#             zi = (xi + 1) % number_of_nodes
#             c = solution[ci]
#             y = solution[yi]
#             x = solution[xi]
#             z = solution[zi]
#             cy = distance(c, y)
#             xz = distance(x, z)
#             cx = distance(c, x)
#             yz = distance(y, z)
#             if xi != ci and xi != yi:
#                 gain = (cy + xz) - (cx + yz)
#                 if gain > best:
#                     best_move = (ci, yi, xi, zi)
#                     best = gain
#     if best_move is not None:
#         (ci, yi, xi, zi) = best_move
#         new_solution = [0 for i in range(0, number_of_nodes)]
#         new_solution[0] = solution[ci]
#         n = 1
#         while xi != yi:
#             new_solution[n] = solution[xi]
#             n = n + 1
#             xi = (xi - 1) % number_of_nodes
#         new_solution[n] = solution[yi]
#         n = n + 1
#         while zi != ci:
#             new_solution[n] = solution[zi]
#             n = n + 1
#             zi = (zi + 1) % number_of_nodes
#         print("Improved 2opt for " + string_ + ": " + str(new_solution))
#         length = 0
#         for i in range(len(new_solution) - 1):
#             current_node = new_solution[i]
#             next_node = new_solution[i + 1]
#             length += distance(current_node, next_node)
#         print("Length: " + str(length))
#     else:
#         print("Improved 2opt for " + string_ + ": No new solution")


def swap_2opt(route, i, k):
    assert i >= 0 and i < (len(route) - 1)
    assert k > i and k < len(route)
    new_route = route[0:i]
    # print(new_route)
    # print(i)
    new_route.extend(reversed(route[i:k + 1]))
    new_route.extend(route[k+1:])
    assert len(new_route) == len(route)
    # print(route[0])
    # print(new_route)
    return new_route


def route_distance(route):
    dist = 0
    prev = route[-1]
    for node in route:
       dist += distance(prev, node)
       prev = node
    return dist


def optimise_2opt(route, str_):
    improvement = True
    best_route = route
    best_distance = route_distance(route)
    while improvement:
        improvement = False
        for i in range(len(best_route) - 1):
            for k in range(i + 1, len(best_route)):
                new_route = swap_2opt(best_route, i, k)
                new_distance = route_distance(new_route)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_route = new_route
                    improvement = True
                    break  # improvement found, return to the top of the while loop
            if improvement:
                break
    assert len(best_route) == len(route)
    print("Improved 2opt for " + str_ + ": " + str(best_route))
    print("Length: " + str(route_distance(best_route)))
    return best_route


def swap_n_reverse(tour, i, j, k):
    a, b, c, d, e, f = tour[i - 1], tour[i], tour[j - 1], tour[j], tour[k - 1], tour[k % len(tour)]
    original_dist = distance(a, b) + distance(c, d) + distance(e, f)
    comb1 = distance(a, c) + distance(b, d) + distance(e, f)
    comb2 = distance(a, b) + distance(c, e) + distance(d, f)
    comb3 = distance(a, d) + distance(e, b) + distance(c, f)
    comb4 = distance(f, b) + distance(c, d) + distance(e, f)
    if original_dist > comb1:
        tour[i: j] = reversed(tour[i: j])
        return comb1 - original_dist
    elif original_dist > comb2:
        tour[j: k] = reversed(tour[j: k])
        return comb2 - original_dist
    elif original_dist > comb4:
        tour[i: k] = reversed(tour[i: k])
        return comb4 - original_dist
    elif original_dist > comb3:
        tour[i: j], tour[j: k] = tour[j: k], tour[i: j]
        return comb3 - original_dist
    return original_dist


def optimize_3opt(tour, st):
    tour_length = len(tour)
    new_distance = 0
    for (a, b, c) in region_combination(tour_length):
        new_distance = swap_n_reverse(tour, a, b, c)
    if new_distance < 0:
        return optimize_3opt(tour, st)
    length = 0
    for i in range(len(tour) - 1):
        current_node = tour[i]
        next_node = tour[i + 1]
        length += distance(current_node, next_node)
    print("Improved 3opt for " + st + " : " + str(tour))
    print("Length: " + str(length))


def region_combination(N):
    return [(i, j, k) for i in range(N)
                        for j in range(i + 2, N)
                        for k in range(j + 2, N + (i > 0))]


filename = "tsp9.txt"
str_sol_nn = "Nearest Neighbour"
str_sol_ni = "Nearest Insertion"
str_sol_ci = "Cheapest Insertion"
nearest_insertion(filename)
nearest_neighbour(filename)
cheapest_insertion(filename)
optimise_2opt(sol_ci, str_sol_ci)
# optimize2opt(sol_ci, int(number_of_nodes), str_sol_ci)
# print(sol_nn)
# print(number_of_nodes)
optimize_3opt(sol_ci, str_sol_ci)