import heapq

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


TARGET = (200, 559)


class HeapEntry:
    def __init__(self, node, priority):
        self.node = node
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority


def traceback_path(target, parents):
    path = []
    while target:
        path.append(target)
        target = parents[target]
    return list(reversed(path))


def shortest_path(starts, finish, graph):
    visited = set()
    queue = []
    parents = {}
    distance = {}

    for start in starts:

        heap_entry = HeapEntry(start, 0.0)
        heapq.heappush(queue, heap_entry)
        parents[start] = None
        distance[start] = 0.0

    while queue:
        current = heapq.heappop(queue).node
        if current == finish:
            return traceback_path(finish, parents)

        if current in visited:
            continue

        visited.add(current)
        for neighbor_data in graph[current]:
            neighbor = neighbor_data[0]
            energy = neighbor_data[1]
            if neighbor in visited:
                continue
            tentative_cost = distance[current] + energy

            if neighbor not in distance.keys() or distance[neighbor] > tentative_cost:
                distance[neighbor] = tentative_cost
                parents[neighbor] = current
                heap_entry = HeapEntry(neighbor, tentative_cost)
                heapq.heappush(queue, heap_entry)


def learn_model(x, y):
    polyf = PolynomialFeatures(degree=4)
    x_poly = polyf.fit_transform(x)
    poly_reg = LinearRegression()
    poly_reg.fit(x_poly, y)

    return poly_reg, polyf


def get_predictor(x, y, poly_reg, polyf):
    eps = 0.01
    min_gradient = float(x[0])
    max_gradient = float(x[-1])
    r_est_coef = (
        float(
            poly_reg.predict(polyf.fit_transform([x[-1]]))
            - poly_reg.predict(polyf.fit_transform([x[-1] - eps]))
        )
        / eps
    )
    l_est_coef = (
        float(
            poly_reg.predict(polyf.fit_transform([x[0] + eps]))
            - poly_reg.predict(polyf.fit_transform([x[0]]))
        )
        / eps
    )
    left_est_value = float(poly_reg.predict(polyf.fit_transform([x[0]])))
    right_est_value = float(poly_reg.predict(polyf.fit_transform([x[-1]])))

    def predict(gradient):
        if gradient < min_gradient:
            return (min_gradient - gradient) * l_est_coef + left_est_value
        if gradient > max_gradient:
            return (gradient - max_gradient) * r_est_coef + right_est_value
        return float(poly_reg.predict(polyf.fit_transform([[gradient]])))

    return predict


def construct_graph(mat, n, m, predict):
    dx = [-1, 1, 0, 0]
    dy = [0, 0, 1, -1]
    graph = {}
    for i in range(n):
        for j in range(m):
            graph[(i, j)] = []
            for k in range(4):
                nx = i + dx[k]
                ny = j + dy[k]
                if -1 < nx < n and -1 < ny < m:
                    gradient = (mat[nx, ny] - mat[i, j]) / 10
                    w = predict(gradient)
                    graph[(i, j)].append([(nx, ny), w])

    return graph


def display_instructions(path):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    current_dir = 0
    navigate = 0
    for i in range(1, len(path)):
        dist_diff = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        diff_index = directions.index(dist_diff)
        if diff_index - current_dir == 0:
            navigate += 1
        else:
            print("Walk ", navigate * 10, "m")
            navigate = 1
            if diff_index - current_dir == 1:
                print("Turn Right")
            else:
                print("Turn Left")
            current_dir = diff_index

    print("Walk ", navigate * 10, "m")


if __name__ == "__main__":
    dataset = pd.read_csv("energy_cost.csv")
    x = dataset.iloc[:, 0].values.reshape(-1, 1)
    y = dataset.iloc[:, -1].values.reshape(-1, 1)

    data = [(x, y) for (x, y) in zip(x, y)]
    data = sorted(data)
    x = [x for (x, _) in data]
    y = [y for (_, y) in data]

    poly_reg, polyf = learn_model(x, y)
    predict = get_predictor(x, y, poly_reg, polyf)

    alt_map = pd.read_csv("altitude_map.csv", header=None)
    mat = alt_map.values
    n, m = mat.shape
    graph = construct_graph(mat, n, m, predict)

    starts = []
    for x in range(n):
        starts.append((x, 0))
    path = shortest_path(starts, TARGET, graph)

    display_instructions(path)

    x_path, y_path = zip(*path)
    pd.DataFrame(list(zip(x_path, y_path)), columns=["x_coor", "y_coor"]).to_csv(
        "path.csv", index=False
    )

    X, Y = np.meshgrid(np.arange(0, m, 1), np.arange(0, n, 1))
    plt.contour(X, Y, alt_map)
    plt.title("Contour Plot")
    plt.plot(x_path, y_path, linestyle="-", marker="o", markersize=2)
    plt.plot(
        x_path[0],
        y_path[0],
        marker="o",
        markersize=20,
        markerfacecolor="green",
        markeredgecolor="red",
    )
    plt.plot(
        x_path[-1],
        y_path[-1],
        marker="o",
        markersize=20,
        markerfacecolor="yellow",
        markeredgecolor="red",
    )
    plt.savefig("path.png")
