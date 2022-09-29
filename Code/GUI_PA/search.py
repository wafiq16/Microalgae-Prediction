def bfs(graph, start):
    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return visited


num_step = []
step = []
finish = False


def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
        print(visited)
    return visited


def dfs_paths(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))


def bfs(graph, start):
    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
            print(vertex)
    return visited


def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    step = 0
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))
            # print(next)


def get_number(input):
    node = list()
    c = 0
    for i in input:
        c += 1
        if i == 1:
            node.append(c)
    return node


def get_price(path, bobot, dim=9):
    price = 0
    step = len(path)-2
    print(step)
    # print(bobot)
    for i in range(step):
        price += bobot[path[i]-1][path[i+1]-1]
        # print(bobot[path[i]-1][path[i+1]-1])
        # print(path[i])
        # print(path[i+1])
    # print(list(comb))
    # for i in path:
    # price = bobot[i][len(path)-1-i]
    # print(i)
    # print(price)
    return price


city = {0: "Mall SMS", 1: "Kalideres", 2: "Bintaro",
        3: "Pluit", 4: "Ancol", 5: "BSD", 6: "Depok", 7: "Monas"}

graph_d = {0: set([1, 5, 2]),
           1: set([3, 5]),
           2: set([3, 7]),
           3: set([4, 7, 6, 7]),
           4: set([6, 7]),
           5: set([3, 6, 7]),
           6: set([7]),
           7: set([]), }

graph_jarak = {0: set([18, 22, 35]),
               1: set([42, 12]),
               2: set([21, 43]),
               3: set([35, 34, 11, 34]),
               4: set([38, 9]),
               5: set([31, 23, 57]),
               6: set([30]),
               7: set([]), }


print(list(bfs_paths(graph_d, 0, 7))[0])
# print(get_price(list(bfs_paths(graph_d, 0, 7))[0], graph_jarak))
