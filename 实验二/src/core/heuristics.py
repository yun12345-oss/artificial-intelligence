def zero(node, goal):
    return 0


def manhattan(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def corrected(minimum):
    return lambda node, goal: minimum * max(manhattan(node, goal) - 1, 0)
