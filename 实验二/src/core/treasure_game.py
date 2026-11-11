from dataclasses import dataclass, replace
from .astar import search_grid

LABELS = ("A", "B", "T1", "T2", "T3", "T4")


@dataclass(frozen=True)
class State:
    a: int = 0
    b: int = 1
    remaining: int = 15
    score_a: int = 0
    score_b: int = 0
    player: int = 0


def distance_matrix(grid, positions):
    return [[0 if i == j else search_grid(grid, positions[i], positions[j], mode="treasure").cost
             for j in range(6)] for i in range(6)]


class TreasureGame:
    def __init__(self, matrix, values):
        if len(matrix) != 6 or any(len(row) != 6 for row in matrix):
            raise ValueError("需要6×6距离矩阵")
        if any(v < 0 for row in matrix for v in row):
            raise ValueError("基础综合题要求关键点两两可达")
        if len(values) != 4:
            raise ValueError("需要四个宝藏价值")
        self.matrix, self.values = matrix, tuple(values)
        self.initial = State()

    def is_max(self, s):
        return s.player == 0

    def terminal(self, s):
        return s.remaining == 0

    def evaluate(self, s):
        return s.score_a - s.score_b

    def gain(self, s, action):
        position = s.a if s.player == 0 else s.b
        return self.values[action] - self.matrix[position][action + 2]

    def actions(self, s, order):
        actions = [i for i in range(4) if s.remaining & (1 << i)]
        if order == "reverse":
            return actions[::-1]
        if order == "gain":
            return sorted(actions, key=lambda i: (-self.gain(s, i), i))
        if order != "number":
            raise ValueError("未知宝藏访问顺序")
        return actions

    def transition(self, s, action):
        if type(action) is not int or action not in range(4) or not s.remaining & (1 << action):
            raise ValueError("必须选择尚未获取的宝藏")
        gain = self.gain(s, action)
        common = {"remaining": s.remaining & ~(1 << action), "player": 1 - s.player}
        if s.player == 0:
            return replace(s, a=action + 2, score_a=s.score_a + gain, **common)
        return replace(s, b=action + 2, score_b=s.score_b + gain, **common)
