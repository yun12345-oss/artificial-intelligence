from dataclasses import dataclass, field
import hashlib
import random


@dataclass
class TreeNode:
    value: int | None = None
    children: list = field(default_factory=list)


def from_dict(data):
    return TreeNode(data.get("value"), [from_dict(x) for x in data.get("children", [])])


def to_dict(node):
    return {"children": [to_dict(c) for c in node.children]} if node.children else {"value": node.value}


def generated_tree(branches, depth, seed):
    rng = random.Random(seed)
    def build(level):
        return TreeNode(rng.randint(-20, 20)) if level == depth else TreeNode(children=[build(level + 1) for _ in range(branches)])
    return build(0)


class TreeProblem:
    """预先保存按结构路径生成的随机排列，不受搜索是否剪枝影响。"""
    def __init__(self, root, seed=20261101):
        self.root = root
        self.nodes, self.permutations = {}, {}
        def register(node, path):
            if not node.children and (type(node.value) is not int):
                raise ValueError("叶子必须提供整数value")
            self.nodes[path] = node
            indices = list(range(len(node.children)))
            digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
            random.Random(int.from_bytes(digest, "big")).shuffle(indices)
            self.permutations[path] = indices
            for i, child in enumerate(node.children):
                register(child, path + "/" + str(i))
        register(root, "root")

    @property
    def initial(self):
        return "root"

    def is_max(self, s):
        return s.count("/") % 2 == 0

    def terminal(self, s):
        return not self.nodes[s].children

    def evaluate(self, s):
        return self.nodes[s].value

    def actions(self, s, order):
        actions = list(range(len(self.nodes[s].children)))
        if order == "random":
            return list(self.permutations[s])
        if order == "reverse":
            return actions[::-1]
        if order != "number":
            raise ValueError("未知树访问顺序")
        return actions

    def transition(self, s, action):
        return s + "/" + str(action)


def best_score(root):
    from .alphabeta import alphabeta
    problem = TreeProblem(root)
    return alphabeta(problem, problem.initial, max(x.count("/") for x in problem.nodes)).value
