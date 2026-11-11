from dataclasses import dataclass
from types import MappingProxyType

DIRECTIONS = ((-1,0),(0,1),(1,0),(0,-1))


@dataclass(frozen=True)
class WeightedGraph:
    grid: tuple
    weights: object
    forward: object
    reverse: object
    minimum: int

    @classmethod
    def build(cls, grid):
        if not grid or not grid[0] or any(len(r) != len(grid[0]) for r in grid):
            raise ValueError("地图必须为非空矩形")
        weights={}
        for r,row in enumerate(grid):
            for c,value in enumerate(row):
                if value == "X":
                    continue
                if type(value) is not int or value < 1:
                    raise ValueError("主实验权值必须为正整数")
                weights[(r,c)]=value
        if not weights:
            raise ValueError("没有可通行节点")
        forward,reverse={}, {p:[] for p in weights}
        for p in weights:
            edges=[]
            for dr,dc in DIRECTIONS:
                q=(p[0]+dr,p[1]+dc)
                if q in weights:
                    edges.append((q,weights[q]))
                    reverse[q].append((p,weights[q]))
            forward[p]=tuple(edges)
        return cls(tuple(tuple(r) for r in grid),MappingProxyType(weights),
                   MappingProxyType(forward),MappingProxyType({p:tuple(e) for p,e in reverse.items()}),min(weights.values()))

    def search(self,start,goal,h,trace=None):
        from .core.astar import astar
        if start not in self.weights or goal not in self.weights:
            raise ValueError("起终点必须可通行")
        return astar(start,goal,self.forward.__getitem__,lambda n,g:h(n),trace=trace)
