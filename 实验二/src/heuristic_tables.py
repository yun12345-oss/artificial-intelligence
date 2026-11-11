"""方法专属预处理。地标严格使用有向正反距离，不与曼哈顿组合。"""
from collections import deque
from dataclasses import dataclass
import heapq
import math


def all_distances(adjacency,source):
    distances={source:0}
    queue=[(0,source)]
    while queue:
        distance,p=heapq.heappop(queue)
        if distance != distances[p]:
            continue
        for q,cost in adjacency[p]:
            value=distance+cost
            if value < distances.get(q,math.inf):
                distances[q]=value
                heapq.heappush(queue,(value,q))
    return distances


def choose_landmarks(graph):
    n,m=len(graph.grid),len(graph.grid[0])
    chosen=[]
    for corner in ((0,0),(0,m-1),(n-1,0),(n-1,m-1)):
        chosen.append(min((p for p in graph.weights if p not in chosen),
                          key=lambda p:(abs(p[0]-corner[0])+abs(p[1]-corner[1]),p)))
    return tuple(chosen)


@dataclass
class PreparedHeuristic:
    method: str
    goal: tuple
    minimum: int
    steps: dict | None = None
    landmarks: tuple = ()
    from_landmark: tuple = ()
    to_landmark: tuple = ()
    dijkstra_calls: int = 0
    bfs_calls: int = 0

    @property
    def entries(self):
        return len(self.steps) if self.steps is not None else sum(map(len,self.from_landmark+self.to_landmark))

    def __call__(self,n):
        if self.method == "manhattan":
            return self.minimum*(abs(n[0]-self.goal[0])+abs(n[1]-self.goal[1]))
        if self.method == "bfs":
            return self.minimum*self.steps.get(n,0)
        best=0
        for forward,backward in zip(self.from_landmark,self.to_landmark):
            if self.goal in forward and n in forward:
                best=max(best,forward[self.goal]-forward[n])
            if self.goal in backward and n in backward:
                best=max(best,backward[n]-backward[self.goal])
        return best


def prepare(graph,goal,method,landmarks):
    if goal not in graph.weights:
        raise ValueError("目标不可通行")
    if method == "manhattan":
        return PreparedHeuristic(method,goal,graph.minimum)
    if method == "bfs":
        steps={goal:0}
        queue=deque([goal])
        while queue:
            p=queue.popleft()
            for q,_ in graph.reverse[p]:
                if q not in steps:
                    steps[q]=steps[p]+1
                    queue.append(q)
        return PreparedHeuristic(method,goal,graph.minimum,steps=steps,bfs_calls=1)
    if method != "landmark":
        raise ValueError("未知启发方法")
    forwards=tuple(all_distances(graph.forward,p) for p in landmarks)
    backwards=tuple(all_distances(graph.reverse,p) for p in landmarks)
    return PreparedHeuristic(method,goal,graph.minimum,landmarks=tuple(landmarks),
                             from_landmark=forwards,to_landmark=backwards,dijkstra_calls=8)
