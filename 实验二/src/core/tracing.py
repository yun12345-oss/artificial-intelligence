"""仅在非计时运行中收集事件；无穷界转换为JSON兼容字符串。"""
import math


def event(trace, kind, **fields):
    if trace is not None:
        trace.append({"event": kind, **{k: (str(v) if isinstance(v, float)
                       and not math.isfinite(v) else v) for k, v in fields.items()}})
