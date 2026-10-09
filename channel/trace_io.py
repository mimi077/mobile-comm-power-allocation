"""轨迹数据 npz 读写 —— env / 基线 / 评估唯一的数据入口

数据契约：
    g   : [N, T] float32, 每时隙信道功率增益（线性）, 已含路径损耗+阴影+小尺度
    d   : [T]    float32, 各时隙车-节点距离 [m]（同速度所有轨迹共享）
    meta: JSON 字符串, 全部场景元数据
"""

import json

import numpy as np


def save_traces(path: str, g: np.ndarray, d: np.ndarray, meta: dict) -> None:
    np.savez_compressed(
        path,
        g=g.astype(np.float32),
        d=d.astype(np.float32),
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )


def load_traces(path: str):
    z = np.load(path, allow_pickle=False)
    return z["g"], z["d"], json.loads(str(z["meta"]))
