"""The published ONNX as a batched policy."""

from __future__ import annotations

import hashlib

import numpy as np
import onnx
import onnxruntime as ort
import torch


def sha256(path) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class OnnxPolicy:
    """The file's batch dimension made dynamic in memory; the file itself is untouched."""

    def __init__(self, path):
        m = onnx.load(str(path))
        for v in (*m.graph.input, *m.graph.output):
            v.type.tensor_type.shape.dim[0].dim_param = 'N'
        self.s = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
        self.inp, self.out = self.s.get_inputs()[0].name, self.s.get_outputs()[0].name
        self.obs_len = self.s.get_inputs()[0].shape[1]
        self.action_len = self.s.get_outputs()[0].shape[1]

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        a = self.s.run([self.out], {self.inp: obs.detach().cpu().numpy().astype(np.float32)})[0]
        return torch.from_numpy(a).to(obs.device)
