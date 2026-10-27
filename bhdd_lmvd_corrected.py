"""Local BHDD LMVD variant: fixes only sequence-first positional encoding.
Original model/BHDD/model.py remains untouched. Not author-confirmed code.
"""
from pathlib import Path
import importlib.util
import torch

_source = Path(__file__).resolve().parent / 'model' / 'BHDD' / 'model.py'
_spec = importlib.util.spec_from_file_location('_bhdd_lmvd_original', _source)
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)


class SequenceFirstPositionalEncoding(_original.PositionalEncoding):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('Expected sequence-first input (T, batch, features)')
        if x.size(0) > self.pe.size(1):
            raise ValueError('Sequence exceeds positional encoding capacity')
        return x + self.pe[:, :x.size(0)].transpose(0, 1).to(device=x.device, dtype=x.dtype)


class Net(_original.Net):
    def __init__(self, adj_matrix: torch.Tensor, K: int = 186) -> None:
        super().__init__(adj_matrix=adj_matrix, K=K)
        self.transformer.pos_encoder = SequenceFirstPositionalEncoding(512)
