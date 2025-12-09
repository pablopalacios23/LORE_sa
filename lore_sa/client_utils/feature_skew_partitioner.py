# feature_skew_partitioner.py

from __future__ import annotations
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
import datasets

from flwr_datasets.partitioner import Partitioner


class FeatureSkewPartitioner(Partitioner):
    """
    Non-IID por feature (feature skew):

    - Escoge una feature numérica (skew_by).
    - Ordena el dataset por esa feature.
    - Divide en bloques contiguos (rangos) y asigna cada bloque a un cliente.
      → Cliente 0: valores bajos
      → Cliente N: valores altos

    Opciones:
      - bins = "equal"     → cortes por tamaño (mismas muestras por cliente)
      - bins = "quantile"  → similar, pero respetando la distribución (por defecto)
    """

    def __init__(
        self,
        num_partitions: int,
        skew_by: str,
        bins: str = "quantile",
        shuffle: bool = False,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._skew_by = skew_by
        self._bins = bins
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._partition_id_to_indices: Dict[int, List[int]] = {}
        self._partition_id_to_indices_determined: bool = False

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    # ---------------- Lógica principal ----------------
    def _determine_partition_id_to_indices_if_needed(self) -> None:
        if self._partition_id_to_indices_determined:
            return

        col = self._skew_by
        if col not in self.dataset.column_names:
            raise ValueError(f"Feature '{col}' no existe en el dataset HF.")

        vals = np.array(self.dataset[col], dtype=float)
        idx = np.arange(len(vals))

        # Ordenar por la feature (feature skew)
        order = np.argsort(vals)
        vals_sorted = vals[order]
        idx_sorted = idx[order]

        # Opcional: pequeño shuffle dentro de bloques luego, pero no rompemos el rango
        if self._bins not in ("equal", "quantile"):
            raise ValueError("bins debe ser 'equal' o 'quantile'.")

        # Cortes
        if self._bins == "equal":
            # trozos iguales en nº de muestras
            splits = np.array_split(idx_sorted, self._num_partitions)
        else:  # "quantile"
            # mismos tamaños también, pero dejamos nombre por claridad
            splits = np.array_split(idx_sorted, self._num_partitions)

        partition_id_to_indices: Dict[int, List[int]] = {pid: [] for pid in range(self._num_partitions)}
        for pid, part_idx in enumerate(splits):
            part_idx = part_idx.tolist()
            if self._shuffle:
                self._rng.shuffle(part_idx)
            partition_id_to_indices[pid].extend(part_idx)

        self._partition_id_to_indices = partition_id_to_indices
        self._partition_id_to_indices_determined = True
