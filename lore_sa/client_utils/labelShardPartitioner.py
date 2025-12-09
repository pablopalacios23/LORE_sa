# label_shard_partitioner.py

from __future__ import annotations

from typing import Optional, Dict, List, Any

import numpy as np
import datasets  # Hugging Face Dataset
import pandas as pd

from flwr_datasets.partitioner import Partitioner


class LabelShardPartitioner(Partitioner):
    """
    Particionador Non-IID tipo "label sharding":

    - Cada clase se puede dividir en varios "shards" (trozos de índices).
    - Cada shard de una clase se asigna a UN cliente.
    - Cada cliente ve como máximo `n_classes_per_client` clases distintas.
    - Una misma clase puede aparecer en varios clientes (no se fuerza 1 clase -> 1 cliente).
    - Controlas cuántos shards por clase con `shards_per_class`.

    Ejemplos de uso:

        from flwr_datasets import FederatedDataset
        from label_shard_partitioner import LabelShardPartitioner

        partitioner = LabelShardPartitioner(
            num_partitions=NUM_CLIENTS,
            partition_by=CLASS_COLUMN,
            n_classes_per_client=1,     # máx clases distintas por cliente
            shards_per_class=4,         # cuántos trozos crear por clase
            shuffle=True,
            seed=42,
        )

        fds = FederatedDataset(
            dataset=DATASET_NAME,
            partitioners={"train": partitioner},
        )

    Idea:
    - Si tienes pocas clases pero muchos clientes, puedes aumentar `shards_per_class`
      para que cada clase se reparta entre varios clientes sin que ningún cliente tenga todas
      las instancias de esa clase y manteniendo el límite de clases por cliente.
    """

    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        n_classes_per_client: int = 1,
        shards_per_class: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()

        self._partition_by = partition_by
        self._n_classes_per_client = n_classes_per_client
        self._shards_per_class = shards_per_class
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=seed)

        # Se rellenan en la primera llamada a load_partition
        self._partition_id_to_indices: Dict[int, List[int]] = {}
        self._partition_id_to_indices_determined: bool = False

    # ------------------------------------------------------------------
    # API que usa FederatedDataset
    # ------------------------------------------------------------------
    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Devuelve la partición `partition_id` como HuggingFace Dataset."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Número total de particiones."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    # ------------------------------------------------------------------
    # Lógica de particionado
    # ------------------------------------------------------------------
    def _determine_partition_id_to_indices_if_needed(self) -> None:
        """Crea el mapeo partition_id -> lista de índices (solo una vez)."""
        if self._partition_id_to_indices_determined:
            return

        # Columna de clase completa sobre todo el dataset HF
        targets = np.array(self.dataset[self._partition_by])
        # preserva orden de aparición de las clases
        unique_classes = list(dict.fromkeys(targets))

        # Índices por clase
        idx_by_class: Dict[Any, List[int]] = {
            c: np.where(targets == c)[0].tolist() for c in unique_classes
        }

        # Inicializar estructura de particiones
        partition_id_to_indices: Dict[int, List[int]] = {
            pid: [] for pid in range(self._num_partitions)
        }

        # Cada cliente tendrá como máximo n_classes_per_client clases distintas
        labels_per_client: List[set] = [set() for _ in range(self._num_partitions)]
        capacity_left: List[int] = [self._n_classes_per_client] * self._num_partitions

        # Orden aleatorio de clases para no sesgar
        classes_order = list(unique_classes)
        if self._shuffle:
            self._rng.shuffle(classes_order)

        # Asignamos CADA CLASE en VARIOS SHARDS a distintos clientes
        for c in classes_order:
            class_indices = idx_by_class[c]
            if self._shuffle:
                self._rng.shuffle(class_indices)

            # Número de shards para esta clase (como mucho shards_per_class,
            # pero no más que el nº de muestras de la clase)
            num_shards = min(self._shards_per_class, len(class_indices))
            # np.array_split reparte lo más uniformemente posible
            shards = np.array_split(np.array(class_indices), num_shards)

            for shard in shards:
                if shard.size == 0:
                    continue

                # Clientes candidatos que aún pueden ver esta clase
                candidates = [
                    i
                    for i in range(self._num_partitions)
                    if (capacity_left[i] > 0) and (c not in labels_per_client[i])
                ]
                if not candidates:
                    # Si todos están "llenos" de clases, relajamos la restricción
                    candidates = list(range(self._num_partitions))

                chosen_pid = int(self._rng.choice(candidates))
                partition_id_to_indices[chosen_pid].extend(shard.tolist())

                # Actualizar clases que ve ese cliente
                if c not in labels_per_client[chosen_pid]:
                    labels_per_client[chosen_pid].add(c)
                    if len(labels_per_client[chosen_pid]) >= self._n_classes_per_client:
                        capacity_left[chosen_pid] = 0

        # Barajar dentro de cada partición (opcional, por si se quiere evitar bloques de clases)
        if self._shuffle:
            for pid in partition_id_to_indices:
                self._rng.shuffle(partition_id_to_indices[pid])

        self._partition_id_to_indices = partition_id_to_indices
        self._partition_id_to_indices_determined = True

    # ------------------------------------------------------------------
    # Comprobaciones auxiliares
    # ------------------------------------------------------------------
    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Comprueba que num_partitions <= num_rows cuando ya hay dataset."""
        if not self._partition_id_to_indices_determined and hasattr(self, "dataset"):
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions must be <= number of samples in dataset."
                )

    def _check_num_partitions_greater_than_zero(self) -> None:
        """num_partitions > 0."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")

    # ------------------------------------------------------------------
    # Versión para pandas, para debug fuera de HF
    # ------------------------------------------------------------------
    def split_pandas(self, df: pd.DataFrame) -> Dict[int, List[int]]:
        """
        Igual que la lógica interna, pero trabajando sobre un DataFrame de pandas.

        Devuelve: dict[partition_id] -> lista de índices (posiciones) de df.
        """
        targets = df[self._partition_by].to_numpy()
        unique_classes = list(dict.fromkeys(targets))

        idx_by_class: Dict[Any, List[int]] = {
            c: df.index[df[self._partition_by] == c].tolist() for c in unique_classes
        }

        partition_id_to_indices: Dict[int, List[int]] = {
            pid: [] for pid in range(self._num_partitions)
        }

        labels_per_client: List[set] = [set() for _ in range(self._num_partitions)]
        capacity_left: List[int] = [self._n_classes_per_client] * self._num_partitions

        classes_order = list(unique_classes)
        if self._shuffle:
            self._rng.shuffle(classes_order)

        for c in classes_order:
            class_indices = idx_by_class[c]
            if self._shuffle:
                self._rng.shuffle(class_indices)

            num_shards = min(self._shards_per_class, len(class_indices))
            shards = np.array_split(np.array(class_indices), num_shards)

            for shard in shards:
                if shard.size == 0:
                    continue

                candidates = [
                    i
                    for i in range(self._num_partitions)
                    if (capacity_left[i] > 0) and (c not in labels_per_client[i])
                ]
                if not candidates:
                    candidates = list(range(self._num_partitions))

                chosen_pid = int(self._rng.choice(candidates))
                partition_id_to_indices[chosen_pid].extend(shard.tolist())

                if c not in labels_per_client[chosen_pid]:
                    labels_per_client[chosen_pid].add(c)
                    if len(labels_per_client[chosen_pid]) >= self._n_classes_per_client:
                        capacity_left[chosen_pid] = 0

        if self._shuffle:
            for pid in partition_id_to_indices:
                self._rng.shuffle(partition_id_to_indices[pid])

        return partition_id_to_indices