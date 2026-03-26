# ============================
# 🚀 SERVIDOR FLOWER — FedAvg puro
# ============================
# Solo agrega pesos de la NN. Sin SuperTree ni fusión de árboles.
# La última ronda marca explain_only=True para que los clientes expliquen.
#
# Uso desde notebook:
#
#   from lore_sa.server_utils.server import make_server_app
#
#   server_app = make_server_app(
#       dataset_name=DATASET_NAME,
#       class_column=CLASS_COLUMN,
#       num_clients=NUM_CLIENTS,
#       num_server_rounds=NUM_SERVER_ROUNDS,
#       min_available_clients=MIN_AVAILABLE_CLIENTS,
#       unique_labels=UNIQUE_LABELS,
#       features=FEATURES,
#       load_data_fn=load_data_general,
#   )
# ============================

import numpy as np
from typing import List, Tuple, Dict

import torch

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from lore_sa.client_utils.client_BASELINE import Net


# ====================================================================
# Utilidades
# ====================================================================

def _get_nn_parameters(nn_model):
    """Extrae pesos de la NN como lista de ndarrays."""
    return [v.cpu().detach().numpy() for v in nn_model.state_dict().values()]


def _weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Promedio ponderado de métricas de los clientes."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for n, met in metrics:
        for k, v in met.items():
            if isinstance(v, (float, int)):
                sums[k] = sums.get(k, 0.0) + n * float(v)
                counts[k] = counts.get(k, 0) + n
    return {k: sums[k] / counts[k] for k in sums}


# ====================================================================
# FACTORY — punto de entrada público
# ====================================================================

def make_server_app(
    dataset_name,
    class_column,
    num_clients,
    num_server_rounds,
    min_available_clients,
    unique_labels,
    features,
    load_data_fn,
):
    """
    Crea un ServerApp de Flower parametrizado (FedAvg puro).

    Parámetros
    ----------
    dataset_name         : nombre del dataset en HuggingFace
    class_column         : columna objetivo
    num_clients          : número de clientes federados
    num_server_rounds    : rondas totales (la última solo para explicaciones)
    min_available_clients: mínimo de clientes disponibles por ronda
    unique_labels        : la MISMA lista [] del notebook
    features             : la MISMA lista [] del notebook
    load_data_fn         : referencia a load_data_general del notebook
    """

    def server_fn(context: Context) -> ServerAppComponents:

        # Asegurar que features y unique_labels están rellenos
        if not features or not unique_labels:
            load_data_fn(
                flower_dataset_name=dataset_name,
                class_col=class_column,
                partition_id=0,
                num_partitions=num_clients,
            )

        feat_count = len(features) if features else 2
        label_count = len(unique_labels) if unique_labels else 2

        # Modelo inicial (pesos aleatorios)
        model = Net(feat_count, label_count)
        initial_params = ndarrays_to_parameters(_get_nn_parameters(model))

        # Estrategia FedAvg
        strategy = FedAvg(
            min_available_clients=min_available_clients,
            fit_metrics_aggregation_fn=_weighted_average,
            evaluate_metrics_aggregation_fn=_weighted_average,
            initial_parameters=initial_params,
        )

        # Inyectar server_round y explain_only en cada ronda
        strategy.configure_fit = _inject_round(
            strategy.configure_fit, num_server_rounds
        )
        strategy.configure_evaluate = _inject_round(
            strategy.configure_evaluate, num_server_rounds
        )

        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=num_server_rounds),
        )

    return ServerApp(server_fn=server_fn)


# ====================================================================
# Helper: inyectar config en cada ronda
# ====================================================================

def _inject_round(original_fn, num_server_rounds):
    """
    Wrapper que añade server_round a la config de cada cliente.
    En la última ronda, marca explain_only=True.
    """
    def wrapper(server_round, parameters, client_manager):
        instructions = original_fn(server_round, parameters, client_manager)
        for _, ins in instructions:
            ins.config["server_round"] = server_round

            if server_round == num_server_rounds:
                ins.config["explain_only"] = True

        return instructions
    return wrapper