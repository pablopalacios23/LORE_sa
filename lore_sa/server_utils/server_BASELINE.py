# ============================
# 🚀 SERVIDOR FLOWER — con SuperTree
# ============================
# Uso desde notebook:
#
#   from lore_sa.server_utils.server import make_server_app
#
#   server_app = make_server_app(
#       dataset_name=DATASET_NAME,
#       class_column=CLASS_COLUMN,
#       num_clients=NUM_CLIENTS,
#       num_server_rounds=NUM_SERVER_ROUNDS,
#       num_train_rounds=NUM_TRAIN_ROUNDS,
#       min_available_clients=MIN_AVAILABLE_CLIENTS,
#       unique_labels=UNIQUE_LABELS,
#       features=FEATURES,
#       load_data_fn=load_data_general,
#   )
# ============================

import os
import json
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from graphviz import Digraph
from lore_sa.surrogate.decision_tree import SuperTree
from lore_sa.client_utils.client_BASELINE import Net


# ====================================================================
# Utilidades de modelo (no dependen de globals)
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
# Funciones de visualización SuperTree (no dependen de globals)
# ====================================================================

def print_supertree_legible_fusionado(
    node,
    feature_names,
    class_names,
    numeric_features,
    scaler,
    global_mapping,
    depth=0
):
    indent = "|   " * depth
    if node is None:
        print(f"{indent}[Nodo None]")
        return

    if getattr(node, "is_leaf", False):
        class_idx = int(np.argmax(node.labels))
        print(f"{indent}class: {class_names[class_idx]} (pred: {node.labels})")
        return

    feat_idx = node.feat
    feat_name = feature_names[feat_idx]
    intervals = node.intervals
    children = node.children

    if feat_name in numeric_features:
        bounds = [-np.inf] + list(intervals)
        while len(bounds) < len(children) + 1:
            bounds.append(np.inf)

        for i, child in enumerate(children):
            left = bounds[i]
            right = bounds[i + 1]

            if i == 0:
                cond = f"{feat_name} ≤ {right:.2f}"
            elif i == len(children) - 1:
                cond = f"{feat_name} > {left:.2f}"
            else:
                cond = f"{feat_name} ∈ ({left:.2f}, {right:.2f}]"
            print(f"{indent}{cond}")
            print_supertree_legible_fusionado(
                child, feature_names, class_names, numeric_features,
                scaler=None, global_mapping=global_mapping, depth=depth + 1
            )

    elif "=" in feat_name or "_" in feat_name:
        if "=" in feat_name:
            var, val = feat_name.split("=", 1)
        else:
            var, val = feat_name.split("_", 1)
        var = var.strip()
        val = val.strip()

        if len(children) != 2:
            print(f"[ERROR] Nodo OneHot {feat_name} tiene {len(children)} hijos, esperado 2.")

        conds = [f'{var} != "{val}"', f'{var} == "{val}"']
        for i, child in enumerate(children):
            print(f"{indent}{conds[i]}")
            print_supertree_legible_fusionado(
                child, feature_names, class_names, numeric_features, scaler, global_mapping, depth + 1
            )

    elif global_mapping and feat_name in global_mapping:
        vals_cat = global_mapping[feat_name]
        for i, child in enumerate(children):
            try:
                val_idx = node.intervals[i] if hasattr(node, "intervals") and i < len(node.intervals) else int(getattr(node, "thresh", 0))
                val = vals_cat[val_idx] if val_idx < len(vals_cat) else f"desconocido({val_idx})"
            except Exception as e:
                print(f"[DEPURACIÓN] Error interpretando categórica: {e}")
                val = "?"
            cond = f'{feat_name} != "{val}"' if i == 0 else f'{feat_name} == "{val}"'
            print(f"{indent}{cond}")
            print_supertree_legible_fusionado(
                child, feature_names, class_names, numeric_features, scaler, global_mapping, depth + 1
            )

    else:
        print(f"{indent}{feat_name} [tipo desconocido]")
        print(f"    [DEPURACIÓN] Nombres de features: {feature_names}")
        print(f"    [DEPURACIÓN] Nombres numéricas: {numeric_features}")
        print(f"    [DEPURACIÓN] global_mapping: {list(global_mapping.keys()) if global_mapping else None}")
        print(f"    [DEPURACIÓN] children: {len(children)}")
        for child in children:
            print_supertree_legible_fusionado(
                child, feature_names, class_names, numeric_features, scaler, global_mapping, depth + 1
            )


def save_supertree_plot(
    root_node,
    round_number,
    feature_names,
    class_names,
    numeric_features,
    global_mapping,
    folder="Supertree",
):
    dot = Digraph()
    node_id = [0]

    def add_node(node, parent=None, edge_label=""):
        curr = str(node_id[0]); node_id[0] += 1

        if node.is_leaf:
            class_index = int(np.argmax(node.labels))
            label = f"class: {class_names[class_index]}\n{node.labels}"
        else:
            fname = feature_names[node.feat]
            label = fname.split("_", 1)[0] if "_" in fname else fname

        dot.node(curr, label)
        if parent: dot.edge(parent, curr, label=edge_label)

        if not node.is_leaf:
            fname = feature_names[node.feat]
            if "_" in fname:
                _, val = fname.split("_", 1)
                add_node(node.children[0], curr, f'≠ "{val.strip()}"')
                add_node(node.children[1], curr, f'= "{val.strip()}"')
            elif fname in numeric_features:
                thr = node.intervals[0] if node.intervals else node.thresh
                add_node(node.children[0], curr, f"≤ {thr:.2f}")
                add_node(node.children[1], curr, f"> {thr:.2f}")
            elif fname in global_mapping:
                vals = global_mapping[fname]
                val = vals[node.intervals[0]] if node.intervals else "?"
                add_node(node.children[0], curr, f'= "{val}"')
                add_node(node.children[1], curr, f'≠ "{val}"')
            else:
                for ch in node.children:
                    add_node(ch, curr, "?")

    folder_path = f"Ronda_{round_number}/{folder}"
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{folder_path}/supertree_ronda_{round_number}"
    add_node(root_node)
    dot.render(filename, format="pdf", cleanup=True)
    return f"{filename}.pdf"


# ====================================================================
# FACTORY — punto de entrada público
# ====================================================================

def make_server_app(
    dataset_name,
    class_column,
    num_clients,
    num_server_rounds,
    num_train_rounds,
    min_available_clients,
    unique_labels,
    features,
    load_data_fn,
):
    """
    Crea un ServerApp de Flower parametrizado.

    Parámetros
    ----------
    dataset_name         : nombre del dataset en HuggingFace
    class_column         : columna objetivo
    num_clients          : número de clientes federados
    num_server_rounds    : rondas totales (últimas para explicación)
    num_train_rounds     : rondas de entrenamiento (donde se fusionan árboles)
    min_available_clients: mínimo de clientes disponibles por ronda
    unique_labels        : la MISMA lista [] del notebook
    features             : la MISMA lista [] del notebook
    load_data_fn         : referencia a load_data_general del notebook
    """

    # Estado interno del servidor (mutable para que las closures puedan escribir)
    state = {
        "latest_supertree_json": None,
        "global_mapping_json": None,
        "feature_names_json": None,
    }

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

        model = Net(feat_count, label_count)
        initial_params = ndarrays_to_parameters(_get_nn_parameters(model))

        strategy = FedAvg(
            min_available_clients=min_available_clients,
            fit_metrics_aggregation_fn=_weighted_average,
            evaluate_metrics_aggregation_fn=_weighted_average,
            initial_parameters=initial_params,
        )

        strategy.configure_fit = _make_inject_round(state, num_server_rounds)( strategy.configure_fit)
        strategy.configure_evaluate = _make_inject_round(state, num_server_rounds)(strategy.configure_evaluate)

        original_aggregate = strategy.aggregate_evaluate

        def custom_aggregate_evaluate(server_round, results, failures):
            aggregated_metrics = original_aggregate(server_round, results, failures)

            # Ronda final: NO fusionar nada
            if server_round > num_train_rounds:
                return aggregated_metrics

            try:
                print(f"\n[SERVIDOR] 🌲 Generando SuperTree - Ronda {server_round}")

                tree_nodes = []
                all_distincts = defaultdict(set)
                client_encoders = {}

                feature_names = None
                numeric_features = None
                class_names = None

                # 1) recolectar mapeos categóricos y metadatos
                for (_, evaluate_res) in results:
                    metrics = evaluate_res.metrics
                    for k, v in metrics.items():
                        if k.startswith("distinct_values_"):
                            cid = k.split("_")[-1]
                            enc = json.loads(v)
                            client_encoders[cid] = enc
                            for feat, d in enc.items():
                                all_distincts[feat].update(d["distinct_values"])

                global_mapping = {feat: sorted(list(vals)) for feat, vals in all_distincts.items()}

                # 2) recolectar árboles y demás metadatos por cliente
                for (_, evaluate_res) in results:
                    metrics = evaluate_res.metrics
                    for k, v in metrics.items():
                        if k.startswith("tree_ensemble_"):
                            cid = k.split("_")[-1]
                            trees_list = json.loads(v)

                            if feature_names is None and f"encoded_feature_names_{cid}" in metrics:
                                feature_names = json.loads(metrics[f"encoded_feature_names_{cid}"])
                            if numeric_features is None and f"numeric_features_{cid}" in metrics:
                                numeric_features = json.loads(metrics[f"numeric_features_{cid}"])
                            if class_names is None and f"unique_labels_{cid}" in metrics:
                                class_names = json.loads(metrics[f"unique_labels_{cid}"])

                            for tdict in trees_list:
                                root = SuperTree.Node.from_dict(tdict)
                                tree_nodes.append(root)

                if not tree_nodes:
                    return aggregated_metrics

                # 3) fusionar
                st = SuperTree()
                st.mergeDecisionTrees(
                    roots=tree_nodes,
                    num_classes=len(class_names),
                    feature_names=feature_names,
                    categorical_features=list(global_mapping.keys()),
                    global_mapping=global_mapping,
                )

                st.prune_redundant_leaves_full()

                # 4) guardar/emitir
                save_supertree_plot(
                    root_node=st.root,
                    round_number=server_round,
                    feature_names=feature_names,
                    class_names=class_names,
                    numeric_features=numeric_features,
                    global_mapping=global_mapping,
                )

                state["latest_supertree_json"] = json.dumps(st.root.to_dict())
                state["global_mapping_json"] = json.dumps(global_mapping)
                state["feature_names_json"] = json.dumps(feature_names)

            except Exception as e:
                print(f"[SERVIDOR] ❌ Error en SuperTree: {e}")

            return aggregated_metrics

        strategy.aggregate_evaluate = custom_aggregate_evaluate
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=num_server_rounds),
        )

    return ServerApp(server_fn=server_fn)


# ====================================================================
# Helper: inyectar config en cada ronda
# ====================================================================

def _make_inject_round(state, num_server_rounds):
    """
    Devuelve un decorador que inyecta server_round, supertree y explain_only
    en la config de cada cliente.
    """
    def decorator(original_fn):
        def wrapper(server_round, parameters, client_manager):
            instructions = original_fn(server_round, parameters, client_manager)
            for _, ins in instructions:
                ins.config["server_round"] = server_round

                # Siempre mandamos el último SuperTree disponible
                if state["latest_supertree_json"]:
                    ins.config["supertree"] = state["latest_supertree_json"]
                    ins.config["global_mapping"] = state["global_mapping_json"]
                    ins.config["feature_names"] = state["feature_names_json"]

                # Ronda final: modo solo explicación
                if server_round == num_server_rounds:
                    ins.config["explain_only"] = True

            return instructions
        return wrapper
    return decorator