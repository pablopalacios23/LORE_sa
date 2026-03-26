# ==========================
# 🌼 CLIENTE FLOWER — Baseline (sin ensemble)
# ==========================
# Fichero único para experimentos sin RF/GB (árbol surrogado entrena sobre y_train).
#
# Uso desde notebook:
#
#   from lore_sa.client_utils.client_BASELINE import make_client_app_baseline
#
#   client_app = make_client_app_baseline(
#       dataset_name=DATASET_NAME,
#       class_column=CLASS_COLUMN,
#       num_clients=NUM_CLIENTS,
#       num_train_rounds=NUM_TRAIN_ROUNDS,
#       unique_labels=UNIQUE_LABELS,
#       features=FEATURES,
#       load_data_fn=load_data_general,
#       set_params_fn=set_model_params,
#       get_params_fn=get_model_parameters,
#   )
# ==========================

import copy
import operator
import os
import json
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score,
)
from sklearn.exceptions import NotFittedError

import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from lore_sa.dataset import TabularDataset
from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.lore import TabularGeneticGeneratorLore
from lore_sa.rule import Expression, Rule
from lore_sa.surrogate.decision_tree import EnsembleDecisionTreeSurrogate, SuperTree
from lore_sa.encoder_decoder import ColumnTransformerEnc

from sklearn.metrics import pairwise_distances

import shap
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular
from tqdm import tqdm

from lore_sa.client_utils import ClientUtilsMixin
from lore_sa.client_utils.cumple_regla import ReglaEvaluator
from lore_sa.client_utils.explanation_metrics import Explainer_metrics


# ====================================================================
# Wrappers / Modelos
# ====================================================================

class TorchNNWrapper:
    def __init__(self, model, num_idx, mean, scale):
        self.model = model
        self.model.eval()
        self.num_idx = np.asarray(num_idx, dtype=int)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.scale_safe = np.where(self.scale == 0, 1.0, self.scale)

    def _scale_internally(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xs = X.copy()
        if Xs.ndim == 1:
            Xs = Xs[None, :]
        Xs[:, self.num_idx] = (Xs[:, self.num_idx] - self.mean) / self.scale_safe
        return Xs

    def predict(self, X):
        Xs = self._scale_internally(X)
        with torch.no_grad():
            X_tensor = torch.tensor(Xs, dtype=torch.float32)
            logits = self.model(X_tensor)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        Xs = self._scale_internally(X)
        with torch.no_grad():
            X_tensor = torch.tensor(Xs, dtype=torch.float32)
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()


class Net(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Net, self).__init__()
        h1 = max(128, input_dim * 2)
        h2 = max(64, h1 // 2)
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# ====================================================================
# FlowerClient — Baseline (sin ensemble)
# ====================================================================

class FlowerClient(NumPyClient, ClientUtilsMixin):
    def __init__(self, tree_model, nn_model, X_train, y_train, X_test, y_test,
                 X_test_global, y_test_global, scaler_nn_mean, scaler_nn_scale,
                 num_idx, dataset, client_id, feature_names, label_encoder,
                 scaler, numeric_features, encoder, preprocessor,
                 # ── Inyectados desde notebook ──
                 _UNIQUE_LABELS, _FEATURES, _NUM_TRAIN_ROUNDS, _DATASET_NAME,
                 _set_model_params, _get_model_parameters):
        self.tree_model = tree_model
        self.nn_model = nn_model
        self.nn_model_local = copy.deepcopy(nn_model)
        self.nn_model_global = nn_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_global = X_test_global
        self.y_test_global = y_test_global
        self.scaler_nn_mean = np.asarray(scaler_nn_mean, dtype=np.float32)
        self.scaler_nn_scale = np.where(np.asarray(scaler_nn_scale, np.float32) == 0, 1.0,
                                         np.asarray(scaler_nn_scale, np.float32))
        self.num_idx = np.asarray(num_idx, dtype=int)
        self.dataset = dataset
        self.client_id = client_id
        os.makedirs("results", exist_ok=True)
        self.local_ckpt = f"results/bb_local_client_{self.client_id}.pth"
        self.local_trained = False
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.numeric_features = numeric_features
        self.encoder = encoder
        self.unique_labels = label_encoder.classes_.tolist()
        self.y_train_nn = y_train.astype(np.int64)
        self.y_test_nn = y_test.astype(np.int64)
        self.received_supertree = None
        self.preprocessor = preprocessor

        # ── Referencias inyectadas desde notebook ──
        self._UNIQUE_LABELS = _UNIQUE_LABELS
        self._FEATURES = _FEATURES
        self._NUM_TRAIN_ROUNDS = _NUM_TRAIN_ROUNDS
        self._DATASET_NAME = _DATASET_NAME
        self._set_model_params = _set_model_params
        self._get_model_parameters = _get_model_parameters

    def _train_nn(self, model, epochs=10, lr=1e-3):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        X = np.asarray(self.X_train, dtype=np.float32).copy()
        scale_safe = np.where(self.scaler_nn_scale == 0, 1.0, self.scaler_nn_scale)
        X[:, self.num_idx] = (X[:, self.num_idx] - self.scaler_nn_mean) / scale_safe

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train_nn, dtype=torch.long)

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = loss_fn(logits, y_tensor)
            loss.backward()
            optimizer.step()

    def fit(self, parameters, config):
        self._set_model_params(
            self.tree_model,
            self.nn_model_global,
            {"tree": [
                self.tree_model.get_params()["max_depth"],
                self.tree_model.get_params()["min_samples_split"],
                self.tree_model.get_params()["min_samples_leaf"],
            ], "nn": parameters}
        )

        round_number = int(config.get("server_round", 1))

        if not self.local_trained:
            if os.path.exists(self.local_ckpt):
                state = torch.load(self.local_ckpt, map_location="cpu")
                self.nn_model_local.load_state_dict(state)
                self.nn_model_local.eval()
                bb_local_tmp = TorchNNWrapper(self.nn_model_local, self.num_idx,
                                              self.scaler_nn_mean, self.scaler_nn_scale)
                with torch.no_grad():
                    acc_train_load = accuracy_score(self.y_train_nn, bb_local_tmp.predict(self.X_train))
                self.local_trained = True
            else:
                self.nn_model_local = copy.deepcopy(self.nn_model_global)
                self._train_nn(self.nn_model_local, epochs=80, lr=1e-3)
                self.nn_model_local.eval()
                bb_local_tmp = TorchNNWrapper(self.nn_model_local, self.num_idx,
                                              self.scaler_nn_mean, self.scaler_nn_scale)
                with torch.no_grad():
                    acc_train_now = accuracy_score(self.y_train_nn, bb_local_tmp.predict(self.X_train))
                torch.save(self.nn_model_local.state_dict(), self.local_ckpt)
                self.local_trained = True
                print(f"[CLIENTE {self.client_id}] ✅ LOCAL baseline entrenado y guardado")

        self._train_nn(self.nn_model_global, epochs=10, lr=1e-3)

        if round_number <= self._NUM_TRAIN_ROUNDS:
            self.tree_model.fit(self.X_train, self.y_train)

        nn_weights = self._get_model_parameters(self.tree_model, self.nn_model_global)["nn"]
        return nn_weights, len(self.X_train), {}

    def evaluate(self, parameters, config):
        self._set_model_params(
            self.tree_model,
            self.nn_model_global,
            {"tree": [
                self.tree_model.get_params()["max_depth"],
                self.tree_model.get_params()["min_samples_split"],
                self.tree_model.get_params()["min_samples_leaf"],
            ], "nn": parameters}
        )

        round_number = int(config.get("server_round", 1))
        explain_only = bool(config.get("explain_only", False))

        if explain_only:
            if not os.path.exists(self.local_ckpt):
                raise RuntimeError(
                    f"[CLIENTE {self.client_id}] ❌ No existe ckpt local para explicar: {self.local_ckpt}"
                )
            state = torch.load(self.local_ckpt, map_location="cpu")
            self.nn_model_local.load_state_dict(state)
            self.nn_model_local.eval()
            self.local_trained = True
            print(f"[CLIENTE {self.client_id}] 📦 LOCAL baseline recargado en evaluate()")

        if explain_only:
            self.nn_model_global.eval()

        if "supertree" in config:
            try:
                print("Recibiendo supertree....")
                supertree_dict = json.loads(config["supertree"])
                self.received_supertree = SuperTree.convert_SuperNode_to_Node(
                    SuperTree.SuperNode.from_dict(supertree_dict)
                )
                self.global_mapping = json.loads(config["global_mapping"])
                self.feature_names = json.loads(config["feature_names"])
            except Exception as e:
                print(f"[CLIENTE {self.client_id}] ❌ Error al recibir SuperTree: {e}")

        # 🔹 CASO 1: rondas de entrenamiento
        if not explain_only:
            self.tree_model.fit(self.X_train, self.y_train)

            supertree = SuperTree()
            root_node = supertree.rec_buildTree(
                self.tree_model,
                list(range(self.X_train.shape[1])),
                len(self.unique_labels)
            )
            root_node = supertree.prune_redundant_leaves_local(root_node)

            self._save_local_tree(
                root_node,
                round_number,
                self._FEATURES,
                self.numeric_features,
                scaler=None,
                unique_labels=self._UNIQUE_LABELS,
                encoder=self.encoder
            )
            tree_json = json.dumps([root_node.to_dict()])

            return 0.0, len(self.X_test), {
                f"tree_ensemble_{self.client_id}": tree_json,
                f"encoded_feature_names_{self.client_id}": json.dumps(self._FEATURES),
                f"numeric_features_{self.client_id}": json.dumps(self.numeric_features),
                f"unique_labels_{self.client_id}": json.dumps(self.unique_labels),
                f"distinct_values_{self.client_id}": json.dumps(self.encoder.dataset_descriptor["categorical"]),
            }

        # 🔹 CASO 2: ronda final
        print(f"[CLIENTE {self.client_id}] 🔍 Ronda final: solo explicaciones")

        self.tree_model.fit(self.X_train, self.y_train)
        y_pred_tree_local = self.tree_model.predict(self.X_test)

        self.local_metrics = {
            "acc_local_tree": accuracy_score(self.y_test, y_pred_tree_local),
            "prec_local_tree": precision_score(self.y_test, y_pred_tree_local, average="weighted", zero_division=0),
            "rec_local_tree": recall_score(self.y_test, y_pred_tree_local, average="weighted", zero_division=0),
            "f1_local_tree": f1_score(self.y_test, y_pred_tree_local, average="weighted", zero_division=0),
        }

        if self.received_supertree is not None:

            self.nn_model_local.eval()
            self.nn_model_global.eval()

            self.bb_local = TorchNNWrapper(
                model=self.nn_model_local,
                num_idx=self.num_idx,
                mean=self.scaler_nn_mean,
                scale=self.scaler_nn_scale
            )

            self.bb_global = TorchNNWrapper(
                model=self.nn_model_global,
                num_idx=self.num_idx,
                mean=self.scaler_nn_mean,
                scale=self.scaler_nn_scale
            )

            # métricas NN
            self.local_metrics["acc_nn_local_localTest"]  = accuracy_score(self.y_test, self.bb_local.predict(self.X_test))
            self.local_metrics["acc_nn_global_localTest"] = accuracy_score(self.y_test, self.bb_global.predict(self.X_test))
            self.local_metrics["acc_nn_local_globalTest"]  = accuracy_score(self.y_test_global, self.bb_local.predict(self.X_test_global))
            self.local_metrics["acc_nn_global_globalTest"] = accuracy_score(self.y_test_global, self.bb_global.predict(self.X_test_global))

            n_bg = min(50, len(self.X_train))
            idx_bg = np.random.choice(len(self.X_train), n_bg, replace=False)
            self.shap_background = self.X_train[idx_bg]

            self.shap_explainer_local = shap.KernelExplainer(
                self.bb_local.predict_proba,
                self.shap_background
            )

            self.shap_explainer_global = shap.KernelExplainer(
                self.bb_global.predict_proba,
                self.shap_background
            )

            categorical_features_lime = [
                i for i, f in enumerate(self.feature_names)
                if "_" in f
            ]

            self.lime_explainer = LimeTabularExplainer(
                training_data=self.X_train,
                feature_names=self.feature_names,
                class_names=self.unique_labels,
                categorical_features=categorical_features_lime,
                mode="classification",
                random_state=42
            )

            self.anchor_explainer_global = AnchorTabular(
                predictor=self.bb_global.predict,
                feature_names=self.feature_names,
                seed=42
            )
            self.anchor_explainer_global.fit(self.X_train)

            self.anchor_explainer_local = AnchorTabular(
                predictor=self.bb_local.predict,
                feature_names=self.feature_names,
                seed=42
            )
            self.anchor_explainer_local.fit(self.X_train)

            # métricas de modelo pre-calculadas una sola vez
            y_pred_supertree        = self.received_supertree.predict(self.X_test)
            y_pred_superTree_global = self.received_supertree.predict(self.X_test_global)
            y_pred_localTree_global = self.tree_model.predict(self.X_test_global)

            self.model_metrics = {
                "acc_supertree_localTest":   accuracy_score(self.y_test, y_pred_supertree),
                "prec_supertree_localTest":  precision_score(self.y_test, y_pred_supertree, average="weighted", zero_division=0),
                "rec_supertree_localTest":   recall_score(self.y_test, y_pred_supertree, average="weighted", zero_division=0),
                "f1_supertree_localTest":    f1_score(self.y_test, y_pred_supertree, average="weighted", zero_division=0),

                "acc_super_globalTest":      accuracy_score(self.y_test_global, y_pred_superTree_global),
                "prec_super_globalTest":     precision_score(self.y_test_global, y_pred_superTree_global, average="weighted", zero_division=0),
                "rec_super_globalTest":      recall_score(self.y_test_global, y_pred_superTree_global, average="weighted", zero_division=0),
                "f1_super_globalTest":       f1_score(self.y_test_global, y_pred_superTree_global, average="weighted", zero_division=0),

                "acc_localTree_globalTest":  accuracy_score(self.y_test_global, y_pred_localTree_global),
                "prec_localTree_globalTest": precision_score(self.y_test_global, y_pred_localTree_global, average="weighted", zero_division=0),
                "rec_localTree_globalTest":  recall_score(self.y_test_global, y_pred_localTree_global, average="weighted", zero_division=0),
                "f1_localTree_globalTest":   f1_score(self.y_test_global, y_pred_localTree_global, average="weighted", zero_division=0),
            }

            self.explain_all_test_instances(config)

        return 0.0, len(self.X_test), {}

    def _explain_one_instance(self, num_row, config, save_trees=False):

        decoded = self.decode_onehot_instance(
            self.X_test[num_row],
            self.numeric_features,
            self.encoder,
            None,
            self.feature_names
        )

        row = np.asarray(self.X_test[num_row], dtype=np.float32)

        pred_class_idx_local, pred_class_local = Explainer_metrics.predict_bbox(
            self.bb_local, row, self.label_encoder
        )

        pred_class_idx_global, pred_class_global = Explainer_metrics.predict_bbox(
            self.bb_global, row, self.label_encoder
        )

        local_tabular_dataset = Explainer_metrics.build_tabular_dataset(
            self.X_train,
            self.y_train_nn,
            self.feature_names,
            self.label_encoder
        )

        x_instance = pd.Series(self.X_test[num_row], index=self.feature_names)
        round_number = config.get("server_round", 1)

        # 2) Vecindad GLOBAL (LORE)
        bbox_global_for_Z = sklearn_classifier_bbox.sklearnBBox(self.bb_global)

        lore_vecindad_global = TabularGeneticGeneratorLore(
            bbox_global_for_Z,
            local_tabular_dataset,
            random_seed=42 + num_row,
        )

        explanation_global = lore_vecindad_global.explain_instance(
            x_instance,
            merge=True,
            num_classes=len(self._UNIQUE_LABELS),
            feature_names=self.feature_names,
            categorical_features=list(self.global_mapping.keys()),
            global_mapping=self.global_mapping,
            UNIQUE_LABELS=self._UNIQUE_LABELS,
            client_id=self.client_id,
            round_number=round_number
        )

        lore_tree_global = explanation_global["merged_tree"]
        Z_global = explanation_global["neighborhood_Z"]
        y_bb_global = explanation_global["neighborhood_Yb"]
        dfZ_global = pd.DataFrame(Z_global, columns=self.feature_names)

        if save_trees:
            self.save_lore_tree_image(
                lore_tree_global.root,
                round_number,
                self.feature_names,
                self.numeric_features,
                self._UNIQUE_LABELS,
                self.encoder,
                folder="lore_tree_global"
            )

        # 3) SHAP GLOBAL
        top_features_global = Explainer_metrics.compute_top_shap_features(
            self.shap_explainer_global,
            row,
            pred_class_idx_global,
            self.feature_names
        )

        # 4) LIME GLOBAL
        lime_rules_global = Explainer_metrics.compute_lime_rules(
            self.lime_explainer,
            row,
            self.bb_global.predict_proba,
            self.feature_names
        )

        # 5) ANCHOR GLOBAL
        anchor_rules_global = Explainer_metrics.compute_anchor_rules(
            self.anchor_explainer_global,
            row,
            0.85,
            self.feature_names
        )

        # 6) Vecindad LOCAL (LORE)
        bbox_local = sklearn_classifier_bbox.sklearnBBox(self.bb_local)

        lore_vecindad_local = TabularGeneticGeneratorLore(
            bbox_local,
            local_tabular_dataset,
            random_seed=42 + num_row,
        )

        explanation_local = lore_vecindad_local.explain_instance(
            x_instance,
            merge=True,
            num_classes=len(self._UNIQUE_LABELS),
            feature_names=self.feature_names,
            categorical_features=list(self.global_mapping.keys()),
            global_mapping=self.global_mapping,
            UNIQUE_LABELS=self._UNIQUE_LABELS,
            client_id=self.client_id,
            round_number=round_number
        )

        lore_tree_local = explanation_local["merged_tree"]
        Z_local = explanation_local["neighborhood_Z"]
        y_bb_local = explanation_local["neighborhood_Yb"]
        dfZ_local = pd.DataFrame(Z_local, columns=self.feature_names)

        if save_trees:
            self.save_lore_tree_image(
                lore_tree_local.root,
                round_number,
                self.feature_names,
                self.numeric_features,
                self._UNIQUE_LABELS,
                self.encoder,
                folder="lore_tree_local"
            )

        # 7) SHAP LOCAL
        top_features_local = Explainer_metrics.compute_top_shap_features(
            self.shap_explainer_local,
            row,
            pred_class_idx_local,
            self.feature_names
        )

        # 8) LIME LOCAL
        lime_rules_local = Explainer_metrics.compute_lime_rules(
            self.lime_explainer,
            row,
            self.bb_local.predict_proba,
            self.feature_names
        )

        # 9) ANCHOR LOCAL
        anchor_rules_local = Explainer_metrics.compute_anchor_rules(
            self.anchor_explainer_local,
            row,
            0.85,
            self.feature_names
        )

        # 10) Árboles → string
        lore_tree_local_str = self.tree_to_str(
            lore_tree_local.root,
            self.feature_names,
            numeric_features=self.numeric_features,
            scaler=None,
            global_mapping=self.global_mapping,
            unique_labels=self.unique_labels
        )

        lore_tree_global_str = self.tree_to_str(
            lore_tree_global.root,
            self.feature_names,
            numeric_features=self.numeric_features,
            scaler=None,
            global_mapping=self.global_mapping,
            unique_labels=self.unique_labels
        )

        supertree_str = self.tree_to_str(
            self.received_supertree,
            self.feature_names,
            numeric_features=self.numeric_features,
            scaler=None,
            global_mapping=self.global_mapping,
            unique_labels=self.unique_labels
        )

        # 11) Extracción de reglas
        rules_lore_local = self.extract_rules_from_str(
            lore_tree_local_str,
            target_class_label=pred_class_local
        )

        rules_lore_global = self.extract_rules_from_str(
            lore_tree_global_str,
            target_class_label=pred_class_global
        )

        rules_supertree_global = self.extract_rules_from_str(
            supertree_str,
            target_class_label=pred_class_global
        )

        # 12) Factual rule
        regla_factual_lore_local = None
        for r in rules_lore_local:
            if ReglaEvaluator.cumple_regla(decoded, r):
                regla_factual_lore_local = r
                break

        regla_factual_lore_global = None
        for r in rules_lore_global:
            if ReglaEvaluator.cumple_regla(decoded, r):
                regla_factual_lore_global = r
                break

        rules_factual_local = [regla_factual_lore_local] if regla_factual_lore_local else []
        rules_factual_global = [regla_factual_lore_global] if regla_factual_lore_global else []

        # Guardar para métricas agregadas
        self.exp_shap_local.append(top_features_local)
        self.exp_shap_global.append(top_features_global)
        self.exp_lime_local.append(lime_rules_local)
        self.exp_lime_global.append(lime_rules_global)
        self.exp_anchor_local.append(anchor_rules_local)
        self.exp_anchor_global.append(anchor_rules_global)
        self.exp_lore_local.append(rules_factual_local[0] if rules_factual_local else [])
        self.exp_lore_global.append(rules_factual_global[0] if rules_factual_global else [])

        # 13) Jaccard / Coverage
        has_factual = bool(rules_factual_local) and bool(rules_factual_global)

        if not has_factual:
            jaccard_cov_global = covL_g = covG_g = covInter_g = covUnion_g = np.nan
            jaccard_cov_local  = covL_l = covG_l = covInter_l = covUnion_l = np.nan
        else:
            jaccard_cov_global, covL_g, covG_g, covInter_g, covUnion_g = \
                Explainer_metrics.compute_rule_overlap(dfZ_global, rules_factual_local[0], rules_factual_global[0])
            jaccard_cov_local, covL_l, covG_l, covInter_l, covUnion_l = \
                Explainer_metrics.compute_rule_overlap(dfZ_local, rules_factual_local[0], rules_factual_global[0])

        # 14) Silhouette
        x = self.X_test[num_row]

        silhouette_local = Explainer_metrics.compute_silhouette(
            x, dfZ_local, y_bb_local, pred_class_idx_local
        )

        silhouette_global = Explainer_metrics.compute_silhouette(
            x, dfZ_global, y_bb_global, pred_class_idx_global
        )

        # 15) Métricas de modelo (pre-calculadas)
        m = self.model_metrics

        # 16) CSV
        row = {
            "round":     int(round_number),
            "dataset":   self._DATASET_NAME,
            "client_id": int(self.client_id),

            "bbox_pred_class_global": str(pred_class_global),
            "bbox_pred_class_local":  str(pred_class_local),

            "silhouette_global": float(silhouette_global),
            "silhouette_local":  float(silhouette_local),

            # SuperTree / local test
            "acc_superTree_localTest":  float(m["acc_supertree_localTest"]),
            "prec_superTree_localTest": float(m["prec_supertree_localTest"]),
            "rec_superTree_localTest":  float(m["rec_supertree_localTest"]),
            "f1_superTree_localTest":   float(m["f1_supertree_localTest"]),

            # SuperTree / global test
            "acc_superTree_globalTest":  float(m["acc_super_globalTest"]),
            "prec_superTree_globalTest": float(m["prec_super_globalTest"]),
            "rec_superTree_globalTest":  float(m["rec_super_globalTest"]),
            "f1_superTree_globalTest":   float(m["f1_super_globalTest"]),

            # Local tree / local test
            "acc_localTree_localTest":  float(self.local_metrics["acc_local_tree"]),
            "prec_localTree_localTest": float(self.local_metrics["prec_local_tree"]),
            "rec_localTree_localTest":  float(self.local_metrics["rec_local_tree"]),
            "f1_localTree_localTest":   float(self.local_metrics["f1_local_tree"]),

            # Local tree / global test
            "acc_localTree_globalTest":  float(m["acc_localTree_globalTest"]),
            "prec_localTree_globalTest": float(m["prec_localTree_globalTest"]),
            "rec_localTree_globalTest":  float(m["rec_localTree_globalTest"]),
            "f1_localTree_globalTest":   float(m["f1_localTree_globalTest"]),

            # NN
            "acc_nn_local_localTest":   float(self.local_metrics["acc_nn_local_localTest"]),
            "acc_nn_global_localTest":  float(self.local_metrics["acc_nn_global_localTest"]),
            "acc_nn_local_globalTest":  float(self.local_metrics["acc_nn_local_globalTest"]),
            "acc_nn_global_globalTest": float(self.local_metrics["acc_nn_global_globalTest"]),

            # Jaccard / coverage
            "jaccard_cov_globalZ": float(jaccard_cov_global),
            "covL_globalZ":        float(covL_g),
            "covG_globalZ":        float(covG_g),
            "covInter_globalZ":    float(covInter_g),
            "covUnion_globalZ":    float(covUnion_g),

            "jaccard_cov_localZ": float(jaccard_cov_local),
            "covL_localZ":        float(covL_l),
            "covG_localZ":        float(covG_l),
            "covInter_localZ":    float(covInter_l),
            "covUnion_localZ":    float(covUnion_l),
        }

        self._append_client_csv(row, filename="Balanced")
        return row

    def explain_all_test_instances(self, config, only_idx=None):
        results = []
        self._reset_explanation_buffers()
        indices, desc_text, save_trees_flag = self._get_explain_indices(only_idx)

        for i in tqdm(indices, desc=desc_text):
            try:
                row = self._explain_one_instance(i, config, save_trees=save_trees_flag)
                results.append(row)
            except Exception as e:
                print(f"[Cliente {self.client_id}] ⚠️ Error en instancia {i}: {e}")

        metrics = self._compute_explanation_metrics()
        df = pd.read_csv(f"results/metrics_Balanced_cliente_{self.client_id}.csv")
        self._save_summary_metrics(df, metrics)
        return df

    def _reset_explanation_buffers(self):
        self.exp_shap_local = []
        self.exp_shap_global = []
        self.exp_lime_local = []
        self.exp_lime_global = []
        self.exp_anchor_local = []
        self.exp_anchor_global = []
        self.exp_lore_local = []
        self.exp_lore_global = []

    def _get_explain_indices(self, only_idx):
        if only_idx is None:
            return (
                range(len(self.X_test)),
                f"Cliente {self.client_id} explicando test completo",
                False
            )
        return (
            [only_idx],
            f"Cliente {self.client_id} explicando instancia {only_idx}",
            True
        )

    def _compute_explanation_metrics(self):
        metrics = {}
        X_test_np = np.asarray(self.X_test, dtype=np.float32)

        # GLOBAL
        metrics["stability_shap_global"]     = ClientUtilsMixin.stability_score(self.y_test, self.exp_shap_global)
        metrics["stability_lime_global"]     = ClientUtilsMixin.stability_score(self.y_test, self.exp_lime_global)
        metrics["stability_anchor_global"]   = ClientUtilsMixin.stability_score(self.y_test, self.exp_anchor_global)
        metrics["stability_lore_global"]     = ClientUtilsMixin.stability_score(self.y_test, self.exp_lore_global)

        metrics["separability_shap_global"]   = ClientUtilsMixin.separability_score(self.exp_shap_global)
        metrics["separability_lime_global"]   = ClientUtilsMixin.separability_score(self.exp_lime_global)
        metrics["separability_anchor_global"] = ClientUtilsMixin.separability_score(self.exp_anchor_global)
        metrics["separability_lore_global"]   = ClientUtilsMixin.separability_score(self.exp_lore_global)

        metrics["similarity_shap_global"]     = ClientUtilsMixin.similarity_score(X_test_np, self.exp_shap_global)
        metrics["similarity_lime_global"]     = ClientUtilsMixin.similarity_score(X_test_np, self.exp_lime_global)
        metrics["similarity_anchor_global"]   = ClientUtilsMixin.similarity_score(X_test_np, self.exp_anchor_global)
        metrics["similarity_lore_global"]     = ClientUtilsMixin.similarity_score(X_test_np, self.exp_lore_global)

        metrics["ratio_has_anchor_global"] = float(np.mean([bool(a) for a in self.exp_anchor_global]))

        # LOCAL
        metrics["stability_shap_local"]     = ClientUtilsMixin.stability_score(self.y_test, self.exp_shap_local)
        metrics["stability_lime_local"]     = ClientUtilsMixin.stability_score(self.y_test, self.exp_lime_local)
        metrics["stability_anchor_local"]   = ClientUtilsMixin.stability_score(self.y_test, self.exp_anchor_local)
        metrics["stability_lore_local"]     = ClientUtilsMixin.stability_score(self.y_test, self.exp_lore_local)

        metrics["separability_shap_local"]   = ClientUtilsMixin.separability_score(self.exp_shap_local)
        metrics["separability_lime_local"]   = ClientUtilsMixin.separability_score(self.exp_lime_local)
        metrics["separability_anchor_local"] = ClientUtilsMixin.separability_score(self.exp_anchor_local)
        metrics["separability_lore_local"]   = ClientUtilsMixin.separability_score(self.exp_lore_local)

        metrics["similarity_shap_local"]     = ClientUtilsMixin.similarity_score(X_test_np, self.exp_shap_local)
        metrics["similarity_lime_local"]     = ClientUtilsMixin.similarity_score(X_test_np, self.exp_lime_local)
        metrics["similarity_anchor_local"]   = ClientUtilsMixin.similarity_score(X_test_np, self.exp_anchor_local)
        metrics["similarity_lore_local"]     = ClientUtilsMixin.similarity_score(X_test_np, self.exp_lore_local)

        metrics["ratio_has_anchor_local"] = float(np.mean([bool(a) for a in self.exp_anchor_local]))

        return metrics

    def _save_summary_metrics(self, df, metrics):
        mean_metrics = df.mean(numeric_only=True)
        count_metrics = df.count(numeric_only=True)

        mean_df = pd.DataFrame({"mean": mean_metrics, "count": count_metrics})

        mean_df.loc["ratio_has_factual_globalZ", ["mean", "count"]] = [
            df["jaccard_cov_globalZ"].notna().mean(),
            int(df["jaccard_cov_globalZ"].notna().sum()),
        ]

        mean_df.loc["ratio_has_factual_localZ", ["mean", "count"]] = [
            df["jaccard_cov_localZ"].notna().mean(),
            int(df["jaccard_cov_localZ"].notna().sum()),
        ]

        for k, v in metrics.items():
            v = float(v) if v is not None else float("nan")
            mean_df.loc[k, "mean"] = v
            mean_df.loc[k, "count"] = 1

        mean_df.to_csv(
            f"results/metrics_cliente_{self.client_id}_balanced_mean.csv",
            index_label="metric"
        )


# ====================================================================
# FACTORY — punto de entrada público
# ====================================================================

def make_client_app_baseline(
    dataset_name,
    class_column,
    num_clients,
    num_train_rounds,
    unique_labels,
    features,
    load_data_fn,
    set_params_fn,
    get_params_fn,
    # ── Label noise (opcional) ──
    label_noise_mode=None,      # None, "symmetric" o "asymmetric"
    label_noise_rate=0.0,       # 0.0 = sin ruido
):
    """
    Crea un ClientApp de Flower parametrizado.
    NN local + NN global (FedAvg), 4 explainers, árboles locales para debug.

    Label noise: si label_noise_mode no es None y label_noise_rate > 0,
    se inyecta ruido en y_train de cada cliente antes de entrenar.
    """

    def client_fn(context: Context):
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]

        np.random.seed(42)
        random.seed(42)

        (X_train, y_train,
         X_test_local, y_test_local,
         X_test_global, y_test_global,
         dataset, feature_names, label_encoder,
         scaler, numeric_features, encoder, preprocessor) = load_data_fn(
            flower_dataset_name=dataset_name,
            class_col=class_column,
            partition_id=partition_id,
            num_partitions=num_clients,
        )

        # ── Label noise (si está activado) ──
        if label_noise_mode is not None and label_noise_rate > 0:
            from lore_sa.client_utils.label_noise import LabelNoiseInjector

            n_clases = len(unique_labels)
            mapping = LabelNoiseInjector.make_default_mapping(range(n_clases))

            injector = LabelNoiseInjector(
                noise_rate=label_noise_rate,
                mode=label_noise_mode,
                mapping=mapping,
                seed=42 + partition_id,
            )
            y_train = injector.transform(y_train)

        tree_model = DecisionTreeClassifier(max_depth=3, min_samples_split=2, random_state=42)

        num_idx = list(range(len(numeric_features)))
        scaler_nn = StandardScaler().fit(X_train[:, num_idx])

        n_clases_global = len(unique_labels)
        input_dim = X_train.shape[1]
        output_dim = n_clases_global

        nn_model = Net(input_dim, output_dim)

        return FlowerClient(
            tree_model=tree_model,
            nn_model=nn_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test_local,
            y_test=y_test_local,
            X_test_global=X_test_global,
            y_test_global=y_test_global,
            scaler_nn_mean=scaler_nn.mean_,
            scaler_nn_scale=scaler_nn.scale_,
            num_idx=num_idx,
            dataset=dataset,
            client_id=partition_id + 1,
            feature_names=feature_names,
            label_encoder=label_encoder,
            scaler=scaler,
            numeric_features=numeric_features,
            encoder=encoder,
            preprocessor=preprocessor,
            _UNIQUE_LABELS=unique_labels,
            _FEATURES=features,
            _NUM_TRAIN_ROUNDS=num_train_rounds,
            _DATASET_NAME=dataset_name,
            _set_model_params=set_params_fn,
            _get_model_parameters=get_params_fn,
        ).to_client()

    return ClientApp(client_fn=client_fn)