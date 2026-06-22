# ==========================
# 🌼 CLIENTE FLOWER — Explicabilidad
# ==========================
# NN local + NN global (FedAvg), explicadas con SHAP, LIME, ANCHOR y LORE.
# Los árboles locales se mantienen (útil para debug) pero NO se envían al servidor.
# Las explicaciones se ejecutan en la última ronda (explain_only=True).
#
# Uso desde notebook:
#
#   from lore_sa.client_utils.client_BASELINE import make_client_app_baseline
#
#   client_app = make_client_app_baseline(
#       dataset_name=DATASET_NAME,
#       class_column=CLASS_COLUMN,
#       num_clients=NUM_CLIENTS,
#       num_server_rounds=NUM_SERVER_ROUNDS,
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

from lore_sa.client_utils.label_noise import LabelNoiseInjector
from lore_sa.dataset import TabularDataset
from lore_sa.lore import TabularGeneticGeneratorLore
from lore_sa.rule import Expression, Rule
from lore_sa.encoder_decoder import ColumnTransformerEnc

from sklearn.metrics import pairwise_distances

import shap
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular
from tqdm import tqdm

from lore_sa.client_utils import ClientUtilsMixin
from lore_sa.client_utils.cumple_regla import ReglaEvaluator
from lore_sa.client_utils.explanation_metrics import Explainer_metrics
from lore_sa.surrogate.decision_tree import SuperTree


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
# LoreBBox — wrapper que desnormaliza numéricas antes de llamar a la NN
# ====================================================================

class LoreBBox:
    """Bbox wrapper para LORE: recibe datos con numéricas normalizadas [0,1],
    las desnormaliza antes de pasarlas a la NN (que normaliza internamente)."""
    def __init__(self, nn_model, lore_scaler, num_idx):
        self.nn = nn_model
        self.scaler = lore_scaler
        self.num_idx = np.asarray(num_idx, dtype=int)

    def _denorm(self, X):
        Xr = np.asarray(X, dtype=np.float32).copy()
        if Xr.ndim == 1:
            Xr = Xr[None, :]
        if len(self.num_idx) > 0:
            Xr[:, self.num_idx] = self.scaler.inverse_transform(Xr[:, self.num_idx])
        return Xr

    def predict(self, X):
        return self.nn.predict(self._denorm(X))

    def predict_proba(self, X):
        return self.nn.predict_proba(self._denorm(X))


# ====================================================================
# FlowerClient
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
        self.preprocessor = preprocessor

        # MinMaxScaler solo para LORE (normaliza numéricas para el GA)
        from sklearn.preprocessing import MinMaxScaler as _MMS
        self.lore_scaler = _MMS()
        if len(self.num_idx) > 0:
            self.lore_scaler.fit(self.X_train[:, self.num_idx])

        # global_mapping para LORE (construido localmente desde el encoder)
        cat_desc = encoder.dataset_descriptor.get("categorical", {})
        self.global_mapping = {
            feat: sorted(info["distinct_values"])
            for feat, info in cat_desc.items()
            if "distinct_values" in info and info["distinct_values"]
        }

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

        # Árbol local (se queda para debug, no se envía al servidor)
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
        print(f"[CLIENTE {self.client_id}] evaluate() round={round_number} explain_only={explain_only}")

        if explain_only:
            if not os.path.exists(self.local_ckpt):
                raise RuntimeError(
                    f"[CLIENTE {self.client_id}] ❌ No existe ckpt local para explicar: {self.local_ckpt}"
                )
            state = torch.load(self.local_ckpt, map_location="cpu")
            self.nn_model_local.load_state_dict(state)
            self.nn_model_local.eval()
            self.nn_model_global.eval()
            self.local_trained = True
            # print(f"[CLIENTE {self.client_id}] 📦 LOCAL baseline recargado en evaluate()")

        # 🔹 CASO 1: rondas de entrenamiento (solo devolvemos métricas vacías)
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
            
            return 0.0, len(self.X_test), {}

        # 🔹 CASO 2: ronda final → explicaciones
        # print(f"[CLIENTE {self.client_id}] 🔍 Ronda final: solo explicaciones")

        self.tree_model.fit(self.X_train, self.y_train)
        y_pred_tree_local = self.tree_model.predict(self.X_test)

        self.local_metrics = {
            "acc_local_tree": accuracy_score(self.y_test, y_pred_tree_local),
            "prec_local_tree": precision_score(self.y_test, y_pred_tree_local, average="weighted", zero_division=0),
            "rec_local_tree": recall_score(self.y_test, y_pred_tree_local, average="weighted", zero_division=0),
            "f1_local_tree": f1_score(self.y_test, y_pred_tree_local, average="weighted", zero_division=0),
        }

        # Wrappers NN
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

        # Métricas NN
        self.local_metrics["acc_nn_local_localTest"]  = accuracy_score(self.y_test, self.bb_local.predict(self.X_test))
        self.local_metrics["acc_nn_global_localTest"] = accuracy_score(self.y_test, self.bb_global.predict(self.X_test))
        self.local_metrics["acc_nn_local_globalTest"]  = accuracy_score(self.y_test_global, self.bb_local.predict(self.X_test_global))
        self.local_metrics["acc_nn_global_globalTest"] = accuracy_score(self.y_test_global, self.bb_global.predict(self.X_test_global))

        self.local_metrics["pred_agreement_localTest"] = float(np.mean(
            self.bb_local.predict(self.X_test) == self.bb_global.predict(self.X_test)
        ))
        self.local_metrics["pred_agreement_globalTest"] = float(np.mean(
            self.bb_local.predict(self.X_test_global) == self.bb_global.predict(self.X_test_global)
        ))


        # SHAP — dos backgrounds: test local y test global
        rng = np.random.RandomState(42)

        n_bg_tl = min(50, len(self.X_test))
        idx_bg_tl = rng.choice(len(self.X_test), n_bg_tl, replace=False)
        shap_bg_testlocal = self.X_test[idx_bg_tl]

        n_bg_tg = min(50, len(self.X_test_global))
        idx_bg_tg = rng.choice(len(self.X_test_global), n_bg_tg, replace=False)
        shap_bg_testglobal = self.X_test_global[idx_bg_tg]

        self.shap_explainer_testlocal_local  = shap.KernelExplainer(self.bb_local.predict_proba,  shap_bg_testlocal)
        self.shap_explainer_testlocal_global = shap.KernelExplainer(self.bb_global.predict_proba, shap_bg_testlocal)
        self.shap_explainer_testglobal_local  = shap.KernelExplainer(self.bb_local.predict_proba,  shap_bg_testglobal)
        self.shap_explainer_testglobal_global = shap.KernelExplainer(self.bb_global.predict_proba, shap_bg_testglobal)

        # LIME — dos backgrounds: test local y test global
        categorical_features_lime = [
            i for i, f in enumerate(self.feature_names) if "_" in f
        ]
        self.lime_explainer_testlocal = LimeTabularExplainer(
            training_data=self.X_test,
            feature_names=self.feature_names,
            class_names=self.unique_labels,
            categorical_features=categorical_features_lime,
            mode="classification",
            feature_selection="none",
            random_state=42
        )
        self.lime_explainer_testglobal = LimeTabularExplainer(
            training_data=self.X_test_global,
            feature_names=self.feature_names,
            class_names=self.unique_labels,
            categorical_features=categorical_features_lime,
            mode="classification",
            feature_selection="none",
            random_state=42
        )

        # Rangos de features numéricas para Jaccard estructural de Anchor
        self.feature_ranges = {
            self.feature_names[i]: (float(self.X_train[:, i].min()), float(self.X_train[:, i].max()))
            for i in range(len(self.feature_names))
            if self.feature_names[i] in self.numeric_features
        }

        # ANCHOR — dos backgrounds: test local y test global
        self.anchor_explainer_testlocal_local = AnchorTabular(
            predictor=self.bb_local.predict,
            feature_names=self.feature_names, seed=42)
        self.anchor_explainer_testlocal_local.fit(self.X_test)

        self.anchor_explainer_testlocal_global = AnchorTabular(
            predictor=self.bb_global.predict,
            feature_names=self.feature_names, seed=42)
        self.anchor_explainer_testlocal_global.fit(self.X_test)

        self.anchor_explainer_testglobal_local = AnchorTabular(
            predictor=self.bb_local.predict,
            feature_names=self.feature_names, seed=42)
        self.anchor_explainer_testglobal_local.fit(self.X_test_global)

        self.anchor_explainer_testglobal_global = AnchorTabular(
            predictor=self.bb_global.predict,
            feature_names=self.feature_names, seed=42)
        self.anchor_explainer_testglobal_global.fit(self.X_test_global)

        # Métricas de modelo (árboles locales, sin SuperTree)
        y_pred_localTree_global = self.tree_model.predict(self.X_test_global)

        self.model_metrics = {
            "acc_localTree_localTest": accuracy_score(self.y_test, y_pred_tree_local),
            "acc_localTree_globalTest": accuracy_score(self.y_test_global, y_pred_localTree_global),
        }

        only_idx = config.get("only_idx", None)
        self.explain_all_test_instances(config, only_idx=only_idx)

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
            self.bb_local, row, self.label_encoder)
        pred_class_idx_global, pred_class_global = Explainer_metrics.predict_bbox(
            self.bb_global, row, self.label_encoder)

        # ── Dataset normalizado para LORE (numéricas en [0,1]) ──
        X_train_lore = self.X_train.copy()
        if len(self.num_idx) > 0:
            X_train_lore[:, self.num_idx] = self.lore_scaler.transform(self.X_train[:, self.num_idx])

        local_tabular_dataset = Explainer_metrics.build_tabular_dataset(
            X_train_lore, self.y_train_nn, self.feature_names, self.label_encoder)

        row_lore = row.copy()
        if len(self.num_idx) > 0:
            row_lore[self.num_idx] = self.lore_scaler.transform(row[self.num_idx].reshape(1, -1))[0]
        x_instance_lore = pd.Series(row_lore, index=self.feature_names)
        x_instance = pd.Series(self.X_test[num_row], index=self.feature_names)
        round_number = config.get("server_round", 1)

        # ── Vecindad GLOBAL (LORE) ──
        np.random.seed(42 + num_row)
        bbox_global_for_Z = LoreBBox(self.bb_global, self.lore_scaler, self.num_idx)
        lore_vecindad_global = TabularGeneticGeneratorLore(
            bbox_global_for_Z, local_tabular_dataset, random_seed=42 + num_row)

        explanation_global = lore_vecindad_global.explain_instance(
            x_instance_lore, merge=True, num_classes=len(self._UNIQUE_LABELS),
            feature_names=self.feature_names,
            categorical_features=list(self.global_mapping.keys()),
            global_mapping=self.global_mapping, UNIQUE_LABELS=self._UNIQUE_LABELS,
            client_id=self.client_id, round_number=round_number)

        lore_tree_global = explanation_global["merged_tree"]
        ratio_opposite_neighborhood_global = explanation_global["ratio_opposite_neighborhood"]
        Z_global_norm = explanation_global["neighborhood_Z"]
        y_bb_global = explanation_global["neighborhood_Yb"]
        # Desnormalizar Z para coverage y reglas (espacio raw)
        Z_global = Z_global_norm.copy()
        if len(self.num_idx) > 0:
            Z_global[:, self.num_idx] = self.lore_scaler.inverse_transform(Z_global_norm[:, self.num_idx])
        dfZ_global = pd.DataFrame(Z_global, columns=self.feature_names)

        if save_trees:
            self.save_lore_tree_image(
                lore_tree_global.root, round_number, self.feature_names,
                self.numeric_features, self._UNIQUE_LABELS, self.encoder,
                folder="lore_tree_global")

        # ── SHAP (testlocal background) ──
        top_features_global, shap_coefs_testlocal_global = Explainer_metrics.compute_shap_all(
            self.shap_explainer_testlocal_global, row, pred_class_idx_global, self.feature_names)

        # ── LIME GLOBAL ──
        lime_seed = 42 + num_row
        lime_rules_global = Explainer_metrics.compute_lime_rules(
            self.lime_explainer_testlocal, row, self.bb_global.predict_proba, self.feature_names)
        lime_coefs_testlocal_global = Explainer_metrics.compute_lime_coefs(
            self.lime_explainer_testlocal, row, self.bb_global.predict_proba,
            len(self.feature_names), pred_class_idx_global, random_state=lime_seed)
        lime_coefs_testglobal_global = Explainer_metrics.compute_lime_coefs(
            self.lime_explainer_testglobal, row, self.bb_global.predict_proba,
            len(self.feature_names), pred_class_idx_global, random_state=lime_seed)

        # ── ANCHOR GLOBAL ──
        anchor_rules_testlocal_global, anchor_prec_testlocal_global, anchor_cov_testlocal_global = \
            Explainer_metrics.compute_anchor_all(self.anchor_explainer_testlocal_global, row, 0.85, self.feature_names)
        anchor_rules_testglobal_global, anchor_prec_testglobal_global, anchor_cov_testglobal_global = \
            Explainer_metrics.compute_anchor_all(self.anchor_explainer_testglobal_global, row, 0.85, self.feature_names)

        # ── Vecindad LOCAL (LORE) ──
        np.random.seed(42 + num_row)
        bbox_local = LoreBBox(self.bb_local, self.lore_scaler, self.num_idx)
        lore_vecindad_local = TabularGeneticGeneratorLore(
            bbox_local, local_tabular_dataset, random_seed=42 + num_row)

        explanation_local = lore_vecindad_local.explain_instance(
            x_instance_lore, merge=True, num_classes=len(self._UNIQUE_LABELS),
            feature_names=self.feature_names,
            categorical_features=list(self.global_mapping.keys()),
            global_mapping=self.global_mapping, UNIQUE_LABELS=self._UNIQUE_LABELS,
            client_id=self.client_id, round_number=round_number)

        lore_tree_local = explanation_local["merged_tree"]
        ratio_opposite_neighborhood_local = explanation_local["ratio_opposite_neighborhood"]
        Z_local_norm = explanation_local["neighborhood_Z"]
        y_bb_local = explanation_local["neighborhood_Yb"]
        # Desnormalizar Z para coverage y reglas (espacio raw)
        Z_local = Z_local_norm.copy()
        if len(self.num_idx) > 0:
            Z_local[:, self.num_idx] = self.lore_scaler.inverse_transform(Z_local_norm[:, self.num_idx])
        dfZ_local = pd.DataFrame(Z_local, columns=self.feature_names)

        # tree_fidelity: usar Z normalizado (el árbol se entrenó en ese espacio)
        tree_fidelity_lore_global = np.nan if lore_tree_global is None or lore_tree_global.root is None else float(np.mean(lore_tree_global.root.predict(Z_global_norm) == y_bb_global))
        tree_fidelity_lore_local = np.nan if lore_tree_local is None or lore_tree_local.root is None else float(np.mean(lore_tree_local.root.predict(Z_local_norm) == y_bb_local))

        if save_trees:
            self.save_lore_tree_image(
                lore_tree_local.root, round_number, self.feature_names,
                self.numeric_features, self._UNIQUE_LABELS, self.encoder,
                folder="lore_tree_local")

        # ── SHAP LOCAL (testlocal background) + testglobal ──
        top_features_local, shap_coefs_testlocal_local = Explainer_metrics.compute_shap_all(
            self.shap_explainer_testlocal_local, row, pred_class_idx_local, self.feature_names)

        if pred_class_idx_local == pred_class_idx_global:
            delta_shap_testlocal = Explainer_metrics.delta_shap(shap_coefs_testlocal_local, shap_coefs_testlocal_global)
        else:
            delta_shap_testlocal = np.nan

        _, shap_coefs_testglobal_local = Explainer_metrics.compute_shap_all(
            self.shap_explainer_testglobal_local, row, pred_class_idx_local, self.feature_names)
        _, shap_coefs_testglobal_global = Explainer_metrics.compute_shap_all(
            self.shap_explainer_testglobal_global, row, pred_class_idx_global, self.feature_names)

        if pred_class_idx_local == pred_class_idx_global:
            delta_shap_testglobal = Explainer_metrics.delta_shap(shap_coefs_testglobal_local, shap_coefs_testglobal_global)
        else:
            delta_shap_testglobal = np.nan
        
        # ── LIME LOCAL ──
        lime_rules_local = Explainer_metrics.compute_lime_rules(
            self.lime_explainer_testlocal, row, self.bb_local.predict_proba, self.feature_names)
        lime_coefs_testlocal_local = Explainer_metrics.compute_lime_coefs(
            self.lime_explainer_testlocal, row, self.bb_local.predict_proba,
            len(self.feature_names), pred_class_idx_local, random_state=lime_seed)
        lime_coefs_testglobal_local = Explainer_metrics.compute_lime_coefs(
            self.lime_explainer_testglobal, row, self.bb_local.predict_proba,
            len(self.feature_names), pred_class_idx_local, random_state=lime_seed)

        if pred_class_idx_local == pred_class_idx_global:
            delta_lime_testlocal  = Explainer_metrics.delta_lime(lime_coefs_testlocal_local,  lime_coefs_testlocal_global)
            delta_lime_testglobal = Explainer_metrics.delta_lime(lime_coefs_testglobal_local, lime_coefs_testglobal_global)
        else:
            delta_lime_testlocal  = np.nan
            delta_lime_testglobal = np.nan

        # ── ANCHOR LOCAL ──
        anchor_rules_testlocal_local, anchor_prec_testlocal_local, anchor_cov_testlocal_local = \
            Explainer_metrics.compute_anchor_all(self.anchor_explainer_testlocal_local, row, 0.85, self.feature_names)
        anchor_rules_testglobal_local, anchor_prec_testglobal_local, anchor_cov_testglobal_local = \
            Explainer_metrics.compute_anchor_all(self.anchor_explainer_testglobal_local, row, 0.85, self.feature_names)

        # ── Jaccard estructural Anchor (mismo background, distinto modelo) ──
        if pred_class_idx_local == pred_class_idx_global:
            jaccard_anchor_struct_testlocal = Explainer_metrics.compute_structural_jaccard(
                anchor_rules_testlocal_local, anchor_rules_testlocal_global, self.feature_ranges)
            jaccard_anchor_struct_testglobal = Explainer_metrics.compute_structural_jaccard(
                anchor_rules_testglobal_local, anchor_rules_testglobal_global, self.feature_ranges)
            sim_aditiva_anchor_testlocal = Explainer_metrics.compute_additive_similarity(
                anchor_rules_testlocal_local, anchor_rules_testlocal_global, self.feature_ranges)
            sim_aditiva_anchor_testglobal = Explainer_metrics.compute_additive_similarity(
                anchor_rules_testglobal_local, anchor_rules_testglobal_global, self.feature_ranges)
        else:
            jaccard_anchor_struct_testlocal  = np.nan
            jaccard_anchor_struct_testglobal = np.nan
            sim_aditiva_anchor_testlocal     = np.nan
            sim_aditiva_anchor_testglobal    = np.nan

        # ── LORE árboles → string → reglas → factual ──
        lore_tree_local_str = self.tree_to_str(
            lore_tree_local.root, self.feature_names,
            numeric_features=self.numeric_features, scaler=self.lore_scaler,
            global_mapping=self.global_mapping, unique_labels=self.unique_labels)

        lore_tree_global_str = self.tree_to_str(
            lore_tree_global.root, self.feature_names,
            numeric_features=self.numeric_features, scaler=self.lore_scaler,
            global_mapping=self.global_mapping, unique_labels=self.unique_labels)

        rules_lore_local = self.extract_rules_from_str(
            lore_tree_local_str, target_class_label=pred_class_local)
        rules_lore_global = self.extract_rules_from_str(
            lore_tree_global_str, target_class_label=pred_class_global)

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
        self.exp_anchor_testlocal_local.append(anchor_rules_testlocal_local)
        self.exp_anchor_testlocal_global.append(anchor_rules_testlocal_global)
        self.exp_anchor_testglobal_local.append(anchor_rules_testglobal_local)
        self.exp_anchor_testglobal_global.append(anchor_rules_testglobal_global)
        self.exp_lore_local.append(rules_factual_local[0] if rules_factual_local else [])
        self.exp_lore_global.append(rules_factual_global[0] if rules_factual_global else [])

        # ── Jaccard / Coverage ──
        has_factual_local = bool(rules_factual_local)
        has_factual_global = bool(rules_factual_global)
        has_factual = has_factual_local and has_factual_global

        cov_ruleLocal_on_localZ = Explainer_metrics.compute_single_rule_coverage(
            dfZ_local, rules_factual_local[0] if has_factual_local else None
        )
        cov_ruleLocal_on_globalZ = Explainer_metrics.compute_single_rule_coverage(
            dfZ_global, rules_factual_local[0] if has_factual_local else None
        )
        cov_ruleGlobal_on_localZ = Explainer_metrics.compute_single_rule_coverage(
            dfZ_local, rules_factual_global[0] if has_factual_global else None
        )
        cov_ruleGlobal_on_globalZ = Explainer_metrics.compute_single_rule_coverage(
            dfZ_global, rules_factual_global[0] if has_factual_global else None
        )

        if not has_factual or pred_class_idx_local != pred_class_idx_global:
            jaccard_cov_global = covL_g = covG_g = covInter_g = covUnion_g = np.nan
            jaccard_cov_local  = covL_l = covG_l = covInter_l = covUnion_l = np.nan
            jaccard_lore_struct  = np.nan
            sim_aditiva_lore     = np.nan
        else:
            jaccard_cov_global, covL_g, covG_g, covInter_g, covUnion_g = \
                Explainer_metrics.compute_rule_overlap(dfZ_global, rules_factual_local[0], rules_factual_global[0])
            jaccard_cov_local, covL_l, covG_l, covInter_l, covUnion_l = \
                Explainer_metrics.compute_rule_overlap(dfZ_local, rules_factual_local[0], rules_factual_global[0])
            jaccard_lore_struct = Explainer_metrics.compute_structural_jaccard(
                rules_factual_local[0], rules_factual_global[0], self.feature_ranges
            )
            sim_aditiva_lore = Explainer_metrics.compute_additive_similarity(
                rules_factual_local[0], rules_factual_global[0], self.feature_ranges
            )

        # ── Cross-explainer consistency (comentado) ──
        # def _extract_feature_names(rules, feature_names):
        #     feats = set()
        #     if isinstance(rules, list):
        #         for r in rules:
        #             if isinstance(r, str):
        #                 for f in feature_names:
        #                     if f in r:
        #                         feats.add(f)
        #             elif isinstance(r, dict) and "feature" in r:
        #                 feats.add(r["feature"])
        #     return feats
        #
        # def _jaccard_pairwise_mean(feature_sets):
        #     non_empty = [set(f) for f in feature_sets if f]
        #     if len(non_empty) < 2:
        #         return np.nan
        #     scores = []
        #     for i in range(len(non_empty)):
        #         for j in range(i + 1, len(non_empty)):
        #             union = non_empty[i] | non_empty[j]
        #             if not union:
        #                 scores.append(1.0)
        #             else:
        #                 scores.append(len(non_empty[i] & non_empty[j]) / len(union))
        #     return float(np.mean(scores)) if scores else np.nan
        #
        # feats_shap_global = set(top_features_global) if top_features_global else set()
        # feats_lime_global = _extract_feature_names(lime_rules_global, self.feature_names)
        # feats_lore_global = _extract_feature_names(
        #     [rules_factual_global[0]] if rules_factual_global else [], self.feature_names
        # )
        # all_feats_global = [feats_shap_global, feats_lime_global, feats_lore_global]
        # cross_explainer_global = _jaccard_pairwise_mean(all_feats_global)
        #
        # feats_shap_local = set(top_features_local) if top_features_local else set()
        # feats_lime_local = _extract_feature_names(lime_rules_local, self.feature_names)
        # feats_lore_local = _extract_feature_names(
        #     [rules_factual_local[0]] if rules_factual_local else [], self.feature_names
        # )
        # all_feats_local = [feats_shap_local, feats_lime_local, feats_lore_local]
        # cross_explainer_local = _jaccard_pairwise_mean(all_feats_local)

        # ── Silhouette ──
        x = self.X_test[num_row]
        silhouette_local = Explainer_metrics.compute_silhouette(
            x, dfZ_local, y_bb_local, pred_class_idx_local)
        silhouette_global = Explainer_metrics.compute_silhouette(
            x, dfZ_global, y_bb_global, pred_class_idx_global)

        # ── CSV ──
        m = self.model_metrics

        row = {
            "round":     int(round_number),
            "dataset":   self._DATASET_NAME,
            "client_id": int(self.client_id),

            "bbox_pred_class_global": str(pred_class_global),
            "bbox_pred_class_local":  str(pred_class_local),

            "silhouette_global_LORE": float(silhouette_global),
            "silhouette_local_LORE":  float(silhouette_local),
            "tree_fidelity_lore_global": float(tree_fidelity_lore_global),
            "tree_fidelity_lore_local":  float(tree_fidelity_lore_local),

            # Local tree (debug)
            "acc_localTree_localTest":  float(self.local_metrics["acc_local_tree"]),
            "acc_localTree_globalTest": float(m["acc_localTree_globalTest"]),

            # NN
            "acc_nn_local_localTest":   float(self.local_metrics["acc_nn_local_localTest"]),
            "acc_nn_global_localTest":  float(self.local_metrics["acc_nn_global_localTest"]),
            "acc_nn_local_globalTest":  float(self.local_metrics["acc_nn_local_globalTest"]),
            "acc_nn_global_globalTest": float(self.local_metrics["acc_nn_global_globalTest"]),

            "has_factual_local":  int(has_factual_local),
            "has_factual_global": int(has_factual_global),
            "model_disagreement": int(pred_class_idx_local != pred_class_idx_global),
            "ratio_opposite_neighborhood_local":  float(ratio_opposite_neighborhood_local),
            "ratio_opposite_neighborhood_global": float(ratio_opposite_neighborhood_global),
            "cov_ruleLocal_on_localZ_LORE":   float(cov_ruleLocal_on_localZ),
            "cov_ruleLocal_on_globalZ_LORE":  float(cov_ruleLocal_on_globalZ),
            "cov_ruleGlobal_on_localZ_LORE":  float(cov_ruleGlobal_on_localZ),
            "cov_ruleGlobal_on_globalZ_LORE": float(cov_ruleGlobal_on_globalZ),

            # Jaccard / coverage
            "jaccard_cov_globalZ_LORE": float(jaccard_cov_global),
            "covInter_globalZ_LORE":    float(covInter_g),
            "covUnion_globalZ_LORE":    float(covUnion_g),

            "jaccard_cov_localZ_LORE": float(jaccard_cov_local),
            "covInter_localZ_LORE":    float(covInter_l),
            "covUnion_localZ_LORE":    float(covUnion_l),

            "jaccard_lore_struct":        float(jaccard_lore_struct),
            "sim_aditiva_lore":           float(sim_aditiva_lore),

            "delta_lime_testlocal":          float(delta_lime_testlocal),
            "delta_lime_testglobal":         float(delta_lime_testglobal),
            "delta_shap_testlocal":          float(delta_shap_testlocal),
            "delta_shap_testglobal":         float(delta_shap_testglobal),

            "jaccard_anchor_struct_testlocal":  float(jaccard_anchor_struct_testlocal),
            "jaccard_anchor_struct_testglobal": float(jaccard_anchor_struct_testglobal),
            "sim_aditiva_anchor_testlocal":     float(sim_aditiva_anchor_testlocal),
            "sim_aditiva_anchor_testglobal":    float(sim_aditiva_anchor_testglobal),

            "anchor_prec_testlocal_local":   float(anchor_prec_testlocal_local),
            "anchor_prec_testlocal_global":  float(anchor_prec_testlocal_global),
            "anchor_prec_testglobal_local":  float(anchor_prec_testglobal_local),
            "anchor_prec_testglobal_global": float(anchor_prec_testglobal_global),
            
            "anchor_cov_testlocal_local":    float(anchor_cov_testlocal_local),
            "anchor_cov_testlocal_global":   float(anchor_cov_testlocal_global),
            "anchor_cov_testglobal_local":   float(anchor_cov_testglobal_local),
            "anchor_cov_testglobal_global":  float(anchor_cov_testglobal_global),

            # "cross_explainer_global": float(cross_explainer_global),
            # "cross_explainer_local":  float(cross_explainer_local),
            "pred_agreement_localTest":  float(self.local_metrics["pred_agreement_localTest"]),
            "pred_agreement_globalTest": float(self.local_metrics["pred_agreement_globalTest"]),
        }

        self._append_client_csv(row, filename="Balanced")
        return row

    # ======================================================================
    # Bucle sobre todo el test
    # ======================================================================
    def explain_all_test_instances(self, config, only_idx=None):
        results = []
        self._reset_explanation_buffers()
        indices, desc_text, save_trees_flag = self._get_explain_indices(only_idx)

        for i in tqdm(indices, desc=desc_text):
            try:
                row = self._explain_one_instance(i, config, save_trees=save_trees_flag)
                results.append(row)
            except Exception as e:
                print(f"[Cliente {self.client_id}] ?????? Error en instancia {i}: {e}")

        metrics = self._compute_explanation_metrics()
        df = pd.read_csv(f"results/metrics_Balanced_cliente_{self.client_id}.csv")
        self._save_summary_metrics(df, metrics)
        return df

    def _reset_explanation_buffers(self):
        self.exp_shap_local = []
        self.exp_shap_global = []
        self.exp_lime_local = []
        self.exp_lime_global = []
        self.exp_anchor_testlocal_local = []
        self.exp_anchor_testlocal_global = []
        self.exp_anchor_testglobal_local = []
        self.exp_anchor_testglobal_global = []
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
        only_idx_mode = len(self.exp_shap_global) == 1

        def _stability_or_nan(explanations):
            n = min(len(X_test_np), len(self.y_test), len(explanations))
            if only_idx_mode and n < 2:
                return np.nan
            return ClientUtilsMixin.local_stability_score(
                X_test_np[:n], np.asarray(self.y_test)[:n], explanations[:n], k=5
            )

        # GLOBAL
        metrics["stability_shap_global"]            = _stability_or_nan(self.exp_shap_global)
        metrics["stability_lime_global"]            = _stability_or_nan(self.exp_lime_global)
        metrics["stability_anchor_testlocal_global"] = _stability_or_nan(self.exp_anchor_testlocal_global)
        metrics["stability_lore_global"]            = _stability_or_nan(self.exp_lore_global)

        metrics["ratio_has_anchor_testlocal_global"]  = float(np.mean([bool(a) for a in self.exp_anchor_testlocal_global]))
        metrics["ratio_has_anchor_testglobal_global"] = float(np.mean([bool(a) for a in self.exp_anchor_testglobal_global]))

        # LOCAL
        metrics["stability_shap_local"]            = _stability_or_nan(self.exp_shap_local)
        metrics["stability_lime_local"]            = _stability_or_nan(self.exp_lime_local)
        metrics["stability_anchor_testlocal_local"] = _stability_or_nan(self.exp_anchor_testlocal_local)
        metrics["stability_lore_local"]            = _stability_or_nan(self.exp_lore_local)

        metrics["ratio_has_anchor_testlocal_local"]  = float(np.mean([bool(a) for a in self.exp_anchor_testlocal_local]))
        metrics["ratio_has_anchor_testglobal_local"] = float(np.mean([bool(a) for a in self.exp_anchor_testglobal_local]))

        return metrics

    def _save_summary_metrics(self, df, metrics):
        mean_metrics = df.mean(numeric_only=True)
        count_metrics = df.count(numeric_only=True)

        mean_df = pd.DataFrame({"mean": mean_metrics, "count": count_metrics})

        n_total = len(df)

        mean_df.loc["ratio_has_factual_globalZ_LORE", ["mean", "count"]] = [
            df["jaccard_cov_globalZ_LORE"].notna().mean(),
            n_total,
        ]

        mean_df.loc["ratio_has_factual_localZ_LORE", ["mean", "count"]] = [
            df["jaccard_cov_localZ_LORE"].notna().mean(),
            n_total,
        ]

        # Fracción de instancias donde local y global predicen clase distinta
        mean_df.loc["ratio_model_disagreement", ["mean", "count"]] = [
            df["model_disagreement"].mean(),
            n_total,
        ]

        # Descomposición mutuamente exclusiva de causas de fallo
        monoclase     = (df["has_factual_local"] == 0) | (df["has_factual_global"] == 0)
        disagreement  = df["model_disagreement"] == 1
        mean_df.loc["ratio_fail_monoclase_only",    ["mean", "count"]] = [
            (monoclase & ~disagreement).mean(), n_total,
        ]
        mean_df.loc["ratio_fail_disagreement_only", ["mean", "count"]] = [
            (~monoclase & disagreement).mean(), n_total,
        ]
        mean_df.loc["ratio_fail_both_causes",       ["mean", "count"]] = [
            (monoclase & disagreement).mean(),  n_total,
        ]

        # Causa del vecindario monoclase: geométrico vs distorsión Non-IID
        fail_local_only  = (df["has_factual_local"] == 0) & (df["has_factual_global"] == 1)
        fail_global_only = (df["has_factual_local"] == 1) & (df["has_factual_global"] == 0)
        fail_both_mono   = (df["has_factual_local"] == 0) & (df["has_factual_global"] == 0)
        mean_df.loc["ratio_fail_noniid_local",  ["mean", "count"]] = [
            fail_local_only.mean(),  n_total,
        ]
        mean_df.loc["ratio_fail_global_only",   ["mean", "count"]] = [
            fail_global_only.mean(), n_total,
        ]
        mean_df.loc["ratio_fail_geometric",     ["mean", "count"]] = [
            fail_both_mono.mean(),   n_total,
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
        # print(
        #     f"[CLIENT_FN] construyendo cliente partition_id={partition_id} "
        #     f"(client_id={partition_id + 1}) de {num_partitions}"
        # )

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


def build_offline_client_baseline(
    dataset_name,
    class_column,
    num_clients,
    num_train_rounds,
    unique_labels,
    features,
    load_data_fn,
    set_params_fn,
    get_params_fn,
    client_id,
    label_noise_mode=None,
    label_noise_rate=0.0,
):
    partition_id = int(client_id) - 1
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

    if label_noise_mode is not None and label_noise_rate > 0:
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
    nn_model = Net(X_train.shape[1], len(unique_labels))

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
        client_id=client_id,
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
    )


def run_offline_explanations_baseline(
    dataset_name,
    class_column,
    num_clients,
    num_train_rounds,
    unique_labels,
    features,
    load_data_fn,
    set_params_fn,
    get_params_fn,
    client_ids=None,
    only_idx=None,
    global_ckpt="results/bb_global_final.pth",
    label_noise_mode=None,
    label_noise_rate=0.0,
):
    if client_ids is None:
        client_ids = list(range(1, num_clients + 1))

    if not os.path.exists(global_ckpt):
        raise FileNotFoundError(
            f"No existe el checkpoint global final: {global_ckpt}. "
            "Ejecuta primero Flower para generar results/bb_global_final.pth"
        )

    global_state = torch.load(global_ckpt, map_location="cpu")
    global_weights = [v.cpu().numpy() for v in global_state.values()]

    outputs = {}
    for client_id in client_ids:
        print(f"[OFFLINE] Ejecutando explicaciones para cliente {client_id}")
        client = build_offline_client_baseline(
            dataset_name=dataset_name,
            class_column=class_column,
            num_clients=num_clients,
            num_train_rounds=num_train_rounds,
            unique_labels=unique_labels,
            features=features,
            load_data_fn=load_data_fn,
            set_params_fn=set_params_fn,
            get_params_fn=get_params_fn,
            client_id=client_id,
            label_noise_mode=label_noise_mode,
            label_noise_rate=label_noise_rate,
        )
        _, n_eval, _ = client.evaluate(
            global_weights,
            {
                "server_round": num_train_rounds + 1,
                "explain_only": True,
                **({"only_idx": only_idx} if only_idx is not None else {}),
            },
        )
        outputs[client_id] = n_eval

    return outputs
