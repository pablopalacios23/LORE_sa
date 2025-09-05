from lore_sa.bbox import AbstractBBox
import pandas as pd
import numpy as np

from lore_sa.dataset import TabularDataset, Dataset
from lore_sa.encoder_decoder import ColumnTransformerEnc, EncDec
from lore_sa.neighgen import GeneticGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.neighgen.random import RandomGenerator
from lore_sa.surrogate import DecisionTreeSurrogate, Surrogate
from lore_sa.surrogate import EnsembleDecisionTreeSurrogate, Surrogate
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


'''Algoritmo 1: loresa(x, b, K, U)
Este está implementado en lore_sa/lore.py, en la función explain_instance.

Correspondencias:

D ← ∅
    → self.trees = []  # En EnsembleDecisionTreeSurrogate (inicializa el conjunto de árboles)

for i ∈ {1,..., N}
    → for _ in range(self.n_estimators):  # Dentro de train() del ensamble

Z(i) ← genetic(x, fitnessx=, b, K);
Z(i) ← genetic(x, fitnessx≠, b, K);
    → neighbour = self.generator.generate(z, num_instances, self.descriptor, self.encoder)

Z(i) ← Z= ∪ Z≠;
    → Se realiza dentro de generator.generate(...) al concatenar Z_eq y Z_noteq

Y(i) ← b(Z(i));
    → neighb_train_y = self.bbox.predict(neighb_train_X)

d(i) ← buildDecisionTree(Z(i), Y(i));
    → Cada árbol se entrena con: tree.fit(Z_sample, Yb_sample)

D ← D ∪ {d(i)};
    → self.trees.append(tree)  # Dentro de EnsembleDecisionTreeSurrogate

c ← mergeDecisionTrees(D);
    → merged_tree = self.surrogate.merge_trees()  # Llama a SuperTree().mergeDecisionTrees()

r = extractDecisionRule(c, x);
    → rule = merged_tree.get_rule(z, self.encoder)

Φ ← extractCounterfactuals(c, r, x, U);
    → crules, deltas = merged_tree.get_counterfactual_rules(z, neighbour, neighb_train_yb, self.encoder)

return e ← ⟨r, Φ⟩
    → return {
            'rule': rule.to_dict(),
            'counterfactuals': [c.to_dict() for c in crules],
            'merged_tree': merged_tree,
        }
'''

class Lore(object):

    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec,
                 generator: NeighborhoodGenerator, surrogate: Surrogate):
        """
        Creates a new instance of the LORE method.


        :param bbox: The black box model to be explained wrapped in a `AbstractBBox object.
        :param dataset:
        :param encoder:
        :param generator:
        :param surrogate:
        """

        super().__init__()
        self.bbox = bbox
        self.descriptor = dataset.descriptor
        self.encoder = encoder
        self.generator = generator
        self.surrogate = surrogate
        self.class_name = dataset.class_name

    def binarize_onehot_features(self, neighbour, feature_names, categorical_features):
        neighbour = neighbour.copy()
        for i, col in enumerate(feature_names):
            # Detecta si la columna es one-hot de alguna categórica
            if any(col.startswith(cat + "_") for cat in categorical_features):
                neighbour[:, i] = (neighbour[:, i] > 0.5).astype(float)
        return neighbour


    def explain(self, x: np.array, num_instances=250, merge=False, num_classes=None, feature_names=None, categorical_features=None, global_mapping=None, UNIQUE_LABELS=None, client_id=None, round_number=None):

        """
        Explains a single instance of the dataset.
        :param x: an array with the values of the instance to explain (the target class is not included)
        :return:
        """
        
        [z] = self.encoder.encode([x])

        neighbour = self.generator.generate(z, num_instances, self.descriptor, self.encoder)


        dec_neighbor = self.encoder.decode(neighbour)

        neighb_train_X = dec_neighbor[:, :]

        neighbour = self.binarize_onehot_features(neighb_train_X, feature_names, categorical_features)

        neighb_train_y = self.bbox.predict(neighb_train_X) # ETIQUETAMOS EL VECINDARIO A PARTIR DEL BLACKBOX (RED NEURONAL)

        neighb_train_yb = self.encoder.encode_target_class(neighb_train_y.reshape(-1, 1), categories_global=UNIQUE_LABELS).squeeze()


        self.surrogate.train(neighbour, neighb_train_yb, features = feature_names)

        y_surrogate = self.surrogate.predict(neighbour)

        # 1. PCA a 2D para el vecindario
        pca = PCA(n_components=num_classes, random_state=0)
        X2 = pca.fit_transform(neighbour)
        y = neighb_train_y

        # 2. Proyectar también la instancia original
        x_proj = pca.transform([x])

        # 3. Dibujar scatter
        plt.figure(figsize=(5,4))
        scatter = plt.scatter(X2[:,0], X2[:,1], c=y, cmap="coolwarm", s=20, alpha=0.7)
        plt.colorbar(scatter, label="Clase predicha por bbox")

        # 4. Marcar la instancia explicada
        plt.scatter(x_proj[0,0], x_proj[0,1], c="k", marker="*", s=120, edgecolors="w", linewidths=1.2)

        plt.title(f"Vecindario + instancia explicada para cliente {client_id}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()

        # # === Guardar en ruta deseada ===
        # # === Guardar en ruta deseada con round_number y client_id ===
        # base_dir = r"C:\Users\pablo\OneDrive - Universidad de Castilla-La Mancha\Escritorio\FLOWER_merge_trees"
        # out_dir = os.path.join(base_dir, f"Ronda_{round_number}", "LoreTree")
        # os.makedirs(out_dir, exist_ok=True)

        # fname = f"neigh_client_{client_id}.png"
        # save_path = os.path.join(out_dir, fname)

        # plt.savefig(save_path, dpi=150)
        # plt.close()
    

        # 👉 Si NO se hace merge, NO devolvemos regla ni contrafactuales
        if not merge:
            return {
                "rule": None,
                "counterfactuals": [],
                "merged_tree": None
            }

        # 👉 Si se hace merge (solo en servidor)
        merged_tree = self.surrogate.get_single_supertree(
            num_classes=num_classes,
            feature_names=feature_names,
            categorical_features=categorical_features,
            global_mapping=global_mapping
        )
        # rule = merged_tree.get_rule(z, self.encoder)
        # crules, deltas = merged_tree.get_counterfactual_rules(z, neighbour, neighb_train_yb, self.encoder)

        return {
            # 'rule': rule.to_dict(),
            # 'counterfactuals': [c.to_dict() for c in crules],
            'merged_tree': merged_tree,
            'neighborhood_Z': neighbour,
            'neighborhood_Yb': neighb_train_yb,
            'surrogate_preds': y_surrogate,
        }



class TabularRandomGeneratorLore(Lore):
    def __init__(self, bbox: AbstractBBox, dataset: TabularDataset):
        encoder = ColumnTransformerEnc(dataset.descriptor)
        generator = RandomGenerator(bbox, dataset, encoder, 0.1)
        surrogate = EnsembleDecisionTreeSurrogate(n_estimators=5)
        super().__init__(bbox, dataset, encoder, generator, surrogate)

    def explain_instance(self, x: np.array, merge=False):
        return self.explain(x.values, merge=merge)
    
    

class TabularGeneticGeneratorLore(Lore):
    def __init__(self, bbox: AbstractBBox, dataset: TabularDataset):
        encoder = ColumnTransformerEnc(dataset.descriptor)
        generator = GeneticGenerator(bbox, dataset, encoder, 0.1, random_seed=42)
        surrogate = EnsembleDecisionTreeSurrogate(n_estimators=1)
        super().__init__(bbox, dataset, encoder, generator, surrogate)

    def explain_instance(self, x: np.array, merge=False, num_classes=None, feature_names=None, categorical_features=None, global_mapping=None, UNIQUE_LABELS=None, client_id=None, round_number=None):
        return self.explain(x.values, merge=merge, num_classes=num_classes, feature_names=feature_names, categorical_features=categorical_features, global_mapping=global_mapping, UNIQUE_LABELS=UNIQUE_LABELS, client_id=client_id, round_number=round_number)
