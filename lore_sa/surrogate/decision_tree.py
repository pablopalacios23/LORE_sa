import copy
import datetime
import operator
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix

from lore_sa.encoder_decoder import EncDec, ColumnTransformerEnc
from lore_sa.logger import logger
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import sklearn.model_selection
from sklearn.experimental import enable_halving_search_cv

__all__ = ["Surrogate", "DecisionTreeSurrogate","Supertree","EnsembleDecisionTreeSurrogate"]

from lore_sa.rule import Expression, Rule
from lore_sa.surrogate.surrogate import Surrogate
from lore_sa.util import vector2dict, multilabel2str
import lore_sa

from lore_sa.rule import Rule, Expression
from lore_sa.util import vector2dict, multilabel2str
from lore_sa.logger import logger
from lore_sa.encoder_decoder import EncDec
from lore_sa.surrogate.surrogate import Surrogate

from sklearn.model_selection import HalvingGridSearchCV


import hashlib
import pickle


class DecisionTreeSurrogate(Surrogate):

    def __init__(self, kind=None, preprocessing=None):
        super().__init__(kind, preprocessing)
        self.dt = None
    
    def merge_trees(self):
        pass

    def train(self, Z, Yb, weights=None, class_values=None, multi_label: bool = False, one_vs_rest: bool = False, cv=5,
              prune_tree: bool = False):
        """

        :param Z: The training input samples
        :param Yb: The target values (class labels) as integers or strings.
        :param weights: Sample weights.
        :param class_values:
        :param [bool] multi_label:
        :param [bool] one_vs_rest:
        :param [int] cv:
        :param [bool] prune_tree:
        :return:
        """
        self.dt = DecisionTreeClassifier()            
        if prune_tree is True:
            param_list = {'min_samples_split': [0.01, 0.05, 0.1, 0.2, 3, 2],
                          'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 2, 4],
                          'splitter': ['best', 'random'],
                          'max_depth': [None, 2, 10, 12, 16, 20, 30],
                          'criterion': ['entropy', 'gini'],
                          'max_features': [0.2, 1, 5, 'auto', 'sqrt', 'log2']
                          }

            if multi_label is False or (multi_label is True and one_vs_rest is True):
                if len(class_values) == 2 or (multi_label and one_vs_rest):
                    scoring = 'precision'
                else:
                    scoring = 'precision_macro'
            else:
                scoring = 'precision_samples'

            dt_search = sklearn.model_selection.HalvingGridSearchCV(self.dt, param_grid=param_list, scoring=scoring,
                                                                    cv=cv, n_jobs=-1)
            logger.info('Search the best estimator')
            logger.info('Start time: {0}'.format(datetime.datetime.now()))
            dt_search.fit(Z, Yb, sample_weight=weights)
            logger.info('End time: {0}'.format(datetime.datetime.now()))
            self.dt = dt_search.best_estimator_
            logger.info('Pruning')
            self.prune_duplicate_leaves(self.dt)
        else:
            self.dt.fit(Z, Yb)

        return self.dt

    def is_leaf(self, inner_tree, index):
        """Check whether node is leaf node"""
        return (inner_tree.children_left[index] == TREE_LEAF and
                inner_tree.children_right[index] == TREE_LEAF)

    def prune_index(self, inner_tree, decisions, index=0):
        """
        Start pruning from the bottom - if we start from the top, we might miss
        nodes that become leaves during pruning.
        Do not use this directly - use prune_duplicate_leaves instead.
        """
        if not self.is_leaf(inner_tree, inner_tree.children_left[index]):
            self.prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not self.is_leaf(inner_tree, inner_tree.children_right[index]):
            self.prune_index(inner_tree, decisions, inner_tree.children_right[index])

        # Prune children if both children are leaves now and make the same decision:
        if (self.is_leaf(inner_tree, inner_tree.children_left[index]) and
                self.is_leaf(inner_tree, inner_tree.children_right[index]) and
                (decisions[index] == decisions[inner_tree.children_left[index]]) and
                (decisions[index] == decisions[inner_tree.children_right[index]])):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            logger.info("Pruned {}".format(index))

    def prune_duplicate_leaves(self, dt):
        """Remove leaves if both"""
        decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
        self.prune_index(dt.tree_, decisions)

    def get_rule(self, z: np.array, encoder: EncDec = None):
        """
        Extract the rules as the promises and consequences {p -> y}, starting from a Decision Tree

             {( income > 90) -> grant),
                ( job = employer) -> grant)
            }

        :param [Numpy Array] z: instance encoded of the dataset to extract the rule
        :param [EncDec] encdec:
        :return [Rule]: Rule objects
        """
        z = z.reshape(1, -1)
        feature = self.dt.tree_.feature
        threshold = self.dt.tree_.threshold
        predicted_class = self.dt.predict(z)
        inv_transform_predicted_class = encoder.decode_target_class([predicted_class])[0]

        target_feature_name = list(encoder.encoded_descriptor['target'].keys())[0]

        consequence = Expression(variable=target_feature_name, operator=operator.eq,
                                 value=inv_transform_predicted_class[0])

        leave_id = self.dt.apply(z)
        node_index = self.dt.decision_path(z).indices

        feature_names = list(encoder.encoded_features.values())
        numeric_columns = list(encoder.encoded_descriptor['numeric'].keys())

        premises = list()
        for node_id in node_index:
            if leave_id[0] == node_id:
                break
            else:
                attribute = feature_names[feature[node_id]]
                if attribute not in numeric_columns:
                    # this is a categorical feature
                    # print(f"{attribute} has value {x[0][feature[node_id]]} and threshold is {threshold[node_id]}")
                    thr = False if z[0][feature[node_id]] <= threshold[node_id] else True
                    op = operator.eq
                else:
                    thr = threshold[node_id]
                    op = operator.le if z[0][feature[node_id]] <= threshold[node_id] else operator.gt

                premises.append(Expression(attribute, op, thr))

        premises = self.compact_premises(premises)
        return Rule(premises=premises, consequences=consequence, encoder=encoder)

    def compact_premises(self, premises_list):
        """
        Remove the same premises with different values of threashold

        :param premises_list: List of Expressions that defines the premises
        :return:
        """
        attribute_list = defaultdict(list)
        for premise in premises_list:
            attribute_list[premise.variable].append(premise)

        compact_plist = list()
        for att, alist in attribute_list.items():
            if len(alist) > 1:
                min_thr = None
                max_thr = None
                for av in alist:
                    if av.operator == operator.le:
                        max_thr = min(av.value, max_thr) if max_thr else av.value
                    elif av.operator == operator.gt:
                        min_thr = max(av.value, min_thr) if min_thr else av.value

                if max_thr:
                    compact_plist.append(Expression(att, operator.le, max_thr))

                if min_thr:
                    compact_plist.append(Expression(att, operator.gt, min_thr))
            else:
                compact_plist.append(alist[0])
        return compact_plist

    def get_counterfactual_rules(self, z: np.array, neighborhood_train_X: np.array, neighborhood_train_Y: np.array,
                                 encoder: EncDec = None,
                                 filter_crules=None, constraints: dict = None, unadmittible_features: list = None):


        feature_names = list(encoder.encoded_features.values())
        # numeric_columns = list(encoder.encoded_descriptor['numeric'].keys())
        predicted_class = self.dt.predict(z.reshape(1, -1))[0]
        # inv_transform_predicted_class = encoder.encoder.named_transformers_.get('target')\
        #     .inverse_transform([predicted_class])[0] #TODO: modify to generalize to multiclasses

        class_name = list(encoder.encoded_descriptor['target'].keys())[0]
        class_values = list(encoder.encoded_descriptor['target'].values())[0]['distinct_values']


        clen = np.inf
        crule_list = list()
        delta_list = list()  # Se inicializa el conjunto de contrafactuales y el mínimo número de condiciones no llevan a cabo la regla.

        # y = self.dt.predict(neighborhood_dataset.df)[0]
        # Y = self.dt.predict(neighborhood_dataset.df)

        x_dict = vector2dict(z, feature_names)
        # select the subset of  neighborhood_train_X that have a classification different from the input x
        Z1 = neighborhood_train_X[np.where(neighborhood_train_Y != predicted_class)] # Se filtran las instancias en el vecindario que tienen una clase diferente a la de x.

        # We search for the shortest rule among those that support the elements in Z1
        for zi in Z1: # Se recorre cada instancia candidata a ser contrafactual.
            #
            crule = self.get_rule(z=zi, encoder=encoder)

            delta = self.get_falsified_conditions(x_dict, crule) # Cuenta cuántas condiciones de la regla q no cumple x.
            num_falsified_conditions = len(delta)

            if unadmittible_features is not None:
                is_feasible = self.check_feasibility_of_falsified_conditions(delta, unadmittible_features)
                if is_feasible is False:
                    continue

            if constraints is not None:
                ##TODO
                to_remove = list()
                for p in crule.premises:
                    if p.variable in constraints.keys():
                        if p.operator == constraints[p.variable]['op']:
                            if p.thr > constraints[p.variable]['thr']:
                                break
                                # caso corretto
                ##TODO

            if filter_crules is not None:
                xc = self.apply_counterfactual(z, delta, feature_names)
                bb_outcomec = filter_crules(xc.reshape(1, -1))[0]
                bb_outcomec = class_values[bb_outcomec] if isinstance(class_name, str) else multilabel2str(bb_outcomec,
                                                                                                           class_values)
                dt_outcomec = crule.cons

                if bb_outcomec == dt_outcomec:
                    if num_falsified_conditions < clen: # Se actualiza si la regla tiene menos condiciones violadas.
                        clen = num_falsified_conditions
                        crule_list = [crule]
                        delta_list = [delta]
                    elif num_falsified_conditions == clen: # Se añaden reglas igualmente buenas si no están repetidas.
                        if delta not in delta_list:
                            crule_list.append(crule)
                            delta_list.append(delta)
            else:
                if num_falsified_conditions < clen: # Se actualiza si la regla tiene menos condiciones violadas.
                    clen = num_falsified_conditions
                    crule_list = [crule]
                    delta_list = [delta]
                    # print(crule, delta)
                elif num_falsified_conditions == clen: # Se añaden reglas igualmente buenas si no están repetidas.
                    if delta not in delta_list:
                        crule_list.append(crule)
                        delta_list.append(delta)

        return crule_list, delta_list # Devuelve las reglas contrafactuales y las condiciones que x tendría que cambiar.

    def get_falsified_conditions(self, x_dict: dict, crule: Rule):
        """
        Check the wrong conditions
        :param x_dict:
        :param crule:
        :return: list of falsified premises
        """
        delta = []
        for p in crule.premises:
            try:
                if p.operator == operator.le and x_dict[p.variable] > p.value:
                    delta.append(p)
                elif p.operator == operator.gt and x_dict[p.variable] <= p.value:
                    delta.append(p)
            except:
                # print('pop', p.operator2string(), 'xd', x_dict, 'xd di p ', p.variable, 'hthrr', p.value)

                continue
        return delta

    def check_feasibility_of_falsified_conditions(self, delta, unadmittible_features: list):
        """
        Check if a falsifield confition is in an unadmittible feature list
        :param delta:
        :param unadmittible_features:
        :return: True or False
        """
        for p in delta:
            if p.variable in unadmittible_features:
                if unadmittible_features[p.variable] is None:
                    return False
                else:
                    if unadmittible_features[p.variable] == p.operator:
                        return False
        return True

    def apply_counterfactual(self, x, delta, feature_names:list, features_map=None, features_map_inv=None, numeric_columns=None):
        x_dict = vector2dict(x, feature_names)
        x_copy_dict = copy.deepcopy(x_dict)
        for p in delta:
            if p.variable in numeric_columns:
                if p.value == int(p.value):
                    gap = 1.0
                else:
                    decimals = list(str(p.value).split('.')[1])
                    for idx, e in enumerate(decimals):
                        if e != '0':
                            break
                    gap = 1 / (10 ** (idx + 1))
                if p.operator == operator.gt:
                    x_copy_dict[p.variable] = p.value + gap
                else:
                    x_copy_dict[p.variable] = p.value
            else:
                fn = p.variable
                if p.operator == operator.gt:
                    if features_map is not None:
                        fi = list(feature_names).index(p.att)
                        fi = features_map_inv[fi]
                        for fv in features_map[fi]:
                            x_copy_dict['%s=%s' % (fn, fv)] = 0.0
                    x_copy_dict[p.att] = 1.0

                else:
                    if features_map is not None:
                        fi = list(feature_names).index(p.att)
                        fi = features_map_inv[fi]
                        for fv in features_map[fi]:
                            x_copy_dict['%s=%s' % (fn, fv)] = 1.0
                    x_copy_dict[p.att] = 0.0

        x_counterfactual = np.zeros(len(x_dict))
        for i, fn in enumerate(feature_names):
            x_counterfactual[i] = x_copy_dict[fn]

        return x_counterfactual
    

'''Algoritmo 3: extractCounterfactuals(c, r, x, U)
Implementado principalmente en:
- lore_sa/surrogate/decision_tree.py → get_counterfactual_rules(...)
- lore_sa/surrogate/decision_tree.py → get_falsified_conditions(...)
- lore_sa/surrogate/decision_tree.py → check_feasibility_of_falsified_conditions(...)
- lore_sa/surrogate/decision_tree.py → apply_counterfactual(...)
- lore_sa/surrogate/decision_tree.py → compact_premises(...)

Correspondencias:

Q ← getPathsWithDifferentLabel(c, y);
    → Z1 = neighborhood_train_X[np.where(neighborhood_train_Y != predicted_class)]

Φ ← ∅; min ← +∞;
    → crule_list = []; clen = np.inf

for q ∈ Q do
    → for zi in Z1:

if not q → U|q then
    → if unadmittible_features is not None:
          is_feasible = self.check_feasibility_of_falsified_conditions(...)
          if is_feasible is False:
              continue

qlen ← nf(q, x) = |{sc ∈ q | ¬sc(x)}|
    → delta = self.get_falsified_conditions(...)
      num_falsified_conditions = len(delta)

if qlen < min then
    → if num_falsified_conditions < clen:

Φ ← {q → y′}; min ← qlen
    → crule_list = [crule]; delta_list = [delta]; clen = num_falsified_conditions

else if qlen == min then
    → elif num_falsified_conditions == clen:

Φ ← Φ ∪ {q → y′}
    → crule_list.append(crule); delta_list.append(delta)

return Φ
    → return crule_list, delta_list
'''



    
class EnsembleDecisionTreeSurrogate(Surrogate):
    def __init__(self, n_estimators=5):
        super().__init__()
        self.n_estimators = n_estimators
        self.trees = []  # D ← ∅

    def train(self, Z, Yb):
        self.trees.clear()
        self.Z = Z              # 👈 guarda vecinos codificados
        self.Y = Yb             # 👈 guarda etiquetas codificadas
        for _ in range(self.n_estimators):
            Z_sample, Yb_sample = resample(Z, Yb)

            tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
            tree.fit(Z_sample, Yb_sample)
            self.trees.append(tree)

    def get_rule(self, z, encoder):
        dt_surrogate = DecisionTreeSurrogate()
        dt_surrogate.dt = self.trees[0]
        return dt_surrogate.get_rule(z, encoder)

    def get_counterfactual_rules(self, z, Z, Yb, encoder, **kwargs):
        dt_surrogate = DecisionTreeSurrogate()
        dt_surrogate.dt = self.trees[0]
        return dt_surrogate.get_counterfactual_rules(z, Z, Yb, encoder, **kwargs)
    
    def get_single_supertree(self, num_classes, feature_names, categorical_features, global_mapping):
        # Solo tienes un árbol entrenado
        tree = self.trees[0]
        supertree = SuperTree()
        root = supertree.rec_buildTree(tree, list(range(tree.n_features_in_)), num_classes)
        supertree.root = root

        return supertree


    def merge_trees(self, num_classes=None, feature_names=None, categorical_features=None, global_mapping=None):
        supertree = SuperTree()
        num_classes = len(np.unique(self.Y))  # <- Asegúrate de tener acceso a Y (etiquetas)
        roots = [supertree.rec_buildTree(tree, list(range(tree.n_features_in_)), num_classes) for tree in self.trees]
        supertree.mergeDecisionTrees(roots, num_classes=num_classes,feature_names=feature_names,categorical_features=categorical_features,global_mapping=global_mapping)        # print("✅ merge_trees() fue llamado")
        supertree.prune_redundant_leaves_full()  # ✅ método mejorado
        supertree.merge_equal_class_leaves()  # 👈 Añade esta línea
        return supertree


class SuperTree(Surrogate):
    def __init__(self, kind=None, preprocessing=None):
        super(SuperTree, self).__init__(kind, preprocessing)
        self.root = None

    def train(self, Z, Yb, **kwargs):
        pass

    def merge_trees(self):
        return self.root
    
    @staticmethod
    def find_leaf(node, z):
        while not node.is_leaf:
            val = z[node.feat]
            for i, threshold in enumerate(node.intervals):
                if val <= threshold:
                    node = node.children[i]
                    break
            else:
                node = node.children[-1]
        return node

        
    @classmethod
    def convert_SuperNode_to_Node(cls, super_node):
        if super_node is None:
            return None

        if super_node.is_leaf:
            return cls.Node(labels=super_node.labels, is_leaf=True)

        if len(super_node.children) != 2:
            raise ValueError("Solo se puede convertir SuperNodes binarios a Nodes")

        left = cls.convert_SuperNode_to_Node(super_node.children[0])
        right = cls.convert_SuperNode_to_Node(super_node.children[1])

        node = cls.Node(
            feat_num=super_node.feat,
            thresh=super_node.intervals[0],
            left_child=left,
            right_child=right
        )

        return node

    @staticmethod
    def convert_Node_to_SuperNode(node, level=0):
        if node is None:
            return None

        if node.is_leaf:
            return SuperTree.SuperNode(
                is_leaf=True,
                labels=node.labels,
                level=level
            )

        # Si es binario (Node)
        if hasattr(node, "_left_child") or hasattr(node, "_right_child"):
            left = SuperTree.convert_Node_to_SuperNode(node._left_child, level + 1)
            right = SuperTree.convert_Node_to_SuperNode(node._right_child, level + 1)
            return SuperTree.SuperNode(
                feat_num=node.feat,
                intervals=[node.thresh],
                children=[left, right],
                labels=node.labels,
                is_leaf=False,
                level=level
            )

        # Si ya tiene hijos multiclase
        children = [SuperTree.convert_Node_to_SuperNode(c, level + 1) for c in node.children]
        return SuperTree.SuperNode(
            feat_num=node.feat,
            intervals=node.intervals,
            children=children,
            labels=node.labels,
            is_leaf=False,
            level=level
        )
    
    def prune_redundant_leaves_full(self): # eliminar nodos redundantes (cuando todos los hijos de un nodo predicen lo mismo)
        # print("🔧 Iniciando poda completa de SuperTree")

        def prune(node):
            if node.is_leaf or not node.children:
                return node

            node.children = [prune(child) for child in node.children]

            # print(f"👀 Evaluando poda en nodo nivel {node.level}")
            # for idx, child in enumerate(node.children):
            #     print(f"   └─ Hijo {idx}: predicción={np.argmax(child.labels)}, labels={child.labels}")

            if all(child.is_leaf for child in node.children):
                predictions = [np.argmax(child.labels) for child in node.children]
                if all(p == predictions[0] for p in predictions):
                    # print(f"✅ 🌿 Poda realizada en nivel {node.level}: clase común = {predictions[0]}")
                    combined = np.sum([child.labels for child in node.children], axis=0)
                    return self.SuperNode(is_leaf=True, labels=combined, level=node.level)

            return node

        if self.root:
            self.root = prune(self.root)
        # print("✅ Poda completa finalizada")

    def merge_equal_class_leaves(self):
        def merge_adjacent(node):
            if node.is_leaf:
                return node

            # Si es binario y ambos hijos son hojas con misma clase → fusión total
            if hasattr(node, "children") and node.children and len(node.children) == 2:
                left, right = node.children
                if left.is_leaf and right.is_leaf:
                    left_class = np.argmax(left.labels)
                    right_class = np.argmax(right.labels)
                    if left_class == right_class:
                        # Fusionar todo el nodo por redundancia
                        merged = self._fuse_group([left, right], node.level)
                        return merged

            # Recursión normal
            if hasattr(node, "children"):
                node.children = [merge_adjacent(child) for child in node.children]

                if not node.intervals or len(node.intervals) < 1 or len(node.children) <= 2:
                    return node  # Evitar errores

                new_children = []
                new_intervals = []
                current_group = []
                current_interval = []

                for i, child in enumerate(node.children):
                    if i >= len(node.intervals):
                        break

                    if child.is_leaf:
                        if not current_group:
                            current_group = [child]
                            current_interval = [
                                node.intervals[i - 1] if i > 0 else -float("inf"),
                                node.intervals[i]
                            ]
                        else:
                            prev_class = np.argmax(current_group[0].labels)
                            curr_class = np.argmax(child.labels)
                            if curr_class == prev_class:
                                current_group.append(child)
                                current_interval[1] = node.intervals[i]
                            else:
                                merged = self._fuse_group(current_group, node.level + 1)
                                new_children.append(merged)
                                new_intervals.append(current_interval[1])
                                current_group = [child]
                                current_interval = [
                                    node.intervals[i - 1] if i > 0 else -float("inf"),
                                    node.intervals[i]
                                ]
                    else:
                        if current_group:
                            merged = self._fuse_group(current_group, node.level + 1)
                            new_children.append(merged)
                            new_intervals.append(current_interval[1])
                            current_group = []
                        new_children.append(child)
                        new_intervals.append(node.intervals[i])

                if current_group:
                    merged = self._fuse_group(current_group, node.level + 1)
                    new_children.append(merged)
                    new_intervals.append(current_interval[1])

                node.children = new_children
                node.intervals = new_intervals

            return node

        if self.root:
            self.root = merge_adjacent(self.root)


    def print_tree(self, node=None, prefix=""):
        if node is None:
            node = self.root

        if node.is_leaf:
            print(f"{prefix}→ Clase: {node.labels}")
            return

        for i, child in enumerate(node.children):
            if i == 0:
                cond = f"≠ {node.intervals[i]}" if len(node.intervals) > 1 else f"≤ {node.intervals[i]}"
            elif i == len(node.children) - 1:
                cond = f"> {node.intervals[i-1]}"
            else:
                cond = f"> {node.intervals[i-1]} ∧ ≤ {node.intervals[i]}"
            print(f"{prefix}{node.feat} {cond}")
            self.print_tree(child, prefix + "   ")


    def _fuse_group(self, group, level):
        total = np.sum([g.labels for g in group], axis=0)
        return self.SuperNode(is_leaf=True, labels=total, level=level)

    def get_rule(self, z, encoder):
        feature_names = list(encoder.encoded_features.values())
        numeric_columns = list(encoder.encoded_descriptor['numeric'].keys())
        target_feature_name = list(encoder.encoded_descriptor['target'].keys())[0]

        premises = []

        def traverse(node, x):
            if node.is_leaf:
                predicted_class = np.argmax(node.labels)
                consequence = Expression(
                    variable=target_feature_name,
                    operator=operator.eq,
                    value=encoder.decode_target_class([[predicted_class]])[0][0]
                )
                return premises, consequence

            val = x[node.feat]
            feat_name = feature_names[node.feat]
            is_numeric = feat_name in numeric_columns
            is_categorical = "=" in feat_name
            base_feat = feat_name.split("=")[0] if is_categorical else feat_name
            cat_value = feat_name.split("=")[1] if is_categorical else None

            for i, thr in enumerate(node.intervals):
                if val <= thr:
                    if is_numeric:
                        if i == 0:
                            premises.append(Expression(base_feat, operator.le, thr))
                        else:
                            premises.append(Expression(base_feat, operator.gt, node.intervals[i - 1]))
                    elif is_categorical:
                        premises.append(Expression(base_feat, operator.ne, cat_value))
                    else:
                        premises.append(Expression(base_feat, operator.eq, val))
                    return traverse(node.children[i], x)

            # Último intervalo
            last_thr = node.intervals[-1]
            if is_numeric:
                premises.append(Expression(base_feat, operator.gt, last_thr))
            elif is_categorical:
                premises.append(Expression(base_feat, operator.eq, cat_value))
            else:
                premises.append(Expression(base_feat, operator.eq, val))

            return traverse(node.children[-1], x)

        prem, cons = traverse(self.root, z)
        compacted = DecisionTreeSurrogate().compact_premises(prem)
        return Rule(premises=compacted, consequences=cons, encoder=encoder)
        
    




    def get_counterfactual_rules(self, z, neighborhood_train_X, neighborhood_train_Y, encoder, **kwargs):
        dt_dummy = DecisionTreeSurrogate()
        dt_dummy.compact_premises = DecisionTreeSurrogate().compact_premises
        rules = []

        pred_z = self.root.predict([z])[0]
        for xi, yi in zip(neighborhood_train_X, neighborhood_train_Y):
            if yi != pred_z:
                rule = self.get_rule(xi, encoder)
                delta = dt_dummy.get_falsified_conditions(vector2dict(z, list(encoder.encoded_features.values())), rule)
                rules.append((rule, delta))

        if not rules:
            return [], []

        # Eliminar reglas duplicadas
        unique = set()
        filtered_rules = []
        for r, d in rules:
            key = str(r)
            if key not in unique:
                unique.add(key)
                filtered_rules.append((r, d))

        min_len = min(len(delta) for _, delta in filtered_rules)
        best = [(r, d) for r, d in filtered_rules if len(d) == min_len]

        crules, deltas = zip(*best)
        return list(crules), list(deltas)
    
    

    def get_counterfactual_rules_merged(self, z, encoder, **kwargs):
        from lore_sa.util import vector2dict
        from lore_sa.surrogate.decision_tree import DecisionTreeSurrogate

        dt_dummy = DecisionTreeSurrogate()
        dt_dummy.compact_premises = DecisionTreeSurrogate().compact_premises

        z_dict = vector2dict(z, list(encoder.encoded_features.values()))
        pred_class = self.root.predict([z])[0]

        # Recorremos todas las reglas del árbol
        all_rules = self.root.get_all_rules(encoder)

        rules_by_class = {}
        for rule in all_rules:
            target_class = rule.consequences.value

            # Solo nos interesan clases distintas a la actual
            if target_class == pred_class:
                continue

            # Calculamos condiciones que z no cumple (para saber qué cambiar)
            delta = dt_dummy.get_falsified_conditions(z_dict, rule)

            # Nos quedamos con la de menor cambio por clase
            if target_class not in rules_by_class or len(delta) < len(rules_by_class[target_class][1]):
                rules_by_class[target_class] = (rule, delta)

        if not rules_by_class:
            return [], []

        crules, deltas = zip(*rules_by_class.values())
        return list(crules), list(deltas)
    
    def __str__(self):
        return self._to_str(self.root)

    def _to_str(self, node, depth=0):
        indent = "  " * depth
        if node.is_leaf:
            return f"{indent}Leaf → pred: {node.labels}\n"

        result = f"{indent}Feature[{node.feat}] split:\n"
        for i, child in enumerate(node.children):
            if i == 0:
                val = node.intervals[i]
                cond = f"<= {val:.2f}" if val is not None else "<= None"
            elif i == len(node.children) - 1:
                val = node.intervals[i - 1]
                cond = f"> {val:.2f}" if val is not None else "> None"
            else:
                val1 = node.intervals[i - 1]
                val2 = node.intervals[i]
                cond = f"({val1:.2f}, {val2:.2f}]" if None not in (val1, val2) else "(?, ?]"
            result += f"{indent}  ├─ If {cond}:\n"
            result += self._to_str(child, depth + 2)
        return result

    class Node:
        def __init__(self, feat_num=None, weights=None, thresh=None, labels=None, is_leaf=False, impurity=1, categorical_val=None, **kwargs):
            self.feat = feat_num
            self.thresh = thresh
            self.is_leaf = is_leaf
            self._weights = weights
            self._left_child = kwargs.get('left_child', None)
            self._right_child = kwargs.get('right_child', None)
            self.children = kwargs.get('children', None)
            self.impurity = impurity
            self.labels = labels
            self.categorical_val = categorical_val  # 👈 añade esta línea
            if weights is not None:
                self._features_involved = np.arange(weights.shape[0] - 1)
            else:
                if not self.children:
                    self.children = []
                    if self._left_child:
                        self.children.append(self._left_child)
                    if self._right_child:
                        self.children.append(self._right_child)

        def predict(self, X):
            def predict_datum(node, x):
                if node.is_leaf:
                    return np.argmax(node.labels)
                else:
                    if node.feat is not None:
                        Xf = node.feat
                        if node.thresh <= x[Xf] and node._left_child:
                            next_node = node._left_child
                        elif node._right_child:
                            next_node = node._right_child
                        else:
                            return np.argmax(node.labels)
                    else:
                        next_node = node._left_child
                    return predict_datum(next_node, x)
            return np.array([predict_datum(self, el) for el in X])
        
        def to_dict(self):
            node_dict = {
                "is_leaf": self.is_leaf,
                "labels": self.labels.tolist() if self.labels is not None else None,
                "feat": self.feat,
                "thresh": self.thresh,
            }
            if self._left_child or self._right_child:
                node_dict["left"] = self._left_child.to_dict() if self._left_child else None
                node_dict["right"] = self._right_child.to_dict() if self._right_child else None
            elif self.children:
                node_dict["children"] = [child.to_dict() for child in self.children]
                node_dict["intervals"] = self.intervals.tolist()
            return node_dict
        
        def __repr__(self):
            return str(self.to_dict())
        
        def __str__(self):
            return str(self.to_dict())
        
        @staticmethod
        def from_dict(d):
            node = SuperTree.Node(
                feat_num=d.get("feat"),
                thresh=d.get("thresh"),
                labels=np.array(d.get("labels")) if d.get("labels") else None,
                is_leaf=d.get("is_leaf", False)
            )
            if "left" in d or "right" in d:
                node._left_child = SuperTree.Node.from_dict(d.get("left")) if d.get("left") else None
                node._right_child = SuperTree.Node.from_dict(d.get("right")) if d.get("right") else None
                node.children = []
                if node._left_child:
                    node.children.append(node._left_child)
                if node._right_child:
                    node.children.append(node._right_child)
            elif "children" in d:
                node.children = [SuperTree.Node.from_dict(c) for c in d.get("children")]
                node.intervals = d.get("intervals", [])
            return node

    def rec_buildTree(self, dt: DecisionTreeClassifier, feature_used, num_classes):
        nodes = dt.tree_.__getstate__()['nodes']
        values = dt.tree_.__getstate__()['values']

        def createNode(idx):
            line = nodes[idx]
            raw_pred = values[idx][0]
            full_pred = np.zeros(num_classes)
            copy_len = min(len(raw_pred), num_classes)
            full_pred[:copy_len] = raw_pred[:copy_len]

            if line[0] == -1:  # hoja
                return self.Node(feat_num=None, thresh=None, labels=full_pred, is_leaf=True)
            LC = createNode(line[0])
            RC = createNode(line[1])
            return self.Node(
                feat_num=feature_used[line[2]],
                thresh=line[3],
                labels=full_pred,
                is_leaf=False,
                left_child=LC,
                right_child=RC
            )

        return createNode(0)
    

    

    def mergeDecisionTrees(self, roots, num_classes, level=0, feature_names=None,
                      categorical_features=None, global_mapping=None, used_feats=None,
                      remaining_categorical_vals=None, excluded_categorical_vars=None):

        indent = "  " * level
        roots = [r for r in roots if r is not None]

        if used_feats is None:
            used_feats = set()
        if excluded_categorical_vars is None:
            excluded_categorical_vars = set()

        if not roots:
            return None

        if all(r.is_leaf for r in roots):
            votes = [np.argmax(r.labels) for r in roots if r.labels is not None]
            val, cou = np.unique(votes, return_counts=True)
            labels = np.zeros(num_classes)
            for v, c in zip(val, cou):
                labels[v] = c
            node = self.SuperNode(is_leaf=True, labels=labels, level=level)
            if level == 0:
                self.root = node
            return node

        val, cou = np.unique([r.feat for r in roots if r.feat is not None], return_counts=True)
        feat_counts = [(f, cou[i]) for i, f in enumerate(val)
                    if f not in used_feats and feature_names[f] not in excluded_categorical_vars]

        if not feat_counts:
            labels = np.zeros(num_classes)
            for r in roots:
                if r.labels is not None:
                    labels += r.labels
            node = self.SuperNode(is_leaf=True, labels=labels, level=level)
            if level == 0:
                self.root = node
            return node

        Xf = max(feat_counts, key=lambda x: x[1])[0]
        fname = feature_names[Xf]
        is_categorical = categorical_features and fname in categorical_features

        if is_categorical:
            # SOLO LOS VALORES PRESENTES EN LOS ÁRBOLES
            used_values = set()
            for r in roots:
                if r.feat == Xf and r.thresh is not None:
                    idx = int(r.thresh)
                    if idx < len(global_mapping[fname]):
                        used_values.add(idx)
            remaining_categorical_vals = sorted(used_values)
            if not remaining_categorical_vals:
                # Si no quedan, hoja
                labels = np.zeros(num_classes)
                for r in roots:
                    if r.labels is not None:
                        labels += r.labels
                node = self.SuperNode(is_leaf=True, labels=labels, level=level)
                if level == 0:
                    self.root = node
                return node

            split_val_idx = remaining_categorical_vals[0]
            remaining_vals = [v for v in remaining_categorical_vals if v != split_val_idx]

            branches = self.computeBranchCategorical(
                node=roots,
                feat_num=Xf,
                global_values=[split_val_idx],
                excluded_values=set()
            )

            left_roots = [l for l, _ in branches]
            right_roots = [r for _, r in branches]

            # print(f"Antes de mergeDecisionTrees (profundidad {level}):")
            # print(self._to_str(self.SuperNode.from_dict(roots[0].to_dict())))

            # IZQUIERDA: valor fijado, ya no se puede volver a dividir por esta variable
            left = self.mergeDecisionTrees(
                roots=left_roots,
                num_classes=num_classes,
                level=level + 1,
                feature_names=feature_names,
                categorical_features=categorical_features,
                global_mapping=global_mapping,
                used_feats=used_feats.copy(),
                remaining_categorical_vals=None,
                excluded_categorical_vars=excluded_categorical_vars | {fname}
            )
            # print(f"Izquierda (profundidad {level}):")
            # print(self._to_str(self.SuperNode.from_dict(roots[0].to_dict())))

            # DERECHA: quedan valores por explorar, puedes seguir dividiendo por la variable
            right = self.mergeDecisionTrees(
                roots=right_roots,
                num_classes=num_classes,
                level=level + 1,
                feature_names=feature_names,
                categorical_features=categorical_features,
                global_mapping=global_mapping,
                used_feats=used_feats.copy(),
                remaining_categorical_vals=remaining_vals,
                excluded_categorical_vars=excluded_categorical_vars
            )
            # print(f"Derecha (profundidad {level}):")
            # print(self._to_str(self.SuperNode.from_dict(roots[0].to_dict())))

            node = self.SuperNode(
                is_leaf=False,
                labels=np.mean([r.labels for r in roots if r.labels is not None], axis=0),
                feat_num=Xf,
                thresh=split_val_idx,
                intervals=[split_val_idx],
                children=[left, right],
                level=level
            )
            if level == 0:
                self.root = node
            return node

        else:
            thresholds = [r.thresh for r in roots if r.feat == Xf and r.thresh is not None]
            if not thresholds:
                labels = np.zeros(num_classes)
                for r in roots:
                    if r.labels is not None:
                        labels += r.labels
                node = self.SuperNode(is_leaf=True, labels=labels, level=level)
                if level == 0:
                    self.root = node
                return node

            split_val = np.median(thresholds)
            branches = self.computeBranch(roots, split_val, Xf, verbose=False)

            left_roots = [l for l, _ in branches]
            right_roots = [r for _, r in branches]

            left = self.mergeDecisionTrees(
                roots=left_roots,
                num_classes=num_classes,
                level=level + 1,
                feature_names=feature_names,
                categorical_features=categorical_features,
                global_mapping=global_mapping,
                used_feats=used_feats.copy(),
                excluded_categorical_vars=excluded_categorical_vars.copy()
            )

            right = self.mergeDecisionTrees(
                roots=right_roots,
                num_classes=num_classes,
                level=level + 1,
                feature_names=feature_names,
                categorical_features=categorical_features,
                global_mapping=global_mapping,
                used_feats=used_feats.copy(),
                excluded_categorical_vars=excluded_categorical_vars.copy()
            )

            node = self.SuperNode(
                is_leaf=False,
                labels=np.mean([r.labels for r in roots if r.labels is not None], axis=0),
                feat_num=Xf,
                thresh=split_val,
                intervals=[split_val],
                children=[left, right],
                level=level
            )
            if level == 0:
                self.root = node
            return node
            

    def computeBranchCategorical(self, node, feat_num, global_values, excluded_values=None):
        """
        Divide cada árbol en ramas izquierda/derecha según si feat == global_values[0].
        Si no se puede dividir directamente, se intenta propagar recursivamente.
        """
        if excluded_values is None:
            excluded_values = set()

        split_val = global_values[0]
        branches = []

        for n in node:
            if n is None:
                branches.append((None, None))

            elif n.is_leaf:
                # Se copia como está
                branches.append((
                    self.Node(labels=n.labels, is_leaf=True),
                    self.Node(labels=n.labels, is_leaf=True)
                ))

            elif n.feat == feat_num and n.thresh == split_val:
                # División directa
                branches.append((n._left_child, n._right_child))

            else:
                # Recurre a los hijos y reconstruye el árbol
                left_child = self.computeBranchCategorical(
                    [n._left_child], feat_num, global_values, excluded_values=excluded_values
                )[0][0]

                right_child = self.computeBranchCategorical(
                    [n._right_child], feat_num, global_values, excluded_values=excluded_values
                )[0][1]

                branches.append((
                    self.Node(
                        is_leaf=n.is_leaf,
                        feat_num=n.feat,
                        thresh=n.thresh,
                        labels=n.labels,
                        left_child=left_child,
                        right_child=right_child
                    ),
                    self.Node(
                        is_leaf=n.is_leaf,
                        feat_num=n.feat,
                        thresh=n.thresh,
                        labels=n.labels,
                        left_child=left_child,
                        right_child=right_child
                    )
                ))

        return branches




    def computeBranch(self, nodes, split_val, feat_idx, verbose=False):
        branches = []
        for i, n in enumerate(nodes):
            if n is None:
                branches.append((None, None))
            elif n.is_leaf:
                branches.append((
                    self.Node(labels=n.labels, is_leaf=True),
                    self.Node(labels=n.labels, is_leaf=True)
                ))
            elif n.feat == feat_idx:
                # if verbose:
                #     print(f"   Nodo {i+1}: división directa en feat {feat_idx} con threshold {n.thresh}")
                branches.append((n._left_child, n._right_child))
            else:
                left = self.computeBranch([n._left_child], split_val, feat_idx)[0][0]
                right = self.computeBranch([n._right_child], split_val, feat_idx)[0][1]
                branches.append((
                    self.Node(feat_num=n.feat, thresh=n.thresh, left_child=left, right_child=right),
                    self.Node(feat_num=n.feat, thresh=n.thresh, left_child=left, right_child=right)
                ))
        return branches







    class SuperNode:
        def __init__(self, feat_num=None, thresh=None, intervals=None, categorical_val=None, children=None, labels=None, is_leaf=False, level=0):
            self.feat = feat_num
            self.thresh = thresh
            self.intervals = intervals
            self.categorical_val = categorical_val
            self.children = children or []
            self.labels = labels
            self.is_leaf = is_leaf
            self.level = level

        def print_superTree(self, level=0):
            if self.is_leaf:
                print("|\t" * level + f"|--- class: {np.argmax(self.labels)} {self.labels}")
            else:
                print("|\t" * level + f"|--- X_{self.feat}")
                for i, child in enumerate(self.children):
                    print("|\t" * (level + 1) + f"[<= {self.intervals[i]}]")
                    child.print_superTree(level + 2)

        def predict(self, X):
            def predict_datum(node, x):
                if node.is_leaf:
                    return np.argmax(node.labels)
                else:
                    val = x[node.feat]
                    for i, thr in enumerate(node.intervals):
                        if val <= thr:
                            return predict_datum(node.children[i], x)
                    return np.argmax(node.labels)
            return np.array([predict_datum(self, el) for el in X])
        
        def to_dict(self):
            node_dict = {
                "is_leaf": self.is_leaf,
                "feat": int(self.feat) if self.feat is not None else None,
                "labels": [int(x) for x in self.labels.tolist()] if isinstance(self.labels, np.ndarray) else self.labels,
                "intervals": self.intervals if hasattr(self, "intervals") else [],
            }

            if self.is_leaf:
                node_dict["children"] = []
            else:
                node_dict["children"] = [child.to_dict() for child in self.children]

            return node_dict
        
        def get_all_rules(self, encoder, feature_path=None):
            if feature_path is None:
                feature_path = []

            from lore_sa.rule import Expression, Rule
            from lore_sa.surrogate.decision_tree import DecisionTreeSurrogate
            import operator

            rules = []
            feature_names = list(encoder.encoded_features.values())
            numeric_columns = list(encoder.encoded_descriptor["numeric"].keys())
            target_feature_name = list(encoder.encoded_descriptor["target"].keys())[0]

            def traverse(node, path):
                if node.is_leaf:
                    pred_class = np.argmax(node.labels)
                    consequence = Expression(
                        variable=target_feature_name,
                        operator=operator.eq,
                        value=encoder.decode_target_class([[pred_class]])[0][0]
                    )
                    compacted = DecisionTreeSurrogate().compact_premises(path)
                    rules.append(Rule(premises=compacted, consequences=consequence, encoder=encoder))
                    return

                for i, child in enumerate(node.children):
                    local_path = path.copy()
                    feat_name = feature_names[node.feat]

                    if feat_name in numeric_columns:
                        if i == 0:
                            local_path.append(Expression(feat_name, operator.le, node.intervals[i]))
                        elif i == len(node.children) - 1:
                            local_path.append(Expression(feat_name, operator.gt, node.intervals[i - 1]))
                        else:
                            local_path.append(Expression(feat_name, operator.gt, node.intervals[i - 1]))
                            local_path.append(Expression(feat_name, operator.le, node.intervals[i]))

                    traverse(child, local_path)

            traverse(self, feature_path)
            return rules

        
        @classmethod
        def from_dict(cls, data):
            return cls(
                feat_num=data.get("feat"),
                intervals=data.get("intervals", []),
                labels=np.array(data.get("labels", [])),
                children=[cls.from_dict(child) for child in data.get("children", [])],
                is_leaf=data.get("is_leaf", False),
                level=data.get("level", 0)
            )



    
    
'''

---------------------------------------------------

Árbol 1

X_1 < 5.0
    ├── X_2 < 3.0 → Clase A
    └── X_2 ≥ 3.0 → Clase B

----------------------------------------------------

Árbol 2

X_1 < 4.0
    ├── X_2 < 2.5 → Clase A
    └── X_2 ≥ 2.5 → Clase C

----------------------------------------------------

Árbol 3

X_3 < 3.5
    ├── X_2 < 2.0 → Clase A
    └── X_2 ≥ 2.0 → Clase C

----------------------------------------------------

PASO 2: ¿Son todos los nodos hojas?

No. Tenemos nodos internos, así que no fusionamos directamente en una hoja.

---------------------------------------------------

PASO 3: ¿Cuál es la variable (X_i) más usada?

Árbol 1 → X_1
Árbol 2 → X_1
Árbol 3 → X_3

X_1 es la más usada (2 veces).


---------------------------------------------------

PASO 4: Obtener los umbrales de X_1

Árbol 1 → X_1 < 5.0
Árbol 2 → X_1 < 4.0

Se generan intervalos:
(-∞, 4.0] , (4.0, 5.0] , (5.0, ∞)

Estos serán los "hijos" del nodo X_1.

---------------------------------------------------

PASO 5: Agrupar los árboles según los intervalos

Intervalo de X_1	Árboles que caen en este rango
(-∞, 4.0]	        Árbol 2
(4.0, 5.0]	        Árbol 1
(5.0, ∞)	        Ningún árbol, crear hoja

---------------------------------------------------

PASO 6: Fusionar los árboles en cada intervalo

🔹Intervalo (-∞, 4.0]
Solo Árbol 2, por lo que se mantiene igual:

X_1 < 4.0
    ├── X_2 < 2.5 → Clase A
    └── X_2 ≥ 2.5 → Clase C


🔹Intervalo (4.0, 5.0]
Solo Árbol 1, por lo que se mantiene igual:

X_1 < 5.0
    ├── X_2 < 3.0 → Clase A
    └── X_2 ≥ 3.0 → Clase B

    
🔹Intervalo (5.0, ∞)
No hay árboles, por lo que se crea una hoja usando mayoría de clases:


    Árbol 1: A, B

    Árbol 2: A, C

    Árbol 3: A, C

Total:

    A → 3 veces

    B → 1 vez

    C → 2 veces

Gana Clase A → esa será la predicción por defecto en este intervalo vacío.

---------------------------------------------------

PASO 7: Construir el Árbol Final

X_1
├── (-∞, 4.0] → Árbol 2 (sin cambios)
├── (4.0, 5.0] → Árbol 1 (sin cambios)
└── (5.0, ∞) → Clase A  (porque es la más común)



'''