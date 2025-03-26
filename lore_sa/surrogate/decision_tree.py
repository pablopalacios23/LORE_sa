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
        for _ in range(self.n_estimators):
            Z_sample, Yb_sample = resample(Z, Yb)
            tree = DecisionTreeClassifier()
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

    def merge_trees(self):
        print("✅ merge_trees() fue llamado")
        supertree = SuperTree()
        roots = [supertree.rec_buildTree(tree, list(range(tree.n_features_in_))) for tree in self.trees]
        supertree.mergeDecisionTrees(roots, num_classes=self.trees[0].n_classes_)
        return supertree


class SuperTree(Surrogate):
    def __init__(self, kind=None, preprocessing=None):
        super(SuperTree, self).__init__(kind, preprocessing)
        self.root = None

    def train(self, Z, Yb, **kwargs):
        pass

    def merge_trees(self):
        return self.root

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
            else:
                val = x[node.feat]
                for i, thr in enumerate(node.intervals):
                    if val <= thr:
                        if feature_names[node.feat] in numeric_columns:
                            if i == 0:
                                premises.append(Expression(feature_names[node.feat], operator.le, thr))
                            else:
                                premises.append(Expression(feature_names[node.feat], operator.gt, node.intervals[i - 1]))
                        else:
                            premises.append(Expression(feature_names[node.feat], operator.eq, val))
                        return traverse(node.children[i], x)

                # Último intervalo: > último umbral
                last_thr = node.intervals[-1]
                if feature_names[node.feat] in numeric_columns:
                    premises.append(Expression(feature_names[node.feat], operator.gt, last_thr))
                else:
                    premises.append(Expression(feature_names[node.feat], operator.eq, val))
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

    class Node:
        def __init__(self, feat_num=None, weights=None, thresh=None, labels=None, is_leaf=False, impurity=1, **kwargs):
            self.feat = feat_num
            self.thresh = thresh
            self.is_leaf = is_leaf
            self._weights = weights
            self._left_child = kwargs.get('left_child', None)
            self._right_child = kwargs.get('right_child', None)
            self.children = kwargs.get('children', None)
            self.impurity = impurity
            self.labels = labels
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

    def rec_buildTree(self, dt: DecisionTreeClassifier, feature_used): # Recorre internamente el árbol de decisión y lo convierte en un árbol de decisión personalizado. Así podemos manipular los árboles fácilmente después (porque no dependes de la estructura rígida de sklearn)
        nodes = dt.tree_.__getstate__()['nodes']
        values = dt.tree_.__getstate__()['values']

        def createNode(idx):
            line = nodes[idx]
            if line[0] == -1:
                return self.Node(feat_num=None, thresh=None, labels=values[idx][0], is_leaf=True)
            LC = createNode(line[0])
            RC = createNode(line[1])
            return self.Node(feat_num=feature_used[line[2]], thresh=line[3], labels=values[idx], is_leaf=False, left_child=LC, right_child=RC)

        return createNode(0)

    def mergeDecisionTrees(self, roots, num_classes, level=0):
        if all(r.is_leaf for r in roots): # Combinar etiquetas y crear hoja con la clase más votada
            votes = [np.argmax(r.labels) for r in roots]
            val, cou = np.unique(votes, return_counts=True)
            labels = np.zeros(num_classes)
            for j, v in enumerate(val):
                labels[v] = cou[j]
            super_node = self.SuperNode(is_leaf=True, labels=labels, level=level + 1)
            self.root = super_node
            return super_node

        val, cou = np.unique([r.feat for r in roots if r.feat is not None], return_counts=True)
        if np.sum(cou) < len(roots) / 2:
            majority = [np.argmax(r.labels) for r in roots if r.is_leaf]
            if majority:
                val_out, cou_out = np.unique(majority, return_counts=True)
                if np.max(cou_out) >= (len(roots) / 2) + 1:
                    labels = np.zeros(num_classes)
                    for j, v in enumerate(val_out):
                        labels[v] = cou_out[j]
                    super_node = self.SuperNode(is_leaf=True, labels=labels, level=level + 1)
                    self.root = super_node
                    return super_node

        Xf = val[np.argmax(cou)] # ← feature más común
        If = sorted(set(r.thresh for r in roots if r.feat == Xf))
        If = np.array([[-np.inf] + If + [np.inf]]).T
        If = np.hstack([If[:-1], If[1:]])

        branches = []
        for r in roots:
            branches.append(self.computeBranch(r, If, Xf, verbose=False)) # Divide los árboles según esos intervalos

        children = []
        for j in range(len(If)):
            child_roots = [b[j] for b in branches]
            children.append(self.mergeDecisionTrees(child_roots, num_classes, level + 1))

        super_node = self.SuperNode(feat_num=Xf, intervals=If[:, 1], children=children, level=level)
        self.root = super_node
        return super_node # Al final tienes un árbol combinado en forma de SuperNode que contiene:

    class SuperNode:
        def __init__(self, feat_num=None, intervals=None, weights=None, labels=None, children=None, is_leaf=False, level=0):
            self.feat = feat_num
            self.intervals = intervals
            self.labels = labels
            self.children = children
            self.is_leaf = is_leaf
            self.level = level
            self._weights = weights

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

    def computeBranch(self, node, intervals, feature_idx, verbose=False):
        if node is None:
            return [None] * len(intervals)
        if node.is_leaf:
            return [self.Node(labels=node.labels, is_leaf=True) for _ in intervals]
        if node.feat != feature_idx:
            left = self.computeBranch(node._left_child, intervals, feature_idx, verbose)
            right = self.computeBranch(node._right_child, intervals, feature_idx, verbose)
            return [self.Node(feat_num=node.feat, thresh=node.thresh, left_child=l, right_child=r) for l, r in zip(left, right)]

        splits = []
        for a, b in intervals:
            if node.thresh <= a:
                splits.append(self.computeBranch(node._right_child, [(a, b)], feature_idx, verbose)[0])
            elif node.thresh >= b:
                splits.append(self.computeBranch(node._left_child, [(a, b)], feature_idx, verbose)[0])
            else:
                left = self.computeBranch(node._left_child, [(a, node.thresh)], feature_idx, verbose)[0]
                right = self.computeBranch(node._right_child, [(node.thresh, b)], feature_idx, verbose)[0]
                splits.append(self.Node(feat_num=feature_idx, thresh=node.thresh, left_child=left, right_child=right))
        return splits
    
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

Árbol 1 → `X_1`
Árbol 2 → `X_1`
Árbol 3 → `X_3`

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