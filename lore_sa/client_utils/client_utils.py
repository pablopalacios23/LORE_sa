# ============================================================
# client_utils.py
# ------------------------------------------------------------
# Utilidades comunes para clientes Flower
# Mantiene los mismos nombres de métodos que usas en FlowerClient.
# Se usa como mixin:
#     class FlowerClient(NumPyClient, ClientUtilsMixin)
# ============================================================

from graphviz import Digraph
import numpy as np
import re, os
from pathlib import Path
import pandas as pd
from filelock import FileLock  # pip install filelock
import pandas as pd, os

class ClientUtilsMixin:

    # ========================================================
    # 📌 Extracción y simplificación de reglas
    # ========================================================


    def extract_rules_from_str(self, tree_str, target_class_label, exclude=False):
        target_class_label = target_class_label.strip()
        lines = tree_str.strip().split("\n")
        path = []
        rules = []
        def recurse(idx, indent_level):
            seen = set()
            while idx < len(lines):
                line = lines[idx]
                current_indent = len(line) - len(line.lstrip())
                if current_indent < indent_level:
                    return idx
                if "⮕" in line:
                    m = re.search(r'class = "([^"]+)"', line)
                    leaf_class = m.group(1).strip() if m else None
                    condition = (leaf_class == target_class_label)
                    if exclude:
                        condition = not condition  # cambia la lógica
                    if condition:
                        cleaned = []
                        for cond in path:
                            if cond not in seen:
                                cleaned.append(cond)
                                seen.add(cond)
                        rules.append(cleaned)
                    return idx + 1
                elif "if" in line:
                    condition = line.strip()[3:]
                    path.append(condition)
                    idx = recurse(idx + 1, current_indent + 2)
                    path.pop()
                else:
                    idx += 1
            return idx
        recurse(0, 0)
        return rules

    def _simplify_rule(self, regla, mode='tight'):
        """
        mode:
        - 'tight' (por defecto): mantiene la regla equivalente (lb = max lowers, ub = min uppers).
        - 'loose': presenta una banda más ancha (lb = min lowers, ub = max uppers).
        """
                
        import re
        bounds = {}   # var -> dict(lb, lb_inc, ub, ub_inc)
        cat_eq = {}   # var -> set(values)
        cat_neq = {}  # var -> set(values)
        others = []   # condiciones originales

        def ensure_num(var):
            if var not in bounds:
                bounds[var] = {"lb": None, "lb_inc": False, "ub": None, "ub_inc": True}

        if mode == 'tight':
            def upd_lower(d, v, inc):
                if d["lb"] is None or (v > d["lb"]) or (v == d["lb"] and d["lb_inc"] and not inc):
                    d["lb"] = v; d["lb_inc"] = inc
            def upd_upper(d, v, inc):
                if d["ub"] is None or (v < d["ub"]) or (v == d["ub"] and d["ub_inc"] and not inc):
                    d["ub"] = v; d["ub_inc"] = inc
        else:
            def upd_lower(d, v, inc):
                if d["lb"] is None or (v < d["lb"]) or (v == d["lb"] and not d["lb_inc"] and inc):
                    d["lb"] = v; d["lb_inc"] = inc
            def upd_upper(d, v, inc):
                if d["ub"] is None or (v > d["ub"]) or (v == d["ub"] and not d["ub_inc"] and inc):
                    d["ub"] = v; d["ub_inc"] = inc

        r_le  = re.compile(r'^\s*(.+?)\s*≤\s*([\-]?\d+(?:\.\d+)?)\s*$')
        r_lt  = re.compile(r'^\s*(.+?)\s*<\s*([\-]?\d+(?:\.\d+)?)\s*$')
        r_ge  = re.compile(r'^\s*(.+?)\s*≥\s*([\-]?\d+(?:\.\d+)?)\s*$')
        r_ge2 = re.compile(r'^\s*(.+?)\s*>=\s*([\-]?\d+(?:\.\d+)?)\s*$')
        r_gt  = re.compile(r'^\s*(.+?)\s*>\s*([\-]?\d+(?:\.\d+)?)\s*$')
        r_eq  = re.compile(r'^\s*(.+?)\s*=\s*"?([^"]+?)"?\s*$')
        r_neq = re.compile(r'^\s*(.+?)\s*≠\s*"?([^"]+?)"?\s*$')

        simplified_out = []

        for cond in regla:
            c = cond.strip()
            if "∧" in c:
                others.append(c); continue

            m = r_le.match(c)
            if m:
                var, val = m.group(1).strip(), float(m.group(2))
                if var in self.numeric_features:
                    ensure_num(var); upd_upper(bounds[var], val, inc=True); continue

            m = r_lt.match(c)
            if m:
                var, val = m.group(1).strip(), float(m.group(2))
                if var in self.numeric_features:
                    ensure_num(var); upd_upper(bounds[var], val, inc=False); continue

            m = r_ge.match(c) or r_ge2.match(c)
            if m:
                var, val = m.group(1).strip(), float(m.group(2))
                if var in self.numeric_features:
                    ensure_num(var); upd_lower(bounds[var], val, inc=True); continue

            m = r_gt.match(c)
            if m:
                var, val = m.group(1).strip(), float(m.group(2))
                if var in self.numeric_features:
                    ensure_num(var); upd_lower(bounds[var], val, inc=False); continue

            m = r_eq.match(c)
            if m:
                var, val = m.group(1).strip(), m.group(2).strip()
                if var not in self.numeric_features:
                    cat_eq.setdefault(var, set()).add(val); continue

            m = r_neq.match(c)
            if m:
                var, val = m.group(1).strip(), m.group(2).strip()
                if var not in self.numeric_features:
                    cat_neq.setdefault(var, set()).add(val); continue

            others.append(c)

        for var, d in bounds.items():
            lb, li = d["lb"], d["lb_inc"]
            ub, ui = d["ub"], d["ub_inc"]
            if lb is not None and ub is not None:
                if (lb > ub) or (lb == ub and (not li or not ui)):
                    continue
            if lb is not None and ub is not None:
                op_lb = "≥" if li else ">"
                op_ub = "≤" if ui else "<"
                simplified_out.append(f"{var} {op_lb} {lb:.2f} ∧ {op_ub} {ub:.2f}")
            elif lb is not None:
                op_lb = "≥" if li else ">"
                simplified_out.append(f"{var} {op_lb} {lb:.2f}")
            elif ub is not None:
                op_ub = "≤" if ui else "<"
                simplified_out.append(f"{var} {op_ub} {ub:.2f}")

        for var, vals in cat_eq.items():
            for v in sorted(vals):
                simplified_out.append(f'{var} = "{v}"')
        for var, vals in cat_neq.items():
            for v in sorted(vals):
                simplified_out.append(f'{var} ≠ "{v}"')

        simplified_out.extend(others)

        def _key(c):
            var = c.split()[0]
            return (0 if var in self.numeric_features else 1, var)
        simplified_out.sort(key=_key)
        return simplified_out

    def _simplify_rules_by_class(self, cf_rules_por_clase, mode='tight'):
        return {clase: self._simplify_rule(regla, mode=mode)
                for clase, regla in cf_rules_por_clase.items()}
    







    # ========================================================
    # 📌 Decodificación de instancias (OneHot → legible)
    # ========================================================

    def decode_onehot_instance(self, X_row, numeric_features, encoder, scaler, feature_names):
        x_named = pd.Series(X_row, index=feature_names)
        data = {}

        # ---- numéricas ----
        for col in numeric_features:
            if col in x_named:
                val = float(x_named[col])
                idx = numeric_features.index(col)

                # Si hay scaler válido, desescala; si no, deja en crudo
                if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                    mean = float(scaler.mean_[idx])
                    std  = float(scaler.scale_[idx]) if scaler.scale_[idx] != 0 else 1.0
                    data[col] = val * std + mean
                elif isinstance(scaler, dict) and "mean" in scaler and "std" in scaler:
                    mean = float(scaler["mean"][idx])
                    std  = float(scaler["std"][idx]) if scaler["std"][idx] != 0 else 1.0
                    data[col] = val * std + mean
                else:
                    data[col] = val
            else:
                data[col] = None

        # ---- categóricas (one-hot) ----
        cat_map = encoder.dataset_descriptor["categorical"]
        for cat in cat_map:
            onehot_names = [c for c in feature_names if c.startswith(cat + "_")]
            val_found = None
            for c in onehot_names:
                if c in x_named and float(x_named[c]) >= 0.5:  # robusto a umbrales
                    val_found = c[len(cat) + 1 :]
                    break
            data[cat] = val_found.strip() if val_found is not None else None

        return pd.Series(data)

    def decode_Xtrain_to_df(self, X_mat, numeric_features, encoder, scaler, feature_names):
        rows = [self.decode_onehot_instance(x, numeric_features, encoder, scaler, feature_names) for x in X_mat]
        return pd.DataFrame(rows)
    







    # ========================================================
    # 📌 Impresión y conversión de árboles
    # ========================================================

    def print_tree_readable(self, node, feature_names, class_names, numeric_features, encoder, depth=0):
        indent = "|   " * depth

        if node.is_leaf:
            class_idx = int(np.argmax(node.labels))
            print(f"{indent}|--- class: {class_names[class_idx]}")
            return

        # Nombre de la feature del nodo
        try:
            feat_name = feature_names[node.feat]
        except Exception:
            feat_name = f"X_{node.feat}"

        # ---------- CASO NUMÉRICA (en crudo) ----------
        if feat_name in numeric_features:
            thr = node.thresh  # <-- ya en crudo, no desescalar
            print(f"{indent}|--- {feat_name} <= {thr:.2f}")
            self.print_tree_readable(node._left_child, feature_names, class_names, numeric_features, encoder, depth + 1)
            print(f"{indent}|--- {feat_name} > {thr:.2f}")
            self.print_tree_readable(node._right_child, feature_names, class_names, numeric_features, encoder, depth + 1)
            return

        # ---------- CASO CATEGÓRICA ONE-HOT ----------
        # Soporta "var=valor" y "var_valor"
        if ("=" in feat_name) or ("_" in feat_name):
            if "=" in feat_name:
                var, valor = feat_name.split("=", 1)
            else:
                var, valor = feat_name.split("_", 1)
            var = var.strip()
            valor = valor.strip()

            # Umbral típico one-hot: <= 0.5 (NO es ese valor) / > 0.5 (SÍ es ese valor)
            if abs(node.thresh - 0.5) < 1e-8:
                # primero rama derecha (= valor), luego izquierda (≠ valor) o viceversa;
                # aquí mantenemos el orden clásico: izquierda <= 0.5 ⇒ ≠ ; derecha > 0.5 ⇒ =
                print(f"{indent}|--- {var} == \"{valor}\"")
                self.print_tree_readable(node._right_child, feature_names, class_names, numeric_features, encoder, depth + 1)
                print(f"{indent}|--- {var} != \"{valor}\"")
                self.print_tree_readable(node._left_child, feature_names, class_names, numeric_features, encoder, depth + 1)
            else:
                # raro, pero por si el split no está en 0.5
                print(f"{indent}|--- {feat_name} <= {node.thresh:.2f}")
                self.print_tree_readable(node._left_child, feature_names, class_names, numeric_features, encoder, depth + 1)
                print(f"{indent}|--- {feat_name} > {node.thresh:.2f}")
                self.print_tree_readable(node._right_child, feature_names, class_names, numeric_features, encoder, depth + 1)
            return

        # ---------- CATEGÓRICA NO ONE-HOT (opcional) ----------
        # Si quieres soportar categóricas no one-hot usando el descriptor del encoder:
        # original = feat_name.split("=",1)[0].split("_",1)[0].strip()
        # if original in encoder.dataset_descriptor.get("categorical", {}):
        #     # Si el árbol guardó índices en node.thresh, se podría mapear a string:
        #     vals = encoder.dataset_descriptor["categorical"][original]["distinct_values"]
        #     val_idx = int(getattr(node, "thresh", 0))
        #     val = vals[val_idx] if 0 <= val_idx < len(vals) else f"desconocido({val_idx})"
        #     print(f'{indent}|--- {original} == "{val}"')
        #     self.print_tree_readable(node._right_child, feature_names, class_names, numeric_features, encoder, depth + 1)
        #     print(f'{indent}|--- {original} != "{val}"')
        #     self.print_tree_readable(node._left_child, feature_names, class_names, numeric_features, encoder, depth + 1)
        #     return

        # ---------- fallback ----------
        print(f"{indent}|--- {feat_name} <= {node.thresh:.2f}")
        self.print_tree_readable(node._left_child, feature_names, class_names, numeric_features, encoder, depth + 1)
        print(f"{indent}|--- {feat_name} > {node.thresh:.2f}")
        self.print_tree_readable(node._right_child, feature_names, class_names, numeric_features, encoder, depth + 1)

    def tree_to_str(self,node,feature_names,numeric_features=None,scaler=None,global_mapping=None,unique_labels=None,depth=0):
        indent = "  " * depth
        result = ""

        if node.is_leaf:
            class_idx = int(np.argmax(node.labels))
            class_label = unique_labels[class_idx] if unique_labels is not None else str(class_idx)
            return f'{indent}⮕ Leaf: class = "{class_label.strip()}" | {node.labels}\n'

        fname = feature_names[node.feat]

        # --- OneHot (binario): hijos [≠ val, = val]
        if "_" in fname:
            var, val = fname.split("_", 1)
            var = var.strip(); val = val.strip()
            for i, child in enumerate(node.children):
                cond = f'{var} {"≠" if i == 0 else "="} "{val}"'
                result += f"{indent}if {cond}\n"
                result += self.tree_to_str(child, feature_names, numeric_features, None, global_mapping, unique_labels, depth + 1)
            return result

        # --- Categórica ordinal con mapping global
        if global_mapping and fname in global_mapping:
            vals_cat = global_mapping[fname]
            for i, child in enumerate(node.children):
                try:
                    val_idx = node.intervals[i] if hasattr(node, "intervals") and i < len(node.intervals) else int(getattr(node, "thresh", 0))
                except Exception:
                    val_idx = 0
                val = vals_cat[val_idx] if 0 <= val_idx < len(vals_cat) else f"desconocido({val_idx})"
                cond = f'{fname} {"≠" if i == 0 else "="} "{val}"'
                result += f"{indent}if {cond}\n"
                result += self.tree_to_str(child, feature_names, numeric_features, None, global_mapping, unique_labels, depth + 1)
            return result

        # --- Numérica (en crudo, sin desescalar)
        if numeric_features and fname in numeric_features:
            # Construir límites crudos
            intervals = list(getattr(node, "intervals", []))
            # Si no hay intervals (p.ej. árbol binario con `thresh`), construiremos [-inf, thresh, +inf]
            if not intervals and hasattr(node, "thresh"):
                try:
                    intervals = [float(node.thresh)]
                except Exception:
                    intervals = []

            bounds = [-np.inf] + intervals
            # Asegurar len(bounds) = len(children)+1
            while len(bounds) < len(node.children) + 1:
                bounds.append(np.inf)

            for i, child in enumerate(node.children):
                left = bounds[i]
                right = bounds[i + 1]
                if i == 0:
                    cond = f"{fname} ≤ {right:.2f}" if np.isfinite(right) else f"{fname} ≤ ?"
                elif i == len(node.children) - 1:
                    cond = f"{fname} > {left:.2f}" if np.isfinite(left) else f"{fname} > ?"
                else:
                    ltxt = f"{left:.2f}" if np.isfinite(left) else "?"
                    rtxt = f"{right:.2f}" if np.isfinite(right) else "?"
                    cond = f"{fname} ∈ ({ltxt}, {rtxt}]"
                result += f"{indent}if {cond}\n"
                result += self.tree_to_str(child, feature_names, numeric_features, None, global_mapping, unique_labels, depth + 1)
            return result

        # --- Desconocido
        for child in node.children:
            result += f"{indent}if {fname} ?\n"
            result += self.tree_to_str(child, feature_names, numeric_features, None, global_mapping, unique_labels, depth + 1)
        return result
    











    
    # ========================================================
    # 📌 Visualización y guardado de árboles (Graphviz)
    # ========================================================

    def save_mergedTree_plot(self,root_node,round_number,feature_names,class_names,numeric_features,scaler, global_mapping,folder="Supertree"):
        from graphviz import Digraph
        import numpy as np
        import os

        dot = Digraph()
        node_id = [0]

        def add_node(node, parent=None, edge_label=""):
            curr = str(node_id[0])
            node_id[0] += 1

            # Etiqueta del nodo
            if node.is_leaf:
                class_index = int(np.argmax(node.labels))
                class_label = class_names[class_index]
                label = f"class: {class_label}\n{node.labels}"
            else:
                fname = feature_names[node.feat]
                if "_" in fname:  # OneHot
                    var, _ = fname.split("_", 1)
                    label = var.strip()
                else:
                    label = fname
            dot.node(curr, label)
            if parent:
                dot.edge(parent, curr, label=edge_label)

            # Hijos (binario)
            if not node.is_leaf:
                fname = feature_names[node.feat]

                # One-hot: hijos [≠ val, = val]
                if "_" in fname:
                    var, val = fname.split("_", 1)
                    var = var.strip(); val = val.strip()
                    add_node(node.children[0], curr, f'≠ "{val}"')
                    add_node(node.children[1], curr, f'= "{val}"')

                # Numérica en crudo: usar el threshold tal cual
                elif fname in numeric_features:
                    thr = node.intervals[0] if getattr(node, "intervals", None) else node.thresh
                    thr = float(thr) if np.isfinite(thr) else thr
                    add_node(node.children[0], curr, f"≤ {thr:.2f}" if np.isfinite(thr) else "≤ ?")
                    add_node(node.children[1], curr, f"> {thr:.2f}" if np.isfinite(thr) else "> ?")

                # Categórica ordinal con mapeo global: hijos [= val, ≠ val]
                elif fname in global_mapping:
                    vals_cat = global_mapping[fname]
                    idx = 0
                    try:
                        idx = int(node.intervals[0]) if getattr(node, "intervals", None) else int(getattr(node, "thresh", 0))
                    except Exception:
                        idx = 0
                    val = vals_cat[idx] if 0 <= idx < len(vals_cat) else "?"
                    add_node(node.children[0], curr, f'= "{val}"')
                    add_node(node.children[1], curr, f'≠ "{val}"')

                # Desconocido
                else:
                    for child in node.children:
                        add_node(child, curr, "?")

        folder_path = f"Ronda_{round_number}/{folder}"
        os.makedirs(folder_path, exist_ok=True)
        filename = f"{folder_path}/MergedTree_cliente{self.client_id}_Lore+Supertree_ronda_{round_number}"
        add_node(root_node)
        dot.render(filename, format="pdf", cleanup=True)
        return f"{filename}.pdf"

    def save_lore_tree_image(self,root_node,round_number,feature_names,numeric_features,unique_labels,encoder,tree_type="LoreTree",folder="LoreTree",):
        from graphviz import Digraph
        import os
        import numpy as np
        import re

        dot = Digraph()
        node_id = [0]

        def base_name(feat: str) -> str:
            m = re.match(r"([a-zA-Z0-9\- ]+)", feat)
            return m.group(1).strip() if m else feat

        def add_node(node, parent=None, edge_label: str = ""):
            curr = str(node_id[0]); node_id[0] += 1

            # etiqueta del nodo
            if getattr(node, "is_leaf", False):
                class_index = int(np.argmax(node.labels))
                class_label = unique_labels[class_index]
                label = f"class: {class_label}\n{node.labels}"
            else:
                try:
                    fname = feature_names[node.feat]
                    label = base_name(fname)
                except Exception:
                    label = f"X_{node.feat}"
            dot.node(curr, label)
            if parent:
                dot.edge(parent, curr, label=edge_label)

            # hijos (binario)
            if not getattr(node, "is_leaf", False):
                try:
                    fname = feature_names[node.feat]
                except Exception:
                    fname = f"X_{node.feat}"

                # --- One-hot: "var_valor" o "var=valor"
                if ("_" in fname) or ("=" in fname):
                    if "_" in fname:
                        var, val = fname.split("_", 1)
                    else:
                        var, val = fname.split("=", 1)
                    var = var.strip(); val = val.strip()

                    # Patrón típico: izquierda (<=0.5) ⇒ ≠, derecha (>0.5) ⇒ =
                    left_label  = f'≠ "{val}"'
                    right_label = f'= "{val}"'

                # --- Categórica no one-hot (raro si ya binarizas, pero por si acaso)
                elif base_name(fname) in encoder.dataset_descriptor.get("categorical", {}):
                    vals_cat = encoder.dataset_descriptor["categorical"][base_name(fname)]["distinct_values"]
                    val_idx = int(getattr(node, "thresh", 0))
                    val = vals_cat[val_idx] if 0 <= val_idx < len(vals_cat) else f"desconocido({val_idx})"
                    # mantenemos el mismo convenio binario
                    left_label  = f'= "{val}"'
                    right_label = f'≠ "{val}"'

                # --- Numérica en crudo
                elif fname in numeric_features:
                    thr = float(getattr(node, "thresh", 0.0))
                    left_label  = f"<= {thr:.2f}"
                    right_label = f"> {thr:.2f}"

                else:
                    left_label, right_label = "≤ ?", "> ?"

                # Recurse (asumimos binario)
                if hasattr(node, "children") and node.children and len(node.children) >= 2:
                    add_node(node.children[0], curr, left_label)
                    add_node(node.children[1], curr, right_label)
                else:
                    # compat con _left_child/_right_child
                    if getattr(node, "_left_child", None) is not None:
                        add_node(node._left_child, curr, left_label)
                    if getattr(node, "_right_child", None) is not None:
                        add_node(node._right_child, curr, right_label)

        # Guardar PDF
        folder_path = f"Ronda_{round_number}/{folder}"
        os.makedirs(folder_path, exist_ok=True)
        filename = f"{folder_path}/{tree_type.lower()}_cliente_{self.client_id}_ronda_{round_number}"
        add_node(root_node)
        dot.render(filename, format="pdf", cleanup=True)
        return f"{filename}.pdf"
    
    
    def _save_local_tree(
        self, root_node, round_number, feature_names, numeric_features,
        scaler=None, unique_labels=None, encoder=None, tree_type="LocalTree"
    ):
        dot = Digraph(); node_id = [0]

        def base_name(feat):
            match = re.match(r"([a-zA-Z0-9\- ]+)", feat)
            return match.group(1).strip() if match else feat

        def add_node(node, parent=None, edge_label=""):
            curr = str(node_id[0]); node_id[0] += 1

            # etiqueta del nodo
            if node.is_leaf:
                class_index = np.argmax(node.labels)
                class_label = unique_labels[class_index]
                label = f"class: {class_label}\n{node.labels}"
            else:
                try:    label = base_name(feature_names[node.feat])
                except: label = f"X_{node.feat}"
            dot.node(curr, label)
            if parent: dot.edge(parent, curr, label=edge_label)

            # --- SuperTree (intervalos múltiples) ---
            if hasattr(node, "children") and node.children is not None and hasattr(node, "intervals"):
                for i, child in enumerate(node.children):
                    try:    fname = feature_names[node.feat]
                    except: fname = f"X_{node.feat}"
                    original_feat = base_name(fname)

                    if original_feat in encoder.dataset_descriptor["categorical"]:
                        val_idx = node.intervals[i] if i == 0 else node.intervals[i-1]
                        val_idx = int(val_idx)
                        vals_cat = encoder.dataset_descriptor["categorical"][original_feat]["distinct_values"]
                        val = vals_cat[val_idx] if val_idx < len(vals_cat) else f"desconocido({val_idx})"
                        edge = f'= "{val}"' if i == 0 else f'≠ "{val}"'
                    elif original_feat in numeric_features:
                        # Árbol en crudo ⇒ umbral en escala cruda (NO desescalar)
                        val = node.intervals[i] if i == 0 else node.intervals[i-1]
                        edge = f"<= {val:.2f}" if i == 0 else f"> {val:.2f}"
                    else:
                        edge = "?"

                    add_node(child, curr, edge)

            # --- Árbol binario (left/right) ---
            elif hasattr(node, "_left_child") or hasattr(node, "_right_child"):
                try:    fname = feature_names[node.feat]
                except: fname = f"X_{node.feat}"

                if "_" in fname:  # one-hot
                    var, val = fname.split("_", 1)
                    left_label  = f'≠ "{val.strip()}"'   # <= 0.5
                    right_label = f'= "{val.strip()}"'   # > 0.5
                else:
                    original_feat = base_name(fname)
                    if original_feat in encoder.dataset_descriptor["categorical"]:
                        val_idx = int(node.thresh)
                        vals_cat = encoder.dataset_descriptor["categorical"][original_feat]["distinct_values"]
                        val = vals_cat[val_idx] if val_idx < len(vals_cat) else f"desconocido({val_idx})"
                        left_label, right_label = f'= "{val}"', f'≠ "{val}"'
                    elif fname in numeric_features:
                        # Árbol en crudo ⇒ umbral en escala cruda (NO desescalar)
                        thresh = node.thresh
                        left_label, right_label = f"<= {thresh:.2f}", f"> {thresh:.2f}"
                    else:
                        left_label, right_label = "≤ ?", "> ?"

                if node._left_child:  add_node(node._left_child,  curr, left_label)
                if node._right_child: add_node(node._right_child, curr, right_label)

        add_node(root_node)
        folder = f"Ronda_{round_number}/{tree_type}_Cliente_{self.client_id}"
        os.makedirs(folder, exist_ok=True)
        filepath = f"{folder}/{tree_type.lower()}_cliente_{self.client_id}_ronda_{round_number}"
        dot.render(filepath, format="pdf", cleanup=True)


    # Lore tree a escala global antes del merge

    def align_tree_local_to_global(self, root, feature_names, numeric_features,
                               mu_local, sd_local, mu_global, sd_global):
        import numpy as np, copy
        mu_local = np.asarray(mu_local, float); sd_local = np.asarray(sd_local, float)
        mu_global = np.asarray(mu_global, float); sd_global = np.asarray(sd_global, float)
        sd_local = np.where(sd_local==0, 1.0, sd_local)
        sd_global= np.where(sd_global==0, 1.0, sd_global)
        num_idx = {f:i for i,f in enumerate(numeric_features)}

        def walk(n):
            if getattr(n, "is_leaf", False): return
            fname = feature_names[n.feat]
            if fname in num_idx and hasattr(n, "intervals") and n.intervals:
                k = num_idx[fname]
                n.intervals = [ (t*sd_local[k] + mu_local[k] - mu_global[k]) / sd_global[k] for t in n.intervals ]
            for c in n.children: walk(c)

        r = copy.deepcopy(root); walk(r); return r
    

    # ---- Utils de estructura del árbol (SuperTree/LORE) ----
    def is_leaf(self, node):
        # Ajusta estos nombres si tu clase usa otros atributos
        if node is None:
            return True
        if hasattr(node, "is_leaf"):
            return bool(node.is_leaf)
        # Fallback: sin hijos (ni left/right ni children) => hoja
        has_left  = hasattr(node, "left")  and node.left  is not None
        has_right = hasattr(node, "right") and node.right is not None
        has_children = hasattr(node, "children") and node.children
        return not (has_left or has_right or has_children)

    def tree_depth_edges(self, node):
        """Profundidad (en aristas). Una hoja tiene profundidad 0."""
        if node is None or self.is_leaf(node):
            return 0
        # Soporta binario (left/right) o N-ario (children)
        if hasattr(node, "children") and node.children:
            return 1 + max(self.tree_depth_edges(ch) for ch in node.children)
        else:
            return 1 + max(self.tree_depth_edges(getattr(node, "left", None)),
                        self.tree_depth_edges(getattr(node, "right", None)))

    def count_nodes(self, node):
        if node is None:
            return 0
        if self.is_leaf(node):
            return 1
        total = 1
        if hasattr(node, "children") and node.children:
            total += sum(self.count_nodes(ch) for ch in node.children)
        else:
            total += self.count_nodes(getattr(node, "left", None))
            total += self.count_nodes(getattr(node, "right", None))
        return total

    def count_leaves(self, node):
        if node is None:
            return 0
        if self.is_leaf(node):
            return 1
        if hasattr(node, "children") and node.children:
            return sum(self.count_leaves(ch) for ch in node.children)
        else:
            return self.count_leaves(getattr(node, "left", None)) + self.count_leaves(getattr(node, "right", None))