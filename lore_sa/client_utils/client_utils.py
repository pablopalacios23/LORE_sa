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
import pandas as pd
import re, os

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
        for i, col in enumerate(numeric_features):
            if col in x_named:
                val = x_named[col]
                idx = numeric_features.index(col)
                mean = scaler.mean_[idx]
                std = scaler.scale_[idx]
                data[col] = val * std + mean
            else:
                data[col] = None
        cat_map = encoder.dataset_descriptor["categorical"]
        for cat in cat_map:
            onehot_names = [c for c in feature_names if c.startswith(cat + "_")]
            val_found = None
            for c in onehot_names:
                if c in x_named and x_named[c] == 1:
                    val_found = c[len(cat) + 1 :]
                    break
            if val_found is not None:
                data[cat] = val_found.strip()
            else:
                data[cat] = None
        return pd.Series(data)

    def decode_Xtrain_to_df(self, X_test, numeric_features, encoder, scaler, feature_names):
        decoded_rows = []
        for x in X_test:
            decoded = self.decode_onehot_instance(x, numeric_features, encoder, scaler, feature_names)
            decoded_rows.append(decoded)
        df = pd.DataFrame(decoded_rows)
        return df
    







    # ========================================================
    # 📌 Impresión y conversión de árboles
    # ========================================================

    def print_tree_readable(self, node, feature_names, class_names, numeric_features, scaler, encoder, depth=0):
        indent = "|   " * depth
        if node.is_leaf:
            class_idx = int(np.argmax(node.labels))
            print(f"{indent}|--- class: {class_names[class_idx]}")
            return
        feat_name = feature_names[node.feat]

        # --- CASO NUMÉRICA ---
        if feat_name in numeric_features:
            idx = numeric_features.index(feat_name)
            threshold = node.thresh * scaler.scale_[idx] + scaler.mean_[idx]
            print(f"{indent}|--- {feat_name} <= {threshold:.2f}")
            self.print_tree_readable(node._left_child, feature_names, class_names, numeric_features, scaler, encoder, depth + 1)
            print(f"{indent}|--- {feat_name} > {threshold:.2f}")
            self.print_tree_readable(node._right_child, feature_names, class_names, numeric_features, scaler, encoder, depth + 1)
            return
        
        # --- CASO CATEGÓRICA ONE-HOT ---
        if "=" in feat_name:

            # Ejemplo: occupation= Adm-clerical
            var, valor = feat_name.split("=")
            var = var.strip(); valor = valor.strip()
            # Si threshold == 0.5, OneHot típico: <= 0.5 (no es ese valor), > 0.5 (es ese valor)
            if node.thresh == 0.5:
                print(f"{indent}|--- {var} == \"{valor}\"")
                self.print_tree_readable(node._right_child, feature_names, class_names, numeric_features, scaler, encoder, depth + 1)
                print(f"{indent}|--- {var} != \"{valor}\"")
                self.print_tree_readable(node._left_child, feature_names, class_names, numeric_features, scaler, encoder, depth + 1)
            else:
                # Por si hay rarezas (poco frecuente)
                print(f"{indent}|--- {feat_name} <= {node.thresh:.2f}")
                self.print_tree_readable(node._left_child, feature_names, class_names, numeric_features, scaler, encoder, depth + 1)
                print(f"{indent}|--- {feat_name} > {node.thresh:.2f}")
                self.print_tree_readable(node._right_child, feature_names, class_names, numeric_features, scaler, encoder, depth + 1)
            return
        
        # --- SI NO ENCAJA ---
        print(f"{indent}|--- {feat_name} <= {node.thresh:.2f}")
        self.print_tree_readable(node._left_child, feature_names, class_names, numeric_features, scaler, encoder, depth + 1)
        print(f"{indent}|--- {feat_name} > {node.thresh:.2f}")
        self.print_tree_readable(node._right_child, feature_names, class_names, numeric_features, scaler, encoder, depth + 1)

    def tree_to_str(self, node, feature_names, numeric_features=None, scaler=None, global_mapping=None, unique_labels=None, depth=0):
        indent = "  " * depth
        result = ""
        if node.is_leaf:
            class_idx = int(np.argmax(node.labels))
            class_label = unique_labels[class_idx] if unique_labels is not None else str(class_idx)
            result += f'{indent}⮕ Leaf: class = "{class_label.strip()}" | {node.labels}\n'
        else:
            fname = feature_names[node.feat]

            # --- Split OneHot ---
            if "_" in fname:
                var, val = fname.split("_", 1)
                var = var.strip()
                val = val.strip()
                for i, child in enumerate(node.children):
                    cond = f'{var} {"≠" if i == 0 else "="} "{val}"'
                    result += f"{indent}if {cond}\n"
                    result += self.tree_to_str(child, feature_names, numeric_features, scaler, global_mapping, unique_labels, depth + 1)

            # --- Split categórico ordinal ---
            elif global_mapping and fname in global_mapping:
                vals_cat = global_mapping[fname]
                for i, child in enumerate(node.children):
                    val_idx = node.intervals[i] if hasattr(node, "intervals") and i < len(node.intervals) else int(getattr(node, "thresh", 0))
                    val = vals_cat[val_idx] if val_idx < len(vals_cat) else f"desconocido({val_idx})"
                    cond = f'{fname} {"≠" if i == 0 else "="} "{val}"'
                    result += f"{indent}if {cond}\n"
                    result += self.tree_to_str(child, feature_names, numeric_features, scaler, global_mapping, unique_labels, depth + 1)

            # --- Split numérico robusto ---
            elif numeric_features and fname in numeric_features:
                idx = numeric_features.index(fname)
                mean = scaler.mean_[idx]
                std = scaler.scale_[idx]
                # bounds siempre de tamaño len(children)+1 si es correcto
                bounds = [-np.inf] + list(getattr(node, "intervals", []))
                for i, child in enumerate(node.children):
                    left = bounds[i]
                    # Si hay suficientes bounds, usa el siguiente, si no, pon np.inf
                    right = bounds[i + 1] if i + 1 < len(bounds) else np.inf
                    left_real = left * std + mean if np.isfinite(left) else -np.inf
                    right_real = right * std + mean if np.isfinite(right) else np.inf
                    if i == 0:
                        cond = f"{fname} ≤ {right_real:.2f}"
                    elif i == len(node.children) - 1:
                        cond = f"{fname} > {left_real:.2f}"
                    else:
                        cond = f"{fname} ∈ ({left_real:.2f}, {right_real:.2f}]"
                    result += f"{indent}if {cond}\n"
                    result += self.tree_to_str(child, feature_names, numeric_features, scaler, global_mapping, unique_labels, depth + 1)
            else:
                # Por si acaso, caso no detectado
                for child in node.children:
                    result += f"{indent}if {fname} ?\n"
                    result += self.tree_to_str(child, feature_names, numeric_features, scaler, global_mapping, unique_labels, depth + 1)
        return result
    











    
    # ========================================================
    # 📌 Visualización y guardado de árboles (Graphviz)
    # ========================================================

    def save_supertree_plot(self, root_node, round_number, feature_names, class_names, numeric_features, scaler, global_mapping, folder="Supertree"):
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
                if "_" in fname:
                    var, val = fname.split("_", 1)
                    label = var.strip()
                else:
                    label = fname
            dot.node(curr, label)
            if parent:
                dot.edge(parent, curr, label=edge_label)

            # Nodos hijos (solo binario)
            if not node.is_leaf:
                fname = feature_names[node.feat]
                if "_" in fname: # OneHotEncoder
                    var, val = fname.split("_", 1)
                    var = var.strip()
                    val = val.strip()
                    left_label = f'≠ "{val}"'
                    right_label = f'= "{val}"'
                    add_node(node.children[0], curr, left_label)
                    add_node(node.children[1], curr, right_label)
                elif fname in numeric_features:
                    idx = numeric_features.index(fname)
                    mean = scaler.mean_[idx]
                    std = scaler.scale_[idx]
                    threshold = node.intervals[0]
                    thresh_real = threshold * std + mean if np.isfinite(threshold) else threshold
                    add_node(node.children[0], curr, f"≤ {thresh_real:.2f}")
                    add_node(node.children[1], curr, f"> {thresh_real:.2f}")
                elif fname in global_mapping:
                    vals_cat = global_mapping[fname]
                    val = vals_cat[node.intervals[0]] if node.intervals and len(node.intervals) > 0 else "?"
                    add_node(node.children[0], curr, f'= "{val}"')
                    add_node(node.children[1], curr, f'≠ "{val}"')
                else:
                    for child in node.children:
                        add_node(child, curr, "?")
        folder_path = f"Ronda_{round_number}/{folder}"
        os.makedirs(folder_path, exist_ok=True)
        filename = f"{folder_path}/LoreTree_cliente{self.client_id}_Supertree_ronda_{round_number}"
        add_node(root_node)
        dot.render(filename, format="png", cleanup=True)
        return f"{filename}.png"

    def save_lore_tree_image(self, root_node, round_number, feature_names, numeric_features, scaler, unique_labels, encoder, tree_type="LoreTree", folder="LoreTree"):
        dot = Digraph()
        node_id = [0]
        def base_name(feat):
            match = re.match(r"([a-zA-Z0-9\- ]+)", feat)
            return match.group(1).strip() if match else feat
        def add_node(node, parent=None, edge_label=""):
            curr = str(node_id[0])
            node_id[0] += 1 
            if node.is_leaf:
                class_index = int(np.argmax(node.labels))
                class_label = unique_labels[class_index]
                label = f"class: {class_label}\n{node.labels}"
            else:
                try:
                    fname = feature_names[node.feat]
                    label = base_name(fname)
                except:
                    label = f"X_{node.feat}"
            dot.node(curr, label)
            if parent:
                dot.edge(parent, curr, label=edge_label)

            # Árbol binario
            if not node.is_leaf:
                fname = feature_names[node.feat]
                if "_" in fname or "=" in fname:
                    if "_" in fname:
                        var, val = fname.split("_", 1)
                    else:
                        var, val = fname.split("=", 1)
                    var = var.strip()
                    val = val.strip()
                    left_label = f'≠ "{val}"'
                    right_label = f'= "{val}"'
                elif base_name(fname) in encoder.dataset_descriptor["categorical"]:
                    val_idx = int(node.thresh)
                    vals_cat = encoder.dataset_descriptor["categorical"][base_name(fname)]["distinct_values"]
                    val = vals_cat[val_idx] if val_idx < len(vals_cat) else f"desconocido({val_idx})"
                    left_label = f'= "{val}"'
                    right_label = f'≠ "{val}"'
                elif fname in numeric_features:
                    idx = numeric_features.index(fname)
                    mean = scaler.mean_[idx]
                    std = scaler.scale_[idx]
                    thresh = node.thresh * std + mean
                    left_label = f"<= {thresh:.2f}"
                    right_label = f"> {thresh:.2f}"
                else:
                    left_label = "≤ ?"
                    right_label = "> ?"
                add_node(node.children[0], curr, left_label)
                add_node(node.children[1], curr, right_label)
        folder_path = f"Ronda_{round_number}/{folder}"
        os.makedirs(folder_path, exist_ok=True)
        filename = f"{folder_path}/{tree_type.lower()}_cliente_{self.client_id}_ronda_{round_number}"
        add_node(root_node)
        dot.render(filename, format="png", cleanup=True)
        return f"{filename}.png"
    
    
    def _save_local_tree(self, root_node, round_number, feature_names, numeric_features, scaler, unique_labels, encoder, tree_type= "LocalTree"):
        dot = Digraph()
        node_id = [0]

        def base_name(feat):
            # Extrae solo el nombre de la variable, antes de '_' o '=' o espacios
            match = re.match(r"([a-zA-Z0-9\- ]+)", feat)
            return match.group(1).strip() if match else feat

        def add_node(node, parent=None, edge_label=""):
            curr = str(node_id[0])
            node_id[0] += 1 

            # Etiqueta del nodo
            if node.is_leaf:
                class_index = np.argmax(node.labels)
                class_label = unique_labels[class_index]
                label = f"class: {class_label}\n{node.labels}"
            else:
                try:
                    fname = feature_names[node.feat]
                    label = base_name(fname)
                except:
                    label = f"X_{node.feat}"

            dot.node(curr, label)
            if parent:
                dot.edge(parent, curr, label=edge_label)

            # Árbol tipo SuperTree
            if hasattr(node, "children") and node.children is not None and hasattr(node, "intervals"):
                for i, child in enumerate(node.children):
                    try:
                        fname = feature_names[node.feat]
                    except:
                        fname = f"X_{node.feat}"

                    original_feat = base_name(fname)
                    if original_feat in encoder.dataset_descriptor["categorical"]:
                        val_idx = node.intervals[i] if i == 0 else node.intervals[i - 1]
                        val_idx = int(val_idx)
                        vals_cat = encoder.dataset_descriptor["categorical"][original_feat]["distinct_values"]
                        val = vals_cat[val_idx] if val_idx < len(vals_cat) else f"desconocido({val_idx})"
                        edge = f'= "{val}"' if i == 0 else f'≠ "{val}"'
                    elif original_feat in numeric_features:
                        idx = numeric_features.index(original_feat)
                        mean = scaler.mean_[idx]
                        std = scaler.scale_[idx]
                        val = node.intervals[i] if i == 0 else node.intervals[i - 1]
                        val = val * std + mean
                        edge = f"<= {val:.2f}" if i == 0 else f"> {val:.2f}"
                    else:
                        edge = "?"

                    add_node(child, curr, edge)

            elif hasattr(node, "_left_child") or hasattr(node, "_right_child"):
                try:
                    fname = feature_names[node.feat]
                except:
                    fname = f"X_{node.feat}"

                # Si es OneHot
                if "_" in fname:
                    var, val = fname.split("_", 1)
                    var = var.strip()
                    val = val.strip()
                    # La split es: Si sex_ Male <= 0.5  (NO es Male)
                    #              Si sex_ Male > 0.5   (SÍ es Male)
                    left_label = f'≠ "{val}"'   # <= 0.5 → no es ese valor
                    right_label = f'= "{val}"'  # > 0.5  → sí es ese valor
                else:
                    original_feat = base_name(fname)

                    if original_feat in encoder.dataset_descriptor["categorical"]:
                        val_idx = int(node.thresh)
                        vals_cat = encoder.dataset_descriptor["categorical"][original_feat]["distinct_values"]
                        val = vals_cat[val_idx] if val_idx < len(vals_cat) else f"desconocido({val_idx})"
                        left_label = f'= "{val}"'
                        right_label = f'≠ "{val}"'

                    elif fname in numeric_features:
                        idx = numeric_features.index(fname)
                        mean = scaler.mean_[idx]
                        std = scaler.scale_[idx]
                        thresh = node.thresh * std + mean
                        left_label = f"<= {thresh:.2f}"
                        right_label = f"> {thresh:.2f}"
                        
                    else:
                        left_label = "≤ ?"
                        right_label = "> ?"

                if node._left_child:
                    add_node(node._left_child, curr, left_label)
                if node._right_child:
                    add_node(node._right_child, curr, right_label)

        add_node(root_node)
        folder = f"Ronda_{round_number}/{tree_type}_Cliente_{self.client_id}"
        os.makedirs(folder, exist_ok=True)
        filepath = f"{folder}/{tree_type.lower()}_cliente_{self.client_id}_ronda_{round_number}"
        dot.render(filepath, format="png", cleanup=True)
