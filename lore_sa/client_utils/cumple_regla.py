import numpy as np


class ReglaEvaluator:

    @staticmethod
    def cumple_regla(instancia, regla):
        import re

        def onehot_value(var: str):
            prefix = var + "_"
            cols = [k for k in instancia.keys() if k.startswith(prefix)]
            if not cols:
                return None
            best = max(cols, key=lambda c: float(instancia.get(c, 0.0)))
            return best.split(prefix, 1)[1]

        for cond in regla:

            cond = cond.strip()

            # normalizar operadores
            cond = cond.replace("≤", "<=").replace("≥", ">=")
            cond = cond.replace(" = ", "=")

            # ----------------------------
            # intervalo con ∧
            # ----------------------------
            if "∧" in cond:
                m = re.match(r'(.+?)([><]=?)\s*([-\d\.]+)\s*∧\s*([><]=?)\s*([-\d\.]+)', cond)
                if m:
                    var = m.group(1).strip()
                    op1, val1 = m.group(2), float(m.group(3))
                    op2, val2 = m.group(4), float(m.group(5))

                    v = float(instancia[var])

                    if not (eval(f"v {op1} {val1}") and eval(f"v {op2} {val2}")):
                        return False

                    continue

            # ----------------------------
            # <=
            # ----------------------------
            if "<=" in cond:
                var, val = cond.split("<=")
                if float(instancia[var.strip()]) > float(val.strip()):
                    return False

            # ----------------------------
            # >=
            # ----------------------------
            elif ">=" in cond:
                var, val = cond.split(">=")
                if float(instancia[var.strip()]) < float(val.strip()):
                    return False

            # ----------------------------
            # >
            # ----------------------------
            elif ">" in cond:
                var, val = cond.split(">")
                if float(instancia[var.strip()]) <= float(val.strip()):
                    return False

            # ----------------------------
            # <
            # ----------------------------
            elif "<" in cond:
                var, val = cond.split("<")
                if float(instancia[var.strip()]) >= float(val.strip()):
                    return False

            # ----------------------------
            # ≠
            # ----------------------------
            elif "≠" in cond:

                var, val = cond.split("≠")
                var = var.strip()
                val = val.strip().replace('"', "")

                if var in instancia and isinstance(instancia[var], str):
                    if instancia[var] == val:
                        return False
                    continue

                col = f"{var}_{val}"

                if col in instancia:
                    if float(instancia[col]) >= 0.5:
                        return False
                    continue

                oh = onehot_value(var)

                if oh is not None and oh == val:
                    return False

            # ----------------------------
            # =
            # ----------------------------
            elif "=" in cond:

                var, val = cond.split("=")
                var = var.strip()
                val = val.strip().replace('"', "")

                if var in instancia and isinstance(instancia[var], str):
                    if instancia[var] != val:
                        return False
                    continue

                col = f"{var}_{val}"

                if col in instancia:
                    if float(instancia[col]) < 0.5:
                        return False
                    continue

                oh = onehot_value(var)

                if oh is None or oh != val:
                    return False

            else:
                return False

        return True

    # --------------------------------------------------------

    @staticmethod
    def group_shap_values(shap_values, feature_names):

        grouped = {}

        for val, name in zip(shap_values, feature_names):

            if "_" in name:
                base = name.split("_")[0]
            else:
                base = name

            grouped[base] = grouped.get(base, 0) + val

        return grouped

    # --------------------------------------------------------

    @staticmethod
    def decode_lime_conditions(conds, row, feature_names):

        decoded = []

        row_dict = dict(zip(feature_names, row))

        cat_values = {}

        for f, v in row_dict.items():

            if "_" in f:
                base, cat = f.split("_", 1)

                if v == 1:
                    cat_values[base] = cat

        for c in conds:

            if "_" in c and "=" in c:

                var, val = c.split("=")
                base, cat = var.split("_", 1)

                if base in cat_values:
                    decoded.append(f"{base} = {cat_values[base]}")
                else:
                    decoded.append(c)

            else:
                decoded.append(c)

        return decoded

    # --------------------------------------------------------

    @staticmethod
    def decode_anchor_conditions(conds, row, feature_names):

        row_dict = dict(zip(feature_names, row))
        cat_values = {}

        for f, v in row_dict.items():

            if "_" in f:
                base, cat = f.split("_", 1)

                if v == 1:
                    cat_values[base] = cat

        decoded = []

        for c in conds:

            if "_" in c:

                var = c.split()[0]

                if "_" in var:

                    base, cat = var.split("_", 1)

                    if base in cat_values:
                        decoded.append(f"{base} = {cat_values[base]}")
                        continue

            decoded.append(c)

        return decoded