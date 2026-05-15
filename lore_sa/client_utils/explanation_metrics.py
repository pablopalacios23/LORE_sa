import re
import numpy as np
import pandas as pd

from lore_sa.client_utils.cumple_regla import ReglaEvaluator
from lore_sa.dataset.tabular_dataset import TabularDataset


class Explainer_metrics:

    @staticmethod
    def compute_top_shap_features(explainer, row, pred_class_idx, feature_names):
        shap_values = explainer.shap_values(row[None, :], nsamples=50, silent=True)
        shap_vals = np.asarray(shap_values)

        if shap_vals.ndim == 3:
            shap_vals = shap_vals[0, :, pred_class_idx]
        elif shap_vals.ndim == 2:
            shap_vals = shap_vals[0]

        shap_vals = shap_vals.flatten()

        shap_grouped = ReglaEvaluator.group_shap_values(shap_vals, feature_names)

        group_names = list(shap_grouped.keys())
        group_vals = np.array(list(shap_grouped.values()))

        topk = min(3, len(group_vals))
        top_idx = np.argsort(np.abs(group_vals))[-topk:][::-1]

        return [group_names[i] for i in top_idx]

    @staticmethod
    def compute_lime_rules(lime_explainer, row, predict_fn, feature_names):
        exp = lime_explainer.explain_instance(
            row,
            predict_fn,
            num_features=3,
        )

        rules = [f for f, _ in exp.as_list()]
        rules = ReglaEvaluator.decode_lime_conditions(rules, row, feature_names)

        return list(dict.fromkeys(rules))

    @staticmethod
    def compute_lime_coefs(
        lime_explainer,
        row,
        predict_fn,
        n_features,
        class_idx,
        random_state=None,
        normalize_l1=True,
    ):
        if random_state is not None:
            np.random.seed(random_state)
        exp = lime_explainer.explain_instance(
            row,
            predict_fn,
            labels=(class_idx,),
            num_features=n_features,
        )

        raw = exp.local_exp.get(class_idx, [])
        coefs = {idx: w for idx, w in raw}

        if not normalize_l1:
            return coefs

        norm = sum(abs(w) for w in coefs.values()) or 1.0
        return {idx: w / norm for idx, w in coefs.items()}

    @staticmethod
    def delta_lime(coefs_local, coefs_global):
        all_feats = set(coefs_local.keys()) | set(coefs_global.keys())
        if not all_feats:
            return np.nan

        diffs = [abs(coefs_local.get(i, 0.0) - coefs_global.get(i, 0.0)) for i in all_feats]
        return float(np.mean(diffs))

    @staticmethod
    def compute_shap_all(explainer, row, pred_class_idx, feature_names, normalize_l1=True):
        """Devuelve (top_features, coefs_dict) desde una única llamada a shap_values."""
        shap_values = explainer.shap_values(row[None, :], nsamples=50, silent=True)
        shap_vals = np.asarray(shap_values)

        if shap_vals.ndim == 3:
            vals = shap_vals[0, :, pred_class_idx].flatten()
        elif shap_vals.ndim == 2:
            vals = shap_vals[0].flatten()
        else:
            vals = shap_vals.flatten()

        # top_features (con agrupación de one-hot)
        shap_grouped = ReglaEvaluator.group_shap_values(vals, feature_names)
        group_names = list(shap_grouped.keys())
        group_vals = np.array(list(shap_grouped.values()))
        topk = min(3, len(group_vals))
        top_idx = np.argsort(np.abs(group_vals))[-topk:][::-1]
        top_features = [group_names[i] for i in top_idx]

        # coefs por feature index (L1-normalizados)
        coefs = {i: float(v) for i, v in enumerate(vals)}
        if normalize_l1:
            norm = sum(abs(v) for v in coefs.values()) or 1.0
            coefs = {i: v / norm for i, v in coefs.items()}

        return top_features, coefs

    @staticmethod
    def delta_shap(coefs_local, coefs_global):
        all_feats = set(coefs_local.keys()) | set(coefs_global.keys())
        if not all_feats:
            return np.nan
        diffs = [abs(coefs_local.get(i, 0.0) - coefs_global.get(i, 0.0)) for i in all_feats]
        return float(np.mean(diffs))

    @staticmethod
    def compute_anchor_rules(anchor_explainer, row, threshold, feature_names):
        anchor_exp = anchor_explainer.explain(
            row, threshold=threshold, delta=0.2, beam_size=3, n_samples=1000
        )

        rules = anchor_exp.anchor
        rules = ReglaEvaluator.decode_anchor_conditions(rules, row, feature_names)

        return list(dict.fromkeys(rules))

    @staticmethod
    def compute_anchor_all(anchor_explainer, row, threshold, feature_names):
        """Devuelve (rules, precision, coverage) desde una única llamada a explain."""
        anchor_exp = anchor_explainer.explain(
            row, threshold=threshold, delta=0.2, beam_size=3, n_samples=300
        )

        rules = ReglaEvaluator.decode_anchor_conditions(anchor_exp.anchor, row, feature_names)
        rules = list(dict.fromkeys(rules))

        if not rules:
            precision = np.nan
            coverage  = np.nan
        else:
            precision = float(anchor_exp.precision) if anchor_exp.precision is not None else np.nan
            coverage  = float(anchor_exp.coverage)  if anchor_exp.coverage  is not None else np.nan

        return rules, precision, coverage

    @staticmethod
    def _parse_condition_to_interval(condition, feature_ranges):
        """
        Parsea una condición numérica (Anchor o LORE) a (feature, lo, hi).
        Devuelve None para condiciones categóricas o no parseables.
        """
        condition = condition.strip()
        # normalizar operadores Unicode de LORE
        condition = condition.replace("≤", "<=").replace("≥", ">=")

        # Rango Anchor: "lo < feature <= hi"
        range_match = re.match(
            r'([\d.eE+\-]+)\s*(?:<|<=)\s*(\S+)\s*(?:<|<=)\s*([\d.eE+\-]+)',
            condition
        )
        if range_match:
            feat = range_match.group(2)
            if feat in feature_ranges:
                return feat, float(range_match.group(1)), float(range_match.group(3))
            return None

        # Intervalo LORE: "feature ∈ (lo, hi]"
        lore_interval = re.match(
            r'(\S+)\s*∈\s*\(?([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\]?',
            condition
        )
        if lore_interval:
            feat = lore_interval.group(1)
            if feat in feature_ranges:
                return feat, float(lore_interval.group(2)), float(lore_interval.group(3))
            return None

        # Simple: "feature > value" o "feature <= value" (Anchor y LORE)
        simple_match = re.match(r'(\S+)\s*(>|>=|<|<=)\s*([\d.eE+\-]+)', condition)
        if simple_match:
            feat = simple_match.group(1)
            op   = simple_match.group(2)
            val  = float(simple_match.group(3))
            if feat not in feature_ranges:
                return None
            fmin, fmax = feature_ranges[feat]
            lo = val  if op in ('>', '>=') else fmin
            hi = fmax if op in ('>', '>=') else val
            return feat, lo, hi

        return None  # categórica ("Sex = Female") u otra

    @staticmethod
    def _parse_categorical_condition(condition):
        """Parsea 'feature = category' (categórica decodificada). Devuelve (feat, val) o None."""
        condition = condition.strip()
        eq_match = re.match(r'(\S+)\s*=\s*(.+)', condition)
        if eq_match:
            val = eq_match.group(2).strip().strip('"')
            try:
                float(val)
                return None
            except ValueError:
                return eq_match.group(1), val
        return None

    @staticmethod
    def compute_additive_similarity(rules1, rules2, feature_ranges):
        """
        Similitud Aditiva: media del Jaccard por variable entre dos reglas.
        - Numéricas: S = |intersección| / |unión| de los intervalos crudos.
        - Categóricas: S = 1 si coinciden, 0 si difieren (solo si ambas mencionan la feature).
        - Features no mencionadas en ninguna regla: ignoradas.
        - Features numéricas mencionadas en solo una: la otra toma rango completo.
        Devuelve NaN si no hay ninguna feature comparable.
        """
        if not rules1 or not rules2:
            return np.nan

        def parse_rules(rules):
            intervals = {}
            categoricals = {}
            for c in rules:
                r = Explainer_metrics._parse_condition_to_interval(c, feature_ranges)
                if r is not None:
                    intervals[r[0]] = (r[1], r[2])
                else:
                    cat = Explainer_metrics._parse_categorical_condition(c)
                    if cat is not None:
                        categoricals[cat[0]] = cat[1]
            return intervals, categoricals

        i1, cat1 = parse_rules(rules1)
        i2, cat2 = parse_rules(rules2)

        scores = []

        # Numéricas: Jaccard de intervalos crudos
        for feat in set(i1.keys()) | set(i2.keys()):
            fmin, fmax = feature_ranges[feat]
            lo1, hi1 = i1.get(feat, (fmin, fmax))
            lo2, hi2 = i2.get(feat, (fmin, fmax))
            inter = max(0.0, min(hi1, hi2) - max(lo1, lo2))
            union = max(hi1, hi2) - min(lo1, lo2)
            scores.append(inter / union if union > 0 else 1.0)

        # Categóricas: 1 si coinciden, 0 si difieren (solo features mencionadas en ambas)
        for feat in set(cat1.keys()) & set(cat2.keys()):
            scores.append(1.0 if cat1[feat] == cat2[feat] else 0.0)

        if not scores:
            return np.nan

        return float(np.mean(scores))

    @staticmethod
    def compute_structural_jaccard(rules1, rules2, feature_ranges):
        """
        Jaccard estructural de hipervolúmenes entre dos conjuntos de reglas anchor.
        Normaliza cada dimensión por el rango total de la feature en feature_ranges.
        Categóricas: si ambas mencionan la misma feature con distinto valor → 0.0 (intersección vacía).
        Si coinciden → no penaliza. Si solo una la menciona → no penaliza.
        Devuelve NaN si alguna regla está vacía o solo tiene condiciones no parseables.
        """
        if not rules1 or not rules2:
            return np.nan

        def parse_rules(rules):
            intervals = {}
            categoricals = {}
            for c in rules:
                r = Explainer_metrics._parse_condition_to_interval(c, feature_ranges)
                if r is not None:
                    intervals[r[0]] = (r[1], r[2])
                else:
                    cat = Explainer_metrics._parse_categorical_condition(c)
                    if cat is not None:
                        categoricals[cat[0]] = cat[1]
            return intervals, categoricals

        i1, cat1 = parse_rules(rules1)
        i2, cat2 = parse_rules(rules2)

        # Factor categórico: desacuerdo → intersección vacía → Jaccard = 0
        for feat, val in cat1.items():
            if feat in cat2 and cat2[feat] != val:
                return 0.0

        relevant = set(i1.keys()) | set(i2.keys())
        if not relevant:
            return np.nan

        def volume(intervals):
            v = 1.0
            for feat in relevant:
                fmin, fmax = feature_ranges[feat]
                span = fmax - fmin
                if span <= 0:
                    continue
                lo, hi = intervals.get(feat, (fmin, fmax))
                v *= max(0.0, (hi - lo) / span)
            return v

        # Intersección: rango más restrictivo por feature
        i_int = {}
        for feat in relevant:
            fmin, fmax = feature_ranges[feat]
            lo1, hi1 = i1.get(feat, (fmin, fmax))
            lo2, hi2 = i2.get(feat, (fmin, fmax))
            lo = max(lo1, lo2)
            hi = min(hi1, hi2)
            if lo >= hi:
                return 0.0  # intersección vacía
            i_int[feat] = (lo, hi)

        v1    = volume(i1)
        v2    = volume(i2)
        v_int = volume(i_int)

        denom = v1 + v2 - v_int
        if denom <= 0:
            return np.nan

        return float(v_int / denom)

    @staticmethod
    def compute_silhouette(x, Z, y_bb, pred_class_idx):
        from sklearn.metrics import pairwise_distances

        mask_same = y_bb == pred_class_idx
        mask_diff = y_bb != pred_class_idx

        Z_plus = Z[mask_same]
        Z_minus = Z[mask_diff]

        a = pairwise_distances([x], Z_plus).mean() if len(Z_plus) > 0 else 0.0
        b = pairwise_distances([x], Z_minus).mean() if len(Z_minus) > 0 else 0.0

        if (a + b) == 0:
            return 0.0

        return (b - a) / max(a, b)

    @staticmethod
    def compute_rule_overlap(dfZ, rule_local, rule_global):
        def mask_regla(df, rule):
            m = np.zeros(len(df), dtype=bool)
            for i, row in enumerate(df.to_dict(orient="records")):
                m[i] = ReglaEvaluator.cumple_regla(row, rule)
            return m

        mL = mask_regla(dfZ, rule_local)
        mG = mask_regla(dfZ, rule_global)

        inter = np.logical_and(mL, mG).sum()
        union = np.logical_or(mL, mG).sum()

        jaccard = 0.0 if union == 0 else inter / union

        covL = mL.mean()
        covG = mG.mean()
        covInter = np.logical_and(mL, mG).mean()
        covUnion = np.logical_or(mL, mG).mean()

        return jaccard, covL, covG, covInter, covUnion

    @staticmethod
    def compute_single_rule_coverage(dfZ, rule):
        def mask_regla(df, rule):
            m = np.zeros(len(df), dtype=bool)
            for i, row in enumerate(df.to_dict(orient="records")):
                m[i] = ReglaEvaluator.cumple_regla(row, rule)
            return m

        if rule is None:
            return np.nan

        return float(mask_regla(dfZ, rule).mean())

    @staticmethod
    def predict_bbox(model, row, label_encoder):
        probs = model.predict_proba(row[None, :])
        pred_idx = int(probs.argmax(axis=1)[0])
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        return pred_idx, pred_label

    @staticmethod
    def build_tabular_dataset(X_train, y_train, feature_names, label_encoder):
        local_df = pd.DataFrame(X_train, columns=feature_names).astype(np.float32)
        local_df["class"] = label_encoder.inverse_transform(y_train)

        return TabularDataset(local_df, class_name="class")
