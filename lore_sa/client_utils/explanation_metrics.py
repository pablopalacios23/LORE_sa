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
            num_features=3
        )

        rules = [f for f, _ in exp.as_list()]
        rules = ReglaEvaluator.decode_lime_conditions(rules, row, feature_names)

        return list(dict.fromkeys(rules))
    


    @staticmethod
    def compute_anchor_rules(anchor_explainer, row, threshold, feature_names):

        anchor_exp = anchor_explainer.explain(row, threshold=threshold, delta=0.2, beam_size=3, n_samples=1000)

        rules = anchor_exp.anchor
        rules = ReglaEvaluator.decode_anchor_conditions(rules, row, feature_names)

        return list(dict.fromkeys(rules))
    


    @staticmethod
    def compute_silhouette(x, Z, y_bb, pred_class_idx):

        from sklearn.metrics import pairwise_distances

        mask_same = (y_bb == pred_class_idx)
        mask_diff = (y_bb != pred_class_idx)

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