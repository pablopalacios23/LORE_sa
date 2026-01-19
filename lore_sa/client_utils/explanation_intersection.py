# explanation_intersection.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import re
import numpy as np

Rule = List[str]
RulesByClass = Dict[str, List[Rule]]

@dataclass
class ExplanationIntersection:
    """
    Métricas por intersección de FEATURES entre explicaciones:
      - FFA_local_global
      - FCC_local
      - FCC_global
      - CMCA_local_global
    No necesita X.
    """
    # regex para extraer variable al inicio de condición:  Var op ...
    var_re: re.Pattern = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*([<>=]|≤|≥|≠)')

    def vars_in_rule(self, rule: Rule) -> set[str]:
        vars_ = set()
        for cond in rule:
            m = self.var_re.match(cond.strip())
            if m:
                vars_.add(m.group(1))
        return vars_

    def flatten_rules(self, rules_by_class: RulesByClass) -> List[Rule]:
        return [r for rules in rules_by_class.values() for r in rules]

    # ----------------------------
    # 1) FFA: factual local vs global
    # ----------------------------
    def FFA_local_global(self, factual_local: Rule, factual_global: Rule) -> float:
        vL = self.vars_in_rule(factual_local)
        vG = self.vars_in_rule(factual_global)
        if not vL:
            return 0.0
        return len(vL & vG) / len(vL)

    # ----------------------------
    # 2) FCC: factual vs CF (media)
    # ----------------------------
    def _FCC_one(self, factual: Rule, cf: Rule) -> float:
        vF = self.vars_in_rule(factual)
        vC = self.vars_in_rule(cf)
        if not vC:
            return 0.0
        return len(vF & vC) / len(vC)

    def FCC(self, factual: Rule, cf_rules_by_class: RulesByClass) -> float:
        cfs = self.flatten_rules(cf_rules_by_class)
        if not cfs:
            return 0.0
        vals = [self._FCC_one(factual, cf) for cf in cfs]
        return float(np.mean(vals))

    def FCC_local(self, factual_local: Rule, cf_local_by_class: RulesByClass) -> float:
        return self.FCC(factual_local, cf_local_by_class)

    def FCC_global(self, factual_global: Rule, cf_global_by_class: RulesByClass) -> float:
        return self.FCC(factual_global, cf_global_by_class)

    # ----------------------------
    # 3) CMCA: CF local vs CF global
    # ----------------------------
    def _align_one(self, cf: Rule, global_vars_list: List[set[str]]) -> float:
        vC = self.vars_in_rule(cf)
        if not vC:
            return 0.0
        best = 0.0
        for vG in global_vars_list:
            best = max(best, len(vC & vG) / len(vC))
        return best

    def CMCA_local_global(self, cf_local_by_class: RulesByClass, cf_global_by_class: RulesByClass) -> float:
        cfL = self.flatten_rules(cf_local_by_class)
        cfG = self.flatten_rules(cf_global_by_class)
        if not cfL or not cfG:
            return 0.0

        global_vars_list = [self.vars_in_rule(r) for r in cfG]
        vals = [self._align_one(cf, global_vars_list) for cf in cfL]
        return float(np.mean(vals))
