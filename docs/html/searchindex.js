Search.setIndex({"docnames": ["examples/tabular_explanations_example", "index", "source/generated/lore_sa.bbox.AbstractBBox", "source/generated/lore_sa.dataset.DataSet", "source/generated/lore_sa.decision_tree.is_leaf", "source/generated/lore_sa.decision_tree.learn_local_decision_tree", "source/generated/lore_sa.decision_tree.prune_duplicate_leaves", "source/generated/lore_sa.decision_tree.prune_index", "source/generated/lore_sa.discretizer.Discretizer", "source/generated/lore_sa.discretizer.RMEPDiscretizer", "source/generated/lore_sa.encoder_decoder.EncDec", "source/generated/lore_sa.encoder_decoder.MyTargetEnc", "source/generated/lore_sa.encoder_decoder.OneHotEnc", "source/generated/lore_sa.explanation.Explanation", "source/generated/lore_sa.explanation.ExplanationEncoder", "source/generated/lore_sa.explanation.ImageExplanation", "source/generated/lore_sa.explanation.MultilabelExplanation", "source/generated/lore_sa.explanation.TextExplanation", "source/generated/lore_sa.explanation.json2explanation", "source/generated/lore_sa.lorem.LOREM", "source/generated/lore_sa.neighgen.CFSGenerator", "source/generated/lore_sa.neighgen.ClosestInstancesGenerator", "source/generated/lore_sa.neighgen.CounterGenerator", "source/generated/lore_sa.neighgen.GeneticGenerator", "source/generated/lore_sa.neighgen.GeneticProbaGenerator", "source/generated/lore_sa.neighgen.NeighborhoodGenerator", "source/generated/lore_sa.neighgen.RandomGenerator", "source/generated/lore_sa.neighgen.RandomGeneticGenerator", "source/generated/lore_sa.neighgen.RandomGeneticProbaGenerator", "source/generated/lore_sa.rule.Condition", "source/generated/lore_sa.rule.Rule", "source/generated/lore_sa.surrogate.DecTree", "source/generated/lore_sa.surrogate.SuperTree", "source/generated/lore_sa.surrogate.Surrogate", "source/generated/lore_sa.util", "source/modules"], "filenames": ["examples\\tabular_explanations_example.rst", "index.rst", "source\\generated\\lore_sa.bbox.AbstractBBox.rst", "source\\generated\\lore_sa.dataset.DataSet.rst", "source\\generated\\lore_sa.decision_tree.is_leaf.rst", "source\\generated\\lore_sa.decision_tree.learn_local_decision_tree.rst", "source\\generated\\lore_sa.decision_tree.prune_duplicate_leaves.rst", "source\\generated\\lore_sa.decision_tree.prune_index.rst", "source\\generated\\lore_sa.discretizer.Discretizer.rst", "source\\generated\\lore_sa.discretizer.RMEPDiscretizer.rst", "source\\generated\\lore_sa.encoder_decoder.EncDec.rst", "source\\generated\\lore_sa.encoder_decoder.MyTargetEnc.rst", "source\\generated\\lore_sa.encoder_decoder.OneHotEnc.rst", "source\\generated\\lore_sa.explanation.Explanation.rst", "source\\generated\\lore_sa.explanation.ExplanationEncoder.rst", "source\\generated\\lore_sa.explanation.ImageExplanation.rst", "source\\generated\\lore_sa.explanation.MultilabelExplanation.rst", "source\\generated\\lore_sa.explanation.TextExplanation.rst", "source\\generated\\lore_sa.explanation.json2explanation.rst", "source\\generated\\lore_sa.lorem.LOREM.rst", "source\\generated\\lore_sa.neighgen.CFSGenerator.rst", "source\\generated\\lore_sa.neighgen.ClosestInstancesGenerator.rst", "source\\generated\\lore_sa.neighgen.CounterGenerator.rst", "source\\generated\\lore_sa.neighgen.GeneticGenerator.rst", "source\\generated\\lore_sa.neighgen.GeneticProbaGenerator.rst", "source\\generated\\lore_sa.neighgen.NeighborhoodGenerator.rst", "source\\generated\\lore_sa.neighgen.RandomGenerator.rst", "source\\generated\\lore_sa.neighgen.RandomGeneticGenerator.rst", "source\\generated\\lore_sa.neighgen.RandomGeneticProbaGenerator.rst", "source\\generated\\lore_sa.rule.Condition.rst", "source\\generated\\lore_sa.rule.Rule.rst", "source\\generated\\lore_sa.surrogate.DecTree.rst", "source\\generated\\lore_sa.surrogate.SuperTree.rst", "source\\generated\\lore_sa.surrogate.Surrogate.rst", "source\\generated\\lore_sa.util.rst", "source\\modules.rst"], "titles": ["Tabular explanations example", "lore_sa", "lore_sa.bbox.AbstractBBox", "lore_sa.dataset.DataSet", "lore_sa.decision_tree.is_leaf", "lore_sa.decision_tree.learn_local_decision_tree", "lore_sa.decision_tree.prune_duplicate_leaves", "lore_sa.decision_tree.prune_index", "lore_sa.discretizer.Discretizer", "lore_sa.discretizer.RMEPDiscretizer", "lore_sa.encoder_decoder.EncDec", "lore_sa.encoder_decoder.MyTargetEnc", "lore_sa.encoder_decoder.OneHotEnc", "lore_sa.explanation.Explanation", "lore_sa.explanation.ExplanationEncoder", "lore_sa.explanation.ImageExplanation", "lore_sa.explanation.MultilabelExplanation", "lore_sa.explanation.TextExplanation", "lore_sa.explanation.json2explanation", "lore_sa.lorem.LOREM", "lore_sa.neighgen.CFSGenerator", "lore_sa.neighgen.ClosestInstancesGenerator", "lore_sa.neighgen.CounterGenerator", "lore_sa.neighgen.GeneticGenerator", "lore_sa.neighgen.GeneticProbaGenerator", "lore_sa.neighgen.NeighborhoodGenerator", "lore_sa.neighgen.RandomGenerator", "lore_sa.neighgen.RandomGeneticGenerator", "lore_sa.neighgen.RandomGeneticProbaGenerator", "lore_sa.rule.Condition", "lore_sa.rule.Rule", "lore_sa.surrogate.DecTree", "lore_sa.surrogate.SuperTree", "lore_sa.surrogate.Surrogate", "lore_sa.util", "Modules"], "terms": {"import": [0, 14], "panda": 0, "pd": 0, "numpi": [0, 19], "np": 0, "from": [0, 3, 7, 11, 14, 20, 21, 22, 23, 24, 25, 26, 27, 28], "sklearn": [0, 2], "preprocess": [0, 31, 32, 33], "ensembl": 0, "randomforestclassifi": 0, "model_select": 0, "train_test_split": 0, "linear_model": 0, "logisticregress": 0, "xailib": 0, "data_load": 0, "dataframe_load": 0, "prepare_datafram": [0, 3], "lime_explain": 0, "limexaitabularexplain": 0, "lore_explain": 0, "loretabularexplain": 0, "shap_explainer_tab": 0, "shapxaitabularexplain": 0, "sklearn_classifier_wrapp": 0, "we": [0, 7, 22], "start": [0, 7, 20, 21, 22, 23, 24, 25, 26, 27, 28], "read": 0, "csv": [0, 3], "file": 0, "analyz": 0, "The": [0, 3, 14], "tabl": 0, "i": [0, 3, 4, 10, 11, 12, 14, 19, 22, 34], "mean": [0, 22], "datafram": [0, 3], "class": [0, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], "librari": 0, "among": 0, "all": [0, 3, 14], "attribut": [0, 3, 14], "select": [0, 12], "class_field": [0, 3], "column": [0, 3], "contain": [0, 3, 14], "observ": 0, "correspond": 0, "row": 0, "source_fil": 0, "german_credit": 0, "default": [0, 14, 19], "transform": [0, 3, 34], "df": [0, 3], "read_csv": 0, "skipinitialspac": 0, "true": [0, 14, 19, 20, 21, 22, 29], "na_valu": 0, "keep_default_na": 0, "after": [0, 3], "memori": 0, "need": [0, 19], "extract": [0, 3, 22], "metadata": 0, "inform": [0, 3, 14], "automat": 0, "handl": [0, 3, 10], "content": 0, "withint": 0, "method": [0, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], "scan": [0, 3], "follow": [0, 3], "trasform": [0, 3], "version": [0, 3, 14], "origin": [0, 3, 20], "where": [0, 3], "discret": [0, 3], "ar": [0, 3, 10, 14, 20], "numer": [0, 3], "us": [0, 3, 7, 14, 20, 21, 22, 23, 24, 25, 26, 27, 28], "one": [0, 3], "hot": [0, 3], "encod": [0, 3, 10, 11, 12, 14], "strategi": [0, 3], "feature_nam": [0, 3], "list": [0, 2, 3, 14, 19], "containint": [0, 3], "name": [0, 3, 11, 19], "featur": [0, 3, 22], "class_valu": [0, 3, 5], "possibl": [0, 3], "valu": [0, 3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 34], "numeric_column": [0, 3], "e": [0, 3, 22], "continu": [0, 3], "rdf": [0, 3], "befor": [0, 3], "real_feature_nam": [0, 3], "features_map": [0, 3, 20, 21, 22, 23, 24, 25, 26, 27, 28], "dictionari": [0, 3, 14, 19], "point": [0, 3, 20], "each": [0, 3, 10, 14, 20], "train": [0, 3, 11], "rf": 0, "classifi": 0, "split": 0, "test": [0, 2, 14], "subset": 0, "test_siz": 0, "0": [0, 7, 14, 20, 21, 22, 23, 24, 25, 26, 27, 28, 34], "3": [0, 19, 22, 23, 24, 27, 28], "random_st": 0, "42": 0, "x_train": 0, "x_test": 0, "y_train": 0, "y_test": 0, "stratifi": 0, "Then": 0, "set": 0, "onc": 0, "ha": 0, "been": 0, "wrapper": 0, "get": [0, 14], "access": 0, "xai": 0, "lib": 0, "bb": [0, 19], "n_estim": 0, "20": 0, "fit": [0, 12, 34], "bbox": 0, "new": [0, 20], "instanc": [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "classfi": 0, "print": [0, 14], "inst": 0, "iloc": 0, "147": 0, "8": 0, "reshap": 0, "1": [0, 2, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 34], "15": 0, "975": 0, "2": [0, 22, 23, 24, 27, 28], "25": 0, "provid": [0, 2, 3, 19], "an": [0, 3, 10, 14, 19], "explant": 0, "everi": [0, 22], "take": [0, 14, 22], "input": 0, "blackbox": [0, 19], "configur": 0, "object": [0, 14, 19], "initi": [0, 19], "config": [0, 19], "tree": 0, "100": [0, 19, 22, 23, 24, 27, 28], "exp": 0, "plot_features_import": 0, "neigh_typ": 0, "rndgen": 0, "size": [0, 20, 21, 22, 23, 24, 25, 26, 27, 28], "1000": [0, 20, 21, 22, 23, 24, 25, 26, 27, 28], "ocr": [0, 20, 21, 22, 23, 24, 25, 26, 27, 28], "ngen": [0, 22, 23, 24, 27, 28], "10": [0, 34], "plotrul": 0, "plotcounterfactualrul": 0, "limeexplain": 0, "feature_select": 0, "lasso_path": 0, "lime_exp": 0, "as_list": 0, "account_check_statu": 0, "check": [0, 4, 14], "account": 0, "03792512128083548": 0, "duration_in_month": 0, "03701527256562679": 0, "dm": 0, "03144299031649348": 0, "save": 0, "020051934530021572": 0, "ag": 0, "019751080001761446": 0, "credit_histori": 0, "critic": 0, "other": 0, "exist": 0, "thi": [0, 3, 7, 11, 14, 19], "bank": [0, 3], "018970043296280513": 0, "other_installment_plan": 0, "none": [0, 9, 10, 11, 12, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34], "018869997928840695": 0, "017658677626390982": 0, "hous": 0, "own": 0, "014948467979451343": 0, "delai": 0, "pai": 0, "off": 0, "past": 0, "012221985897781883": 0, "plot_lime_valu": 0, "5": [0, 5, 19, 20, 21, 22, 23, 24, 27, 28, 34], "regress": [0, 14], "scaler": 0, "normal": 0, "standardscal": 0, "x_scale": 0, "c": 0, "penalti": 0, "l2": 0, "pass": [0, 2], "record": [0, 10, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "182": 0, "27797454": 0, "35504085": 0, "94540357": 0, "07634233": 0, "04854891": 0, "72456474": 0, "43411405": 0, "65027399": 0, "61477862": 0, "25898489": 0, "80681063": 0, "4": [0, 20], "17385345": 0, "6435382": 0, "32533856": 0, "03489416": 0, "20412415": 0, "22941573": 0, "33068147": 0, "75885396": 0, "34899122": 0, "60155441": 0, "15294382": 0, "09298136": 0, "46852129": 0, "12038585": 0, "08481889": 0, "23623492": 0, "21387736": 0, "36174054": 0, "24943031": 0, "15526362": 0, "59715086": 0, "45485883": 0, "73610476": 0, "43875307": 0, "23307441": 0, "65242771": 0, "23958675": 0, "90192655": 0, "72581563": 0, "2259448": 0, "15238005": 0, "54212562": 0, "70181003": 0, "63024248": 0, "30354212": 0, "40586384": 0, "49329429": 0, "88675135": 0, "59227935": 0, "46170508": 0, "46388049": 0, "33747696": 0, "13206764": 0, "same": 0, "previou": 0, "In": 0, "case": 0, "few": 0, "adjust": 0, "necessari": 0, "For": [0, 14, 22], "specif": [0, 14], "linear": 0, "feature_pert": 0, "intervent": 0, "shapxaitabularexplan": 0, "0x12a72dac8": 0, "geneticp": 0, "loretabularexplan": 0, "0x12bc41a90": 0, "why": 0, "becaus": 0, "condit": 0, "happen": 0, "726173400878906credit": 0, "amount": 0, "439": 0, "6443485021591purpos": 0, "retrain": 0, "11524588242173195durat": 0, "month": 0, "9407005310058594purpos": 0, "furnitur": 0, "equip": 0, "18370826542377472foreign": 0, "worker": 0, "7168410122394562purpos": 0, "domest": 0, "applianc": 0, "015466570854187save": 0, "7176859378814697purpos": 0, "vacat": 0, "doe": 0, "4622504562139511credit": 0, "histori": 0, "9085964262485504": 0, "It": [0, 10, 14], "would": [0, 14], "have": [0, 20], "hold": 0, "6443485021591": 0, "26": 0, "468921303749084durat": 0, "795059680938721instal": 0, "incom": [0, 14], "perc": 0, "603440999984741": 0, "other_debtor": 0, "co": 0, "applic": 0, "3046177878918616e": 0, "09": 0, "paid": 0, "back": 0, "duli": 0, "0114574629252053e": 0, "present_emp_sinc": 0, "unemploi": 0, "87554096296626e": 0, "7": 0, "43754044231906e": 0, "free": 0, "4157786564097103e": 0, "properti": 0, "unknown": 0, "275710719845092e": 0, "credit_amount": 0, "271233788564153e": 0, "job": 0, "manag": 0, "self": [0, 11, 12], "emploi": [0, 19], "highli": 0, "qualifi": 0, "employe": 0, "offic": 0, "164190703926506e": 0, "8902027822084106e": 0, "604277452741881e": 0, "skill": 0, "offici": 0, "3808188198617575e": 0, "foreign_work": 0, "ye": 0, "365347360238489e": 0, "telephon": 0, "2048259721367863e": 0, "171945479826713e": 0, "1116662177987812e": 0, "credits_this_bank": 0, "9999632029038067e": 0, "till": 0, "now": 0, "9243622007776865e": 0, "people_under_mainten": 0, "902008911572941e": 0, "purpos": 0, "car": 0, "7104663723358493e": 0, "6584313433238958e": 0, "200": [0, 34], "639544710042764e": 0, "317487567892989e": 0, "unskil": 0, "resid": 0, "307761159896724e": 0, "store": 0, "2347569776391545e": 0, "1825353902253505e": 0, "year": 0, "1478921168922655e": 0, "a121": 0, "a122": 0, "6": 0, "1222769011436428e": 0, "personal_status_sex": 0, "femal": 0, "divorc": 0, "separ": [0, 14], "marri": 0, "1002871894681165e": 0, "500": [0, 20], "0982251402773794e": 0, "0567984890752028e": 0, "present_res_sinc": 0, "9": 0, "869484730455045e": 0, "11": 0, "salari": 0, "assign": 0, "least": 0, "721716212812873e": 0, "327030468700815e": 0, "installment_as_income_perc": 0, "192261925231111e": 0, "real": 0, "estat": 0, "180043418264463e": 0, "974505020571898e": 0, "848004118893571e": 0, "80910843922895e": 0, "educ": 0, "803520453193465e": 0, "busi": 0, "330599059469541e": 0, "rent": 0, "975475868460632e": 0, "build": 0, "societi": 0, "agreement": 0, "life": 0, "insur": 0, "826524390749874e": 0, "guarantor": 0, "385760952840171e": 0, "338094381227495e": 0, "689756440260244e": 0, "582965568284186e": 0, "non": [0, 14], "473736018584135e": 0, "230002403518189e": 0, "974714318917145e": 0, "radio": 0, "televis": 0, "909852887925919e": 0, "620862803354922e": 0, "582941358078461e": 0, "501318386790144e": 0, "male": 0, "widow": 0, "500125372750834e": 0, "regist": 0, "under": 0, "custom": [0, 14], "495252929908006e": 0, "repair": 0, "2177896575440796e": 0, "0557757647139625e": 0, "627184253632623e": 0, "singl": [0, 19], "9862189862658355e": 0, "taken": 0, "8131802175589855e": 0, "9548368945624186e": 0, "modul": 1, "exampl": [1, 14], "tabular": [1, 3], "explan": [1, 19], "learn": [], "explain": 19, "german": [], "credit": 3, "dataset": [10, 11, 12, 19], "index": [1, 4, 7], "search": 1, "page": 1, "sourc": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34], "gener": [2, 10, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 33], "black": [2, 19], "box": [2, 19], "witch": 2, "two": 2, "like": 2, "__init__": [2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], "abstract": [2, 25], "predict": 2, "sample_matrix": 2, "wrap": 2, "label": 2, "data": [2, 3, 14, 19, 20, 34], "paramet": [2, 3, 11, 12, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 34], "arrai": [2, 14, 19, 20], "spars": 2, "matrix": 2, "shape": [2, 20, 34], "n_queri": 2, "n_featur": 2, "sampl": [2, 19, 20], "return": [2, 3, 11, 12, 14, 19, 20, 22, 34], "ndarrai": 2, "n_class": 2, "n_output": 2, "probabl": [2, 20], "order": [], "lexicograph": [], "predict_proba": 2, "estim": 2, "filenam": 3, "str": [3, 14], "class_nam": [3, 10, 11, 12, 19, 30], "encdec": [11, 12, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "interfac": 3, "imag": 3, "etc": 3, "incapsul": [3, 19], "expos": 3, "prepar": 3, "prepare_bank_dataset": 3, "http": 3, "www": 3, "kaggl": 3, "com": 3, "aniruddhachoudhuri": 3, "risk": 3, "model": [3, 34], "home": 3, "riccardo": 3, "scaricati": 3, "param": [3, 14], "prepare_dataset": 3, "inner_tre": [4, 7], "whether": 4, "node": [4, 7], "leaf": 4, "z": 5, "yb": 5, "weight": 5, "multi_label": 5, "fals": [5, 14, 19, 20, 21, 22, 23, 24, 27, 28], "one_vs_rest": 5, "cv": 5, "prune_tre": [5, 19], "dt": 6, "remov": 6, "leav": [6, 7], "both": 6, "decis": 7, "prune": 7, "bottom": 7, "top": 7, "might": 7, "miss": 7, "becom": 7, "dure": [7, 14, 19], "do": 7, "directli": 7, "prune_duplicate_leav": 7, "instead": 7, "to_discret": 9, "proto_fn": 9, "implement": 10, "decod": [10, 14], "differ": 10, "which": [10, 14, 19], "must": 10, "function": [10, 11, 14, 19, 21, 22, 23, 24, 27, 28, 34], "enc": 10, "dec": 10, "enc_fit_transform": [10, 11, 12], "idea": 10, "user": 10, "send": 10, "complet": 10, "here": [10, 22], "onli": [10, 14], "categor": [10, 11, 12], "variabl": [10, 11, 12], "extend": 11, "targetencod": 11, "category_encod": 11, "given": [11, 14], "appli": [11, 12], "target": 11, "dataset_enc": [11, 12], "kwarg": [11, 12, 19, 20], "onehot": 12, "them": 12, "alreadi": [12, 22], "skipkei": 14, "ensure_ascii": 14, "check_circular": 14, "allow_nan": 14, "sort_kei": 14, "indent": 14, "special": 14, "json": 14, "rule": [14, 19], "type": [14, 20], "constructor": 14, "jsonencod": 14, "sensibl": 14, "If": 14, "typeerror": 14, "attempt": 14, "kei": 14, "int": [14, 20, 21, 22, 23, 24, 25, 26, 27, 28], "float": [14, 20], "item": 14, "simpli": 14, "skip": 14, "output": 14, "guarante": 14, "ascii": 14, "charact": 14, "escap": 14, "can": 14, "dict": 14, "circular": 14, "refer": 14, "prevent": 14, "infinit": 14, "recurs": 14, "caus": 14, "overflowerror": 14, "otherwis": 14, "place": 14, "nan": 14, "infin": 14, "behavior": 14, "compliant": 14, "consist": 14, "most": 14, "javascript": 14, "base": [14, 19, 22], "valueerror": 14, "sort": 14, "ensur": 14, "serial": 14, "compar": 14, "dai": 14, "basi": 14, "neg": 14, "integ": 14, "element": 14, "member": 14, "pretti": 14, "level": 14, "insert": 14, "newlin": 14, "compact": 14, "represent": 14, "specifi": 14, "should": 14, "item_separ": 14, "key_separ": 14, "tupl": 14, "To": 14, "you": 14, "elimin": 14, "whitespac": 14, "call": 14, "t": 14, "rais": 14, "obj": [14, 18], "report": 14, "about": 14, "objgect": 14, "o": 14, "string": [14, 19], "python": 14, "structur": 14, "foo": 14, "bar": 14, "baz": 14, "iterencod": 14, "_one_shot": 14, "yield": 14, "avail": 14, "chunk": 14, "bigobject": 14, "mysocket": 14, "write": 14, "img": 15, "segment": 15, "text": 17, "indexed_text": 17, "abstractbbox": 19, "local": 19, "defin": 22, "option": [], "unadmittible_featur": [], "neigh_gen": 19, "filter_crul": [], "binari": [], "extreme_fidel": [], "verbos": [20, 21, 22, 23, 24, 27, 28], "datamanag": 19, "explain_inst": [], "x": [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 34], "use_weight": 19, "metric": [19, 22, 23, 24, 27, 28], "neuclidean": [19, 21, 22, 23, 24, 27, 28], "run": 19, "exemplar_num": 19, "n_job": 19, "express": [], "number": [19, 20], "neighbourhood": [19, 22], "measur": 19, "distanc": 19, "between": 19, "time": 19, "done": 19, "examplar": 19, "retriev": 19, "add": 19, "cf": 19, "get_config": [], "get_neighborhood_gener": [], "factori": [], "neighborhood": [20, 21, 22, 23, 24, 25, 26, 27, 28], "closest_inst": [], "neighborhoodgener": 19, "bb_predict": [20, 21, 22, 23, 24, 25, 26, 27, 28], "feature_valu": [20, 21, 22, 23, 24, 25, 26, 27, 28], "nbr_featur": [20, 21, 22, 23, 24, 25, 26, 27, 28], "nbr_real_featur": [20, 21, 22, 23, 24, 25, 26, 27, 28], "numeric_columns_index": [20, 21, 22, 23, 24, 25, 26, 27, 28], "n_search": 20, "10000": 20, "n_batch": 20, "lower_threshold": 20, "upper_threshold": 20, "kind": [20, 31, 32, 33], "gaussian_match": 20, "sampling_kind": 20, "stopping_ratio": 20, "01": 20, "check_upper_threshold": 20, "final_counterfactual_search": 20, "custom_sampling_threshold": 20, "custom_closest_counterfactu": 20, "n": 20, "balanc": 20, "cut_radiu": 20, "forced_balance_ratio": 20, "downward_onli": 20, "num_sampl": [20, 21, 22, 23, 24, 25, 26, 27, 28], "synthet": [20, 21, 22, 23, 24, 25, 26, 27, 28], "orgin": [20, 21, 22, 23, 24, 25, 26, 27, 28], "ani": [20, 21, 22, 23, 24, 25, 26, 27, 28], "seed": [20, 21, 22, 23, 24, 25, 26, 27, 28], "translat": 20, "center": 20, "axi": 20, "uniform_sphere_origin": 20, "d": 20, "r": 20, "num_point": 20, "random": 20, "dimens": 20, "uniform": 20, "over": 20, "unit": 20, "ball": 20, "scale": 20, "radiu": 20, "length": 20, "rang": 20, "dimension": 20, "sphere": 20, "k": [21, 34], "rk": 21, "core_neigh_typ": 21, "unifi": 21, "alphaf": 21, "alphal": 21, "metric_featur": 21, "metric_label": 21, "ham": 21, "categorical_use_prob": 21, "continuous_fun_estim": 21, "bb_predict_proba": [22, 24, 25, 28], "original_data": [22, 25], "alpha1": [22, 23, 24, 27, 28], "alpha2": [22, 23, 24, 27, 28], "mutpb": [22, 23, 24, 27, 28], "random_se": [22, 23, 24, 27, 28], "cxpb": [22, 23, 24, 27, 28], "tournsiz": [22, 23, 24, 27, 28], "halloffame_ratio": [22, 23, 24, 27, 28], "closest": 22, "max_count": 22, "counterfactu": 22, "code": 22, "henc": 22, "latent": 22, "space": 22, "create_bin": 22, "bin": [22, 34], "feature_bin": 22, "find_closest_count": 22, "counter_list": 22, "inserisco": 22, "un": 22, "per": 22, "ogni": 22, "combinazion": 22, "di": 22, "il": 22, "maggiorment": 22, "vicino": 22, "clost": 22, "ho": 22, "con": [22, 30], "le": 22, "che": 22, "sono": 22, "state": 22, "cambiat": 22, "quando": 22, "trovato": 22, "cerco": 22, "piu": 22, "generando": 22, "caso": 22, "tra": 22, "l": [22, 34], "original": 22, "nuovo": 22, "premis": 30, "best_fit_distribut": 34, "ax": 34, "find": 34, "best": 34, "distribut": 34, "sigmoid": 34, "x0": 34, "A": 34, "logist": 34, "curv": 34, "common": 34, "": 34, "midpoint": 34, "maximum": 34, "steep": 34, "counter": [], "genet": [], "genetic_proba": [], "random_genet": [], "random_genetic_proba": [], "cfsgener": [], "closestinstancesgener": [], "countergener": [], "geneticgener": [], "geneticprobagener": [], "randomgener": [], "randomgeneticgener": [], "randomgeneticprobagener": [], "_target_": [], "get_encoder_decod": [], "encdec_typ": [], "onehotenc": [], "mytargetenc": [], "surrog": 19, "explain_instance_st": 19, "stabl": 19, "att": 29, "op": 29, "thr": 29, "is_continu": 29}, "objects": {"lore_sa.bbox": [[2, 0, 1, "", "AbstractBBox"]], "lore_sa.bbox.AbstractBBox": [[2, 1, 1, "", "__init__"], [2, 1, 1, "", "predict"], [2, 1, 1, "", "predict_proba"]], "lore_sa.dataset": [[3, 0, 1, "", "DataSet"]], "lore_sa.dataset.DataSet": [[3, 1, 1, "", "__init__"], [3, 1, 1, "", "prepare_bank_dataset"], [3, 1, 1, "", "prepare_dataset"]], "lore_sa.decision_tree": [[4, 2, 1, "", "is_leaf"], [5, 2, 1, "", "learn_local_decision_tree"], [6, 2, 1, "", "prune_duplicate_leaves"], [7, 2, 1, "", "prune_index"]], "lore_sa.discretizer": [[8, 0, 1, "", "Discretizer"], [9, 0, 1, "", "RMEPDiscretizer"]], "lore_sa.discretizer.Discretizer": [[8, 1, 1, "", "__init__"]], "lore_sa.discretizer.RMEPDiscretizer": [[9, 1, 1, "", "__init__"]], "lore_sa.encoder_decoder": [[10, 0, 1, "", "EncDec"], [11, 0, 1, "", "MyTargetEnc"], [12, 0, 1, "", "OneHotEnc"]], "lore_sa.encoder_decoder.EncDec": [[10, 1, 1, "", "__init__"]], "lore_sa.encoder_decoder.MyTargetEnc": [[11, 1, 1, "", "__init__"], [11, 1, 1, "", "enc_fit_transform"]], "lore_sa.encoder_decoder.OneHotEnc": [[12, 1, 1, "", "__init__"], [12, 1, 1, "", "enc_fit_transform"]], "lore_sa.explanation": [[13, 0, 1, "", "Explanation"], [14, 0, 1, "", "ExplanationEncoder"], [15, 0, 1, "", "ImageExplanation"], [16, 0, 1, "", "MultilabelExplanation"], [17, 0, 1, "", "TextExplanation"], [18, 2, 1, "", "json2explanation"]], "lore_sa.explanation.Explanation": [[13, 1, 1, "", "__init__"]], "lore_sa.explanation.ExplanationEncoder": [[14, 1, 1, "", "__init__"], [14, 1, 1, "", "default"], [14, 1, 1, "", "encode"], [14, 1, 1, "", "iterencode"]], "lore_sa.explanation.ImageExplanation": [[15, 1, 1, "", "__init__"]], "lore_sa.explanation.MultilabelExplanation": [[16, 1, 1, "", "__init__"]], "lore_sa.explanation.TextExplanation": [[17, 1, 1, "", "__init__"]], "lore_sa.lorem": [[19, 0, 1, "", "LOREM"]], "lore_sa.lorem.LOREM": [[19, 1, 1, "", "__init__"], [19, 1, 1, "", "explain_instance_stable"]], "lore_sa.neighgen": [[20, 0, 1, "", "CFSGenerator"], [21, 0, 1, "", "ClosestInstancesGenerator"], [22, 0, 1, "", "CounterGenerator"], [23, 0, 1, "", "GeneticGenerator"], [24, 0, 1, "", "GeneticProbaGenerator"], [25, 0, 1, "", "NeighborhoodGenerator"], [26, 0, 1, "", "RandomGenerator"], [27, 0, 1, "", "RandomGeneticGenerator"], [28, 0, 1, "", "RandomGeneticProbaGenerator"]], "lore_sa.neighgen.CFSGenerator": [[20, 1, 1, "", "__init__"], [20, 1, 1, "", "generate"], [20, 1, 1, "", "translate"], [20, 1, 1, "", "uniform_sphere_origin"]], "lore_sa.neighgen.ClosestInstancesGenerator": [[21, 1, 1, "", "__init__"], [21, 1, 1, "", "generate"]], "lore_sa.neighgen.CounterGenerator": [[22, 1, 1, "", "__init__"], [22, 1, 1, "", "create_bins"], [22, 1, 1, "", "find_closest_counter"], [22, 1, 1, "", "generate"]], "lore_sa.neighgen.GeneticGenerator": [[23, 1, 1, "", "__init__"], [23, 1, 1, "", "generate"]], "lore_sa.neighgen.GeneticProbaGenerator": [[24, 1, 1, "", "__init__"], [24, 1, 1, "", "generate"]], "lore_sa.neighgen.NeighborhoodGenerator": [[25, 1, 1, "", "__init__"], [25, 1, 1, "", "generate"]], "lore_sa.neighgen.RandomGenerator": [[26, 1, 1, "", "__init__"], [26, 1, 1, "", "generate"]], "lore_sa.neighgen.RandomGeneticGenerator": [[27, 1, 1, "", "__init__"], [27, 1, 1, "", "generate"]], "lore_sa.neighgen.RandomGeneticProbaGenerator": [[28, 1, 1, "", "__init__"], [28, 1, 1, "", "generate"]], "lore_sa.rule": [[29, 0, 1, "", "Condition"], [30, 0, 1, "", "Rule"]], "lore_sa.rule.Condition": [[29, 1, 1, "", "__init__"]], "lore_sa.rule.Rule": [[30, 1, 1, "", "__init__"]], "lore_sa.surrogate": [[31, 0, 1, "", "DecTree"], [32, 0, 1, "", "SuperTree"], [33, 0, 1, "", "Surrogate"]], "lore_sa.surrogate.DecTree": [[31, 1, 1, "", "__init__"]], "lore_sa.surrogate.SuperTree": [[32, 1, 1, "", "__init__"]], "lore_sa.surrogate.Surrogate": [[33, 1, 1, "", "__init__"]], "lore_sa": [[34, 3, 0, "-", "util"]], "lore_sa.util": [[34, 2, 1, "", "best_fit_distribution"], [34, 2, 1, "", "sigmoid"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:function", "3": "py:module"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "function", "Python function"], "3": ["py", "module", "Python module"]}, "titleterms": {"tabular": 0, "explan": [0, 13, 14, 15, 16, 17, 18, 35], "exampl": 0, "learn": 0, "explain": 0, "german": 0, "credit": 0, "dataset": [0, 3, 35], "load": 0, "prepar": 0, "data": 0, "random": 0, "forest": 0, "classfier": 0, "predict": 0, "shap": 0, "lore": 0, "lime": 0, "differ": 0, "model": 0, "logist": 0, "regressor": 0, "lore_sa": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "indic": 1, "tabl": 1, "bbox": [2, 35], "abstractbbox": 2, "decision_tre": [4, 5, 6, 7, 35], "is_leaf": 4, "learn_local_decision_tre": 5, "prune_duplicate_leav": 6, "prune_index": 7, "discret": [8, 9, 35], "rmepdiscret": 9, "encoder_decod": [10, 11, 12, 35], "encdec": 10, "mytargetenc": 11, "onehotenc": 12, "explanationencod": 14, "imageexplan": 15, "multilabelexplan": 16, "textexplan": 17, "json2explan": 18, "lorem": [19, 35], "neighgen": [20, 21, 22, 23, 24, 25, 26, 27, 28, 35], "cfsgener": 20, "closestinstancesgener": 21, "countergener": 22, "geneticgener": 23, "geneticprobagener": 24, "neighborhoodgener": 25, "randomgener": 26, "randomgeneticgener": 27, "randomgeneticprobagener": 28, "rule": [29, 30, 35], "surrog": [31, 32, 33, 35], "dectre": 31, "supertre": 32, "util": [34, 35], "modul": 35, "class": 35, "blackbox": 35, "abstract": 35, "neighborhood": 35, "gener": 35, "decis": 35, "tree": 35, "function": 35, "encod": 35, "decod": 35, "condit": 29}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx": 57}, "alltitles": {"lore_sa": [[1, "lore-sa"]], "Indices and tables": [[1, "indices-and-tables"]], "lore_sa.decision_tree.is_leaf": [[4, "lore-sa-decision-tree-is-leaf"]], "lore_sa.decision_tree.learn_local_decision_tree": [[5, "lore-sa-decision-tree-learn-local-decision-tree"]], "lore_sa.decision_tree.prune_duplicate_leaves": [[6, "lore-sa-decision-tree-prune-duplicate-leaves"]], "lore_sa.decision_tree.prune_index": [[7, "lore-sa-decision-tree-prune-index"]], "lore_sa.discretizer.Discretizer": [[8, "lore-sa-discretizer-discretizer"]], "lore_sa.discretizer.RMEPDiscretizer": [[9, "lore-sa-discretizer-rmepdiscretizer"]], "lore_sa.explanation.Explanation": [[13, "lore-sa-explanation-explanation"]], "lore_sa.explanation.ExplanationEncoder": [[14, "lore-sa-explanation-explanationencoder"]], "lore_sa.explanation.ImageExplanation": [[15, "lore-sa-explanation-imageexplanation"]], "lore_sa.explanation.MultilabelExplanation": [[16, "lore-sa-explanation-multilabelexplanation"]], "lore_sa.explanation.TextExplanation": [[17, "lore-sa-explanation-textexplanation"]], "lore_sa.explanation.json2explanation": [[18, "lore-sa-explanation-json2explanation"]], "lore_sa.neighgen.CFSGenerator": [[20, "lore-sa-neighgen-cfsgenerator"]], "lore_sa.neighgen.ClosestInstancesGenerator": [[21, "lore-sa-neighgen-closestinstancesgenerator"]], "lore_sa.neighgen.CounterGenerator": [[22, "lore-sa-neighgen-countergenerator"]], "lore_sa.neighgen.GeneticGenerator": [[23, "lore-sa-neighgen-geneticgenerator"]], "lore_sa.neighgen.GeneticProbaGenerator": [[24, "lore-sa-neighgen-geneticprobagenerator"]], "lore_sa.neighgen.NeighborhoodGenerator": [[25, "lore-sa-neighgen-neighborhoodgenerator"]], "lore_sa.neighgen.RandomGenerator": [[26, "lore-sa-neighgen-randomgenerator"]], "lore_sa.neighgen.RandomGeneticGenerator": [[27, "lore-sa-neighgen-randomgeneticgenerator"]], "lore_sa.neighgen.RandomGeneticProbaGenerator": [[28, "lore-sa-neighgen-randomgeneticprobagenerator"]], "lore_sa.surrogate.DecTree": [[31, "lore-sa-surrogate-dectree"]], "lore_sa.surrogate.SuperTree": [[32, "lore-sa-surrogate-supertree"]], "lore_sa.surrogate.Surrogate": [[33, "lore-sa-surrogate-surrogate"]], "Tabular explanations example": [[0, "tabular-explanations-example"]], "Learning and explaining German Credit Dataset": [[0, "learning-and-explaining-german-credit-dataset"]], "Loading and preparation of data": [[0, "loading-and-preparation-of-data"]], "Learning a Random Forest classfier": [[0, "learning-a-random-forest-classfier"]], "Explaining the prediction": [[0, "explaining-the-prediction"], [0, "id1"]], "SHAP explainer": [[0, "shap-explainer"]], "LORE explainer": [[0, "lore-explainer"], [0, "id2"]], "LIME explainer": [[0, "lime-explainer"], [0, "id3"]], "Learning a different model": [[0, "learning-a-different-model"]], "Learning a Logistic Regressor": [[0, "learning-a-logistic-regressor"]], "lore_sa.bbox.AbstractBBox": [[2, "lore-sa-bbox-abstractbbox"]], "lore_sa.dataset.DataSet": [[3, "lore-sa-dataset-dataset"]], "lore_sa.encoder_decoder.EncDec": [[10, "lore-sa-encoder-decoder-encdec"]], "lore_sa.encoder_decoder.MyTargetEnc": [[11, "lore-sa-encoder-decoder-mytargetenc"]], "lore_sa.encoder_decoder.OneHotEnc": [[12, "lore-sa-encoder-decoder-onehotenc"]], "lore_sa.lorem.LOREM": [[19, "lore-sa-lorem-lorem"]], "lore_sa.rule.Condition": [[29, "lore-sa-rule-condition"]], "lore_sa.rule.Rule": [[30, "lore-sa-rule-rule"]], "lore_sa.util": [[34, "module-lore_sa.util"]], "Modules": [[35, "modules"]], "lore_sa.lorem: LOREM class": [[35, "lore-sa-lorem-lorem-class"]], "lore_sa.bbox: BlackBox abstract class": [[35, "lore-sa-bbox-blackbox-abstract-class"]], "lore_sa.dataset: Dataset class": [[35, "lore-sa-dataset-dataset-class"]], "lore_sa.neighgen: Neighborhood Generator classes": [[35, "lore-sa-neighgen-neighborhood-generator-classes"]], "lore_sa.decision_tree: Decision tree functions": [[35, "lore-sa-decision-tree-decision-tree-functions"]], "lore_sa.discretizer: Discretizer classes and functions": [[35, "lore-sa-discretizer-discretizer-classes-and-functions"]], "lore_sa.encoder_decoder: Encoder/Decoder classes and functions": [[35, "lore-sa-encoder-decoder-encoder-decoder-classes-and-functions"]], "lore_sa.explanation: Explanation classes and functions": [[35, "lore-sa-explanation-explanation-classes-and-functions"]], "lore_sa.rule: Rule classes and functions": [[35, "lore-sa-rule-rule-classes-and-functions"]], "lore_sa.surrogate: Surrogate classes and functions": [[35, "lore-sa-surrogate-surrogate-classes-and-functions"]], "lore_sa.util: Util functions": [[35, "lore-sa-util-util-functions"]]}, "indexentries": {"abstractbbox (class in lore_sa.bbox)": [[2, "lore_sa.bbox.AbstractBBox"]], "__init__() (lore_sa.bbox.abstractbbox method)": [[2, "lore_sa.bbox.AbstractBBox.__init__"]], "predict() (lore_sa.bbox.abstractbbox method)": [[2, "lore_sa.bbox.AbstractBBox.predict"]], "predict_proba() (lore_sa.bbox.abstractbbox method)": [[2, "lore_sa.bbox.AbstractBBox.predict_proba"]], "dataset (class in lore_sa.dataset)": [[3, "lore_sa.dataset.DataSet"]], "__init__() (lore_sa.dataset.dataset method)": [[3, "lore_sa.dataset.DataSet.__init__"]], "prepare_bank_dataset() (lore_sa.dataset.dataset method)": [[3, "lore_sa.dataset.DataSet.prepare_bank_dataset"]], "prepare_dataset() (lore_sa.dataset.dataset method)": [[3, "lore_sa.dataset.DataSet.prepare_dataset"]], "encdec (class in lore_sa.encoder_decoder)": [[10, "lore_sa.encoder_decoder.EncDec"]], "__init__() (lore_sa.encoder_decoder.encdec method)": [[10, "lore_sa.encoder_decoder.EncDec.__init__"]], "mytargetenc (class in lore_sa.encoder_decoder)": [[11, "lore_sa.encoder_decoder.MyTargetEnc"]], "__init__() (lore_sa.encoder_decoder.mytargetenc method)": [[11, "lore_sa.encoder_decoder.MyTargetEnc.__init__"]], "enc_fit_transform() (lore_sa.encoder_decoder.mytargetenc method)": [[11, "lore_sa.encoder_decoder.MyTargetEnc.enc_fit_transform"]], "onehotenc (class in lore_sa.encoder_decoder)": [[12, "lore_sa.encoder_decoder.OneHotEnc"]], "__init__() (lore_sa.encoder_decoder.onehotenc method)": [[12, "lore_sa.encoder_decoder.OneHotEnc.__init__"]], "enc_fit_transform() (lore_sa.encoder_decoder.onehotenc method)": [[12, "lore_sa.encoder_decoder.OneHotEnc.enc_fit_transform"]], "lorem (class in lore_sa.lorem)": [[19, "lore_sa.lorem.LOREM"]], "__init__() (lore_sa.lorem.lorem method)": [[19, "lore_sa.lorem.LOREM.__init__"]], "explain_instance_stable() (lore_sa.lorem.lorem method)": [[19, "lore_sa.lorem.LOREM.explain_instance_stable"]], "condition (class in lore_sa.rule)": [[29, "lore_sa.rule.Condition"]], "__init__() (lore_sa.rule.condition method)": [[29, "lore_sa.rule.Condition.__init__"]], "rule (class in lore_sa.rule)": [[30, "lore_sa.rule.Rule"]], "__init__() (lore_sa.rule.rule method)": [[30, "lore_sa.rule.Rule.__init__"]], "best_fit_distribution() (in module lore_sa.util)": [[34, "lore_sa.util.best_fit_distribution"]], "lore_sa.util": [[34, "module-lore_sa.util"]], "module": [[34, "module-lore_sa.util"]], "sigmoid() (in module lore_sa.util)": [[34, "lore_sa.util.sigmoid"]]}})