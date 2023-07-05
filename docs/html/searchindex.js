Search.setIndex({"docnames": ["examples/tabular_explanations_example", "index", "source/generated/lore_sa.bbox.AbstractBBox", "source/generated/lore_sa.dataset.Dataset", "source/generated/lore_sa.dataset.TabularDataset", "source/generated/lore_sa.dataset.utils", "source/generated/lore_sa.discretizer.Discretizer", "source/generated/lore_sa.discretizer.RMEPDiscretizer", "source/generated/lore_sa.encoder_decoder.EncDec", "source/generated/lore_sa.encoder_decoder.LabelEnc", "source/generated/lore_sa.encoder_decoder.OneHotEnc", "source/generated/lore_sa.explanation.Explanation", "source/generated/lore_sa.explanation.ExplanationEncoder", "source/generated/lore_sa.explanation.ImageExplanation", "source/generated/lore_sa.explanation.MultilabelExplanation", "source/generated/lore_sa.explanation.TextExplanation", "source/generated/lore_sa.explanation.json2explanation", "source/generated/lore_sa.lorem.LOREM", "source/generated/lore_sa.neighgen.RandomGenerator", "source/generated/lore_sa.rule.Rule", "source/generated/lore_sa.rule.RuleGetter", "source/generated/lore_sa.rule.RuleGetterBinary", "source/generated/lore_sa.rule.rule.Condition", "source/generated/lore_sa.rule.rule.ConditionEncoder", "source/generated/lore_sa.rule.rule.NumpyEncoder", "source/generated/lore_sa.rule.rule.RuleEncoder", "source/generated/lore_sa.rule.rule.json2cond", "source/generated/lore_sa.rule.rule.json2rule", "source/generated/lore_sa.surrogate.DecisionTreeSurrogate", "source/generated/lore_sa.surrogate.Surrogate", "source/generated/lore_sa.util", "source/modules"], "filenames": ["examples\\tabular_explanations_example.rst", "index.rst", "source\\generated\\lore_sa.bbox.AbstractBBox.rst", "source\\generated\\lore_sa.dataset.Dataset.rst", "source\\generated\\lore_sa.dataset.TabularDataset.rst", "source\\generated\\lore_sa.dataset.utils.rst", "source\\generated\\lore_sa.discretizer.Discretizer.rst", "source\\generated\\lore_sa.discretizer.RMEPDiscretizer.rst", "source\\generated\\lore_sa.encoder_decoder.EncDec.rst", "source\\generated\\lore_sa.encoder_decoder.LabelEnc.rst", "source\\generated\\lore_sa.encoder_decoder.OneHotEnc.rst", "source\\generated\\lore_sa.explanation.Explanation.rst", "source\\generated\\lore_sa.explanation.ExplanationEncoder.rst", "source\\generated\\lore_sa.explanation.ImageExplanation.rst", "source\\generated\\lore_sa.explanation.MultilabelExplanation.rst", "source\\generated\\lore_sa.explanation.TextExplanation.rst", "source\\generated\\lore_sa.explanation.json2explanation.rst", "source\\generated\\lore_sa.lorem.LOREM.rst", "source\\generated\\lore_sa.neighgen.RandomGenerator.rst", "source\\generated\\lore_sa.rule.Rule.rst", "source\\generated\\lore_sa.rule.RuleGetter.rst", "source\\generated\\lore_sa.rule.RuleGetterBinary.rst", "source\\generated\\lore_sa.rule.rule.Condition.rst", "source\\generated\\lore_sa.rule.rule.ConditionEncoder.rst", "source\\generated\\lore_sa.rule.rule.NumpyEncoder.rst", "source\\generated\\lore_sa.rule.rule.RuleEncoder.rst", "source\\generated\\lore_sa.rule.rule.json2cond.rst", "source\\generated\\lore_sa.rule.rule.json2rule.rst", "source\\generated\\lore_sa.surrogate.DecisionTreeSurrogate.rst", "source\\generated\\lore_sa.surrogate.Surrogate.rst", "source\\generated\\lore_sa.util.rst", "source\\modules.rst"], "titles": ["Tabular explanations example", "lore_sa", "lore_sa.bbox.AbstractBBox", "lore_sa.dataset.Dataset", "lore_sa.dataset.TabularDataset", "lore_sa.dataset.utils", "lore_sa.discretizer.Discretizer", "lore_sa.discretizer.RMEPDiscretizer", "lore_sa.encoder_decoder.EncDec", "lore_sa.encoder_decoder.LabelEnc", "lore_sa.encoder_decoder.OneHotEnc", "lore_sa.explanation.Explanation", "lore_sa.explanation.ExplanationEncoder", "lore_sa.explanation.ImageExplanation", "lore_sa.explanation.MultilabelExplanation", "lore_sa.explanation.TextExplanation", "lore_sa.explanation.json2explanation", "lore_sa.lorem.LOREM", "lore_sa.neighgen.RandomGenerator", "lore_sa.rule.Rule", "lore_sa.rule.RuleGetter", "lore_sa.rule.RuleGetterBinary", "lore_sa.rule.rule.Condition", "lore_sa.rule.rule.ConditionEncoder", "lore_sa.rule.rule.NumpyEncoder", "lore_sa.rule.rule.RuleEncoder", "lore_sa.rule.rule.json2cond", "lore_sa.rule.rule.json2rule", "lore_sa.surrogate.DecisionTreeSurrogate", "lore_sa.surrogate.Surrogate", "lore_sa.util", "Modules"], "terms": {"import": [0, 12, 23, 24, 25], "panda": [0, 4], "pd": 0, "numpi": [0, 8, 9, 10, 17, 24], "np": 0, "from": [0, 4, 5, 9, 10, 12, 18, 23, 24, 25, 28], "sklearn": [0, 2, 10], "preprocess": [0, 28, 29], "ensembl": 0, "randomforestclassifi": 0, "model_select": 0, "train_test_split": 0, "linear_model": 0, "logisticregress": 0, "xailib": 0, "data_load": 0, "dataframe_load": 0, "prepare_datafram": 0, "lime_explain": 0, "limexaitabularexplain": 0, "lore_explain": 0, "loretabularexplain": 0, "shap_explainer_tab": 0, "shapxaitabularexplain": 0, "sklearn_classifier_wrapp": 0, "we": [0, 28], "start": [0, 18, 28], "read": [0, 4], "csv": [0, 4, 5], "file": [0, 4], "analyz": 0, "The": [0, 12, 18, 23, 24, 25, 28], "tabl": 0, "i": [0, 8, 12, 17, 18, 23, 24, 25, 28, 30], "mean": [0, 4], "datafram": [0, 4, 8], "class": [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29], "librari": 0, "among": 0, "all": [0, 12, 23, 24, 25], "attribut": [0, 12, 23, 24, 25], "select": 0, "class_field": 0, "column": [0, 4, 8], "contain": [0, 4, 8, 12, 18, 23, 24, 25], "observ": 0, "correspond": 0, "row": 0, "source_fil": 0, "german_credit": 0, "default": [0, 12, 17, 23, 24, 25], "transform": [0, 30], "df": [0, 4, 8], "read_csv": 0, "skipinitialspac": 0, "true": [0, 12, 17, 22, 23, 24, 25], "na_valu": 0, "keep_default_na": 0, "after": 0, "memori": 0, "need": [0, 17], "extract": [0, 21], "metadata": 0, "inform": [0, 4], "automat": 0, "handl": [0, 3, 4, 8], "content": 0, "withint": 0, "method": [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29], "scan": 0, "follow": 0, "trasform": 0, "version": [0, 12, 23, 24, 25], "origin": [0, 9, 10], "where": 0, "discret": [0, 17], "ar": [0, 8, 12, 23, 24, 25], "numer": [0, 4, 18], "us": [0, 12, 18, 23, 24, 25, 28], "one": 0, "hot": [0, 10], "encod": [0, 8, 9, 10, 12, 23, 24, 25], "strategi": 0, "feature_nam": 0, "list": [0, 2, 8, 12, 17, 18, 23, 24, 25], "containint": 0, "name": [0, 4, 17], "featur": [0, 4, 8, 9, 10, 18], "class_valu": [0, 28], "possibl": 0, "valu": [0, 4, 18, 28, 30], "numeric_column": 0, "e": 0, "continu": 0, "rdf": 0, "befor": 0, "real_feature_nam": 0, "features_map": 0, "dictionari": [0, 3, 4, 12, 17, 23, 24, 25], "point": 0, "each": [0, 4, 8, 12, 18, 23, 24, 25], "train": [0, 5, 28], "rf": 0, "classifi": [0, 2], "split": 0, "test": [0, 2, 12, 23, 24, 25], "subset": 0, "test_siz": 0, "0": [0, 12, 23, 24, 25, 28, 30], "3": [0, 17], "random_st": [0, 17], "42": 0, "x_train": 0, "x_test": 0, "y_train": 0, "y_test": 0, "stratifi": 0, "Then": 0, "set": [0, 4], "onc": 0, "ha": 0, "been": 0, "wrapper": 0, "get": [0, 12, 20, 23, 24, 25], "access": [0, 9, 10], "xai": 0, "lib": 0, "bb": [0, 17], "n_estim": 0, "20": 0, "fit": [0, 30], "bbox": 0, "new": [0, 18], "instanc": [0, 17, 18], "classfi": 0, "print": [0, 12, 23, 24, 25], "inst": 0, "iloc": 0, "147": 0, "8": 0, "reshap": 0, "1": [0, 2, 17, 30], "15": 0, "975": 0, "2": 0, "25": 0, "provid": [0, 2, 4, 9, 10, 17], "an": [0, 4, 8, 9, 10, 12, 17, 18, 23, 24, 25], "explant": 0, "everi": 0, "take": [0, 12, 23, 24, 25], "input": [0, 8, 9, 10, 28], "blackbox": [0, 17], "configur": 0, "object": [0, 4, 12, 17, 18, 23, 24, 25], "initi": [0, 17], "config": [0, 17], "tree": 0, "100": [0, 17], "exp": 0, "plot_features_import": 0, "neigh_typ": 0, "rndgen": 0, "size": 0, "1000": 0, "ocr": 0, "ngen": 0, "10": [0, 30], "plotrul": 0, "plotcounterfactualrul": 0, "limeexplain": 0, "feature_select": 0, "lasso_path": 0, "lime_exp": 0, "as_list": 0, "account_check_statu": 0, "check": [0, 12, 18, 23, 24, 25, 28], "account": 0, "03792512128083548": 0, "duration_in_month": 0, "03701527256562679": 0, "dm": 0, "03144299031649348": 0, "save": 0, "020051934530021572": 0, "ag": 0, "019751080001761446": 0, "credit_histori": 0, "critic": 0, "other": 0, "exist": 0, "thi": [0, 12, 17, 23, 24, 25, 28], "bank": [0, 5], "018970043296280513": 0, "other_installment_plan": 0, "none": [0, 4, 7, 12, 17, 18, 21, 23, 24, 25, 28, 29, 30], "018869997928840695": 0, "017658677626390982": 0, "hous": 0, "own": 0, "014948467979451343": 0, "delai": 0, "pai": 0, "off": 0, "past": 0, "012221985897781883": 0, "plot_lime_valu": 0, "5": [0, 17, 28, 30], "regress": [0, 12, 23, 24, 25], "scaler": 0, "normal": 0, "standardscal": 0, "x_scale": 0, "c": 0, "penalti": 0, "l2": 0, "pass": [0, 2, 12, 23, 24, 25], "record": [0, 8, 17], "182": 0, "27797454": 0, "35504085": 0, "94540357": 0, "07634233": 0, "04854891": 0, "72456474": 0, "43411405": 0, "65027399": 0, "61477862": 0, "25898489": 0, "80681063": 0, "4": 0, "17385345": 0, "6435382": 0, "32533856": 0, "03489416": 0, "20412415": 0, "22941573": 0, "33068147": 0, "75885396": 0, "34899122": 0, "60155441": 0, "15294382": 0, "09298136": 0, "46852129": 0, "12038585": 0, "08481889": 0, "23623492": 0, "21387736": 0, "36174054": 0, "24943031": 0, "15526362": 0, "59715086": 0, "45485883": 0, "73610476": 0, "43875307": 0, "23307441": 0, "65242771": 0, "23958675": 0, "90192655": 0, "72581563": 0, "2259448": 0, "15238005": 0, "54212562": 0, "70181003": 0, "63024248": 0, "30354212": 0, "40586384": 0, "49329429": 0, "88675135": 0, "59227935": 0, "46170508": 0, "46388049": 0, "33747696": 0, "13206764": 0, "same": 0, "previou": 0, "In": 0, "case": 0, "few": 0, "adjust": 0, "necessari": 0, "For": [0, 12, 23, 24, 25], "specif": [0, 12, 23, 24, 25], "linear": 0, "feature_pert": 0, "intervent": 0, "shapxaitabularexplan": 0, "0x12a72dac8": 0, "geneticp": 0, "loretabularexplan": 0, "0x12bc41a90": 0, "why": 0, "becaus": 0, "condit": [0, 23], "happen": 0, "726173400878906credit": 0, "amount": 0, "439": 0, "6443485021591purpos": 0, "retrain": 0, "11524588242173195durat": 0, "month": 0, "9407005310058594purpos": 0, "furnitur": 0, "equip": 0, "18370826542377472foreign": 0, "worker": 0, "7168410122394562purpos": 0, "domest": 0, "applianc": 0, "015466570854187save": 0, "7176859378814697purpos": 0, "vacat": 0, "doe": 0, "4622504562139511credit": 0, "histori": 0, "9085964262485504": 0, "It": [0, 4, 8, 9, 10, 12, 18, 23, 24, 25], "would": [0, 12, 23, 24, 25], "have": 0, "hold": 0, "6443485021591": 0, "26": 0, "468921303749084durat": 0, "795059680938721instal": 0, "incom": [0, 12, 23, 24, 25], "perc": 0, "603440999984741": 0, "other_debtor": 0, "co": 0, "applic": 0, "3046177878918616e": 0, "09": 0, "paid": 0, "back": 0, "duli": 0, "0114574629252053e": 0, "present_emp_sinc": 0, "unemploi": 0, "87554096296626e": 0, "7": 0, "43754044231906e": 0, "free": 0, "4157786564097103e": 0, "properti": 0, "unknown": 0, "275710719845092e": 0, "credit_amount": 0, "271233788564153e": 0, "job": 0, "manag": 0, "self": [0, 5, 12, 23, 24, 25], "emploi": [0, 17], "highli": 0, "qualifi": 0, "employe": 0, "offic": 0, "164190703926506e": 0, "8902027822084106e": 0, "604277452741881e": 0, "skill": 0, "offici": 0, "3808188198617575e": 0, "foreign_work": 0, "ye": 0, "365347360238489e": 0, "telephon": 0, "2048259721367863e": 0, "171945479826713e": 0, "1116662177987812e": 0, "credits_this_bank": 0, "9999632029038067e": 0, "till": 0, "now": 0, "9243622007776865e": 0, "people_under_mainten": 0, "902008911572941e": 0, "purpos": 0, "car": 0, "7104663723358493e": 0, "6584313433238958e": 0, "200": [0, 30], "639544710042764e": 0, "317487567892989e": 0, "unskil": 0, "resid": 0, "307761159896724e": 0, "store": 0, "2347569776391545e": 0, "1825353902253505e": 0, "year": 0, "1478921168922655e": 0, "a121": 0, "a122": 0, "6": 0, "1222769011436428e": 0, "personal_status_sex": 0, "femal": 0, "divorc": 0, "separ": [0, 4, 12, 23, 24, 25], "marri": 0, "1002871894681165e": 0, "500": 0, "0982251402773794e": 0, "0567984890752028e": 0, "present_res_sinc": 0, "9": 0, "869484730455045e": 0, "11": 0, "salari": 0, "assign": 0, "least": 0, "721716212812873e": 0, "327030468700815e": 0, "installment_as_income_perc": 0, "192261925231111e": 0, "real": [0, 18], "estat": 0, "180043418264463e": 0, "974505020571898e": 0, "848004118893571e": 0, "80910843922895e": 0, "educ": 0, "803520453193465e": 0, "busi": 0, "330599059469541e": 0, "rent": 0, "975475868460632e": 0, "build": 0, "societi": 0, "agreement": 0, "life": 0, "insur": 0, "826524390749874e": 0, "guarantor": 0, "385760952840171e": 0, "338094381227495e": 0, "689756440260244e": 0, "582965568284186e": 0, "non": [0, 12, 23, 24, 25], "473736018584135e": 0, "230002403518189e": 0, "974714318917145e": 0, "radio": 0, "televis": 0, "909852887925919e": 0, "620862803354922e": 0, "582941358078461e": 0, "501318386790144e": 0, "male": 0, "widow": 0, "500125372750834e": 0, "regist": 0, "under": 0, "custom": [0, 12, 23, 24, 25], "495252929908006e": 0, "repair": 0, "2177896575440796e": 0, "0557757647139625e": 0, "627184253632623e": 0, "singl": [0, 17], "9862189862658355e": 0, "taken": 0, "8131802175589855e": 0, "9548368945624186e": 0, "modul": 1, "exampl": [1, 12, 23, 24, 25], "tabular": 1, "explan": [1, 17], "index": [1, 4, 28], "search": 1, "page": 1, "sourc": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], "gener": [2, 3, 8, 17, 18, 29], "black": [2, 17], "box": [2, 17], "witch": 2, "two": 2, "like": [2, 12, 23, 24, 25], "__init__": [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29], "model": [2, 5, 30], "abstract": [2, 3, 8, 18], "predict": 2, "sample_matrix": 2, "wrap": 2, "label": [2, 9, 28], "data": [2, 4, 12, 17, 18, 23, 24, 25, 30], "paramet": [2, 9, 10, 17, 18, 21, 28, 30], "arrai": [2, 4, 8, 9, 10, 12, 17, 23, 24, 25], "spars": 2, "matrix": 2, "shape": [2, 30], "n_queri": 2, "n_featur": 2, "sampl": [2, 17, 28], "return": [2, 4, 5, 9, 10, 12, 17, 18, 21, 23, 24, 25, 28, 30], "ndarrai": 2, "n_class": 2, "n_output": 2, "predict_proba": 2, "probabl": 2, "estim": 2, "class_nam": [4, 17, 19], "option": [4, 17, 21], "str": [4, 12, 17, 23, 24, 25], "interfac": [4, 9, 10, 20], "includ": 4, "some": 4, "essenti": 4, "structur": [4, 12, 23, 24, 25], "semant": 4, "whole": 4, "type": [4, 12, 18, 23, 24, 25], "descriptor": [3, 4, 9, 10, 18], "informationregard": 4, "format": 4, "min": 4, "max": 4, "std": 4, "standard": 4, "deviat": 4, "median": 4, "q1": 4, "first": 4, "quartil": 4, "distribut": [4, 30], "q3": 4, "third": 4, "categor": [4, 8], "distinct_valu": 4, "distinct": 4, "value_count": 4, "element": [4, 12, 18, 23, 24, 25], "count": 4, "dict": [4, 9, 10, 12, 18, 23, 24, 25], "classmethod": 4, "from_csv": 4, "filenam": [4, 5], "comma": 4, "param": [4, 5, 8], "from_dict": 4, "seri": 4, "get_class_valu": 4, "set_class_nam": 4, "onli": [4, 8, 12, 18, 23, 24, 25], "string": [4, 12, 17, 23, 24, 25, 28], "update_descriptor": [3, 4], "creat": [3, 4, 18], "function": [5, 8, 9, 10, 12, 17, 23, 24, 25, 30], "prepare_bank_dataset": 5, "http": [5, 10], "www": 5, "kaggl": 5, "com": 5, "aniruddhachoudhuri": 5, "credit": 5, "risk": 5, "home": 5, "riccardo": 5, "scaricati": 5, "to_discret": 7, "proto_fn": 7, "implement": [8, 12, 23, 24, 25], "decod": [8, 9, 10, 12, 23, 24, 25], "differ": 8, "which": [8, 12, 17, 23, 24, 25], "must": 8, "enc": 8, "dec": 8, "enc_fit_transform": 8, "idea": 8, "user": 8, "send": 8, "complet": 8, "here": 8, "variabl": 8, "x": [8, 9, 10, 17, 18, 21, 30], "tabulardataset": [17, 21], "features_to_encod": 8, "appli": [8, 9, 10], "dataset": [8, 17, 18, 21], "encond": [9, 10], "feature_encod": [], "convert": [], "One": 10, "en": 10, "wikipedia": 10, "org": 10, "wiki": 10, "reli": 10, "onehotencod": 10, "inverse_transform": [], "part": [], "dataset_encod": [], "also": [], "modifi": [], "ad": [], "skipkei": [12, 23, 24, 25], "fals": [12, 17, 21, 23, 24, 25, 28], "ensure_ascii": [12, 23, 24, 25], "check_circular": [12, 23, 24, 25], "allow_nan": [12, 23, 24, 25], "sort_kei": [12, 23, 24, 25], "indent": [12, 23, 24, 25], "special": [12, 23, 24, 25], "json": [12, 23, 24, 25], "rule": [12, 17], "constructor": [12, 23, 24, 25], "jsonencod": [12, 23, 24, 25], "sensibl": [12, 23, 24, 25], "If": [12, 23, 24, 25], "typeerror": [12, 23, 24, 25], "attempt": [12, 23, 24, 25], "kei": [12, 18, 23, 24, 25], "int": [12, 17, 18, 23, 24, 25, 28], "float": [12, 23, 24, 25], "item": [12, 23, 24, 25], "simpli": [12, 23, 24, 25], "skip": [12, 23, 24, 25], "output": [12, 23, 24, 25], "guarante": [12, 23, 24, 25], "ascii": [12, 23, 24, 25], "charact": [12, 23, 24, 25], "escap": [12, 23, 24, 25], "can": [12, 23, 24, 25], "circular": [12, 23, 24, 25], "refer": [12, 23, 24, 25], "dure": [12, 17, 23, 24, 25, 28], "prevent": [12, 23, 24, 25], "infinit": [12, 23, 24, 25], "recurs": [12, 23, 24, 25], "caus": [12, 23, 24, 25], "overflowerror": [12, 23, 24, 25], "otherwis": [12, 23, 24, 25], "place": [12, 23, 24, 25], "nan": [12, 23, 24, 25], "infin": [12, 23, 24, 25], "behavior": [12, 23, 24, 25], "compliant": [12, 23, 24, 25], "consist": [12, 23, 24, 25], "most": [12, 23, 24, 25], "javascript": [12, 23, 24, 25], "base": [12, 17, 23, 24, 25], "valueerror": [12, 23, 24, 25], "sort": [12, 23, 24, 25], "ensur": [12, 23, 24, 25], "serial": [12, 23, 24, 25], "compar": [12, 23, 24, 25], "dai": [12, 23, 24, 25], "basi": [12, 23, 24, 25], "neg": [12, 23, 24, 25], "integ": [12, 23, 24, 25, 28], "member": [12, 23, 24, 25], "pretti": [12, 23, 24, 25], "level": [12, 23, 24, 25], "insert": [12, 23, 24, 25], "newlin": [12, 23, 24, 25], "compact": [12, 23, 24, 25], "represent": [12, 23, 24, 25], "specifi": [12, 23, 24, 25], "should": [12, 23, 24, 25], "item_separ": [12, 23, 24, 25], "key_separ": [12, 23, 24, 25], "tupl": [12, 23, 24, 25], "To": [12, 23, 24, 25], "you": [12, 23, 24, 25], "elimin": [12, 23, 24, 25], "whitespac": [12, 23, 24, 25], "call": [12, 23, 24, 25], "t": [12, 23, 24, 25], "rais": [12, 23, 24, 25], "obj": [12, 16, 23, 24, 25, 26, 27], "subclass": [12, 23, 24, 25], "serializ": [12, 23, 24, 25], "o": [12, 23, 24, 25], "support": [12, 23, 24, 25], "arbitrari": [12, 23, 24, 25], "iter": [12, 23, 24, 25], "could": [12, 23, 24, 25], "def": [12, 23, 24, 25], "try": [12, 23, 24, 25], "except": [12, 23, 24, 25], "els": [12, 23, 24, 25], "let": [12, 23, 24, 25], "python": [12, 23, 24, 25], "foo": [12, 23, 24, 25], "bar": [12, 23, 24, 25], "baz": [12, 23, 24, 25], "iterencod": [12, 23, 24, 25], "_one_shot": [12, 23, 24, 25], "given": [12, 23, 24, 25], "yield": [12, 23, 24, 25], "avail": [12, 23, 24, 25], "chunk": [12, 23, 24, 25], "bigobject": [12, 23, 24, 25], "mysocket": [12, 23, 24, 25], "write": [12, 23, 24, 25], "img": 13, "segment": 13, "text": 15, "indexed_text": 15, "abstractbbox": 17, "encdec": [17, 21], "neigh_gen": 17, "neighborhoodgener": 17, "surrog": 17, "k_transform": 17, "multi_label": [17, 21, 28], "filter_crul": 17, "kernel_width": 17, "kernel": 17, "binari": 17, "extreme_fidel": 17, "bool": [17, 21, 28], "constraint": 17, "verbos": 17, "kwarg": 17, "local": 17, "incapsul": 17, "datamanag": 17, "explain_instance_st": 17, "use_weight": 17, "metric": 17, "neuclidean": 17, "run": 17, "exemplar_num": 17, "n_job": 17, "prune_tre": [17, 28], "explain": [17, 18], "stabl": 17, "number": [17, 18], "neighbourhood": 17, "measur": 17, "distanc": 17, "between": 17, "time": 17, "done": 17, "examplar": 17, "retriev": 17, "add": 17, "cf": 17, "random": 18, "neighbor": 18, "check_gener": 18, "filter_funct": 18, "check_fuct": 18, "logic": 18, "requir": 18, "num_inst": 18, "detect": 18, "order": 18, "rang": 18, "associ": 18, "randomli": 18, "choic": 18, "within": 18, "premis": 19, "con": 19, "counterfactu": 20, "get_rul": 21, "y": 21, "dt": [21, 28], "decisiontreesurrog": 21, "att": 22, "op": 22, "thr": 22, "is_continu": 22, "kind": [28, 29], "is_leaf": 28, "inner_tre": 28, "whether": 28, "node": 28, "leaf": 28, "prune_duplicate_leav": 28, "remov": 28, "leav": 28, "both": 28, "prune_index": 28, "decis": 28, "prune": 28, "bottom": 28, "top": 28, "might": 28, "miss": 28, "becom": 28, "do": 28, "directli": 28, "instead": 28, "z": 28, "yb": 28, "weight": 28, "one_vs_rest": 28, "cv": 28, "target": 28, "best_fit_distribut": 30, "bin": 30, "ax": 30, "find": 30, "best": 30, "sigmoid": 30, "x0": 30, "k": 30, "l": 30, "A": 30, "logist": 30, "curv": 30, "common": 30, "": 30, "midpoint": 30, "maximum": 30, "steep": 30, "dataset_descriptor": 8, "stare": [9, 10]}, "objects": {"lore_sa.bbox": [[2, 0, 1, "", "AbstractBBox"]], "lore_sa.bbox.AbstractBBox": [[2, 1, 1, "", "__init__"], [2, 1, 1, "", "model"], [2, 1, 1, "", "predict"], [2, 1, 1, "", "predict_proba"]], "lore_sa.dataset": [[3, 0, 1, "", "Dataset"], [4, 0, 1, "", "TabularDataset"], [5, 3, 0, "-", "utils"]], "lore_sa.dataset.Dataset": [[3, 1, 1, "", "__init__"], [3, 1, 1, "", "update_descriptor"]], "lore_sa.dataset.TabularDataset": [[4, 1, 1, "", "__init__"], [4, 2, 1, "", "descriptor"], [4, 2, 1, "", "df"], [4, 1, 1, "", "from_csv"], [4, 1, 1, "", "from_dict"], [4, 1, 1, "", "get_class_values"], [4, 1, 1, "", "set_class_name"], [4, 1, 1, "", "update_descriptor"]], "lore_sa.dataset.utils": [[5, 4, 1, "", "prepare_bank_dataset"]], "lore_sa.discretizer": [[6, 0, 1, "", "Discretizer"], [7, 0, 1, "", "RMEPDiscretizer"]], "lore_sa.discretizer.Discretizer": [[6, 1, 1, "", "__init__"]], "lore_sa.discretizer.RMEPDiscretizer": [[7, 1, 1, "", "__init__"]], "lore_sa.encoder_decoder": [[8, 0, 1, "", "EncDec"], [9, 0, 1, "", "LabelEnc"], [10, 0, 1, "", "OneHotEnc"]], "lore_sa.encoder_decoder.EncDec": [[8, 1, 1, "", "__init__"], [8, 1, 1, "", "encode"]], "lore_sa.encoder_decoder.LabelEnc": [[9, 1, 1, "", "__init__"], [9, 1, 1, "", "decode"], [9, 1, 1, "", "encode"]], "lore_sa.encoder_decoder.OneHotEnc": [[10, 1, 1, "", "__init__"], [10, 1, 1, "", "decode"], [10, 1, 1, "", "encode"]], "lore_sa.explanation": [[11, 0, 1, "", "Explanation"], [12, 0, 1, "", "ExplanationEncoder"], [13, 0, 1, "", "ImageExplanation"], [14, 0, 1, "", "MultilabelExplanation"], [15, 0, 1, "", "TextExplanation"], [16, 4, 1, "", "json2explanation"]], "lore_sa.explanation.Explanation": [[11, 1, 1, "", "__init__"]], "lore_sa.explanation.ExplanationEncoder": [[12, 1, 1, "", "__init__"], [12, 1, 1, "", "default"], [12, 1, 1, "", "encode"], [12, 1, 1, "", "iterencode"]], "lore_sa.explanation.ImageExplanation": [[13, 1, 1, "", "__init__"]], "lore_sa.explanation.MultilabelExplanation": [[14, 1, 1, "", "__init__"]], "lore_sa.explanation.TextExplanation": [[15, 1, 1, "", "__init__"]], "lore_sa.lorem": [[17, 0, 1, "", "LOREM"]], "lore_sa.lorem.LOREM": [[17, 1, 1, "", "__init__"], [17, 1, 1, "", "explain_instance_stable"]], "lore_sa.neighgen": [[18, 0, 1, "", "RandomGenerator"]], "lore_sa.neighgen.RandomGenerator": [[18, 1, 1, "", "__init__"], [18, 1, 1, "", "check_generated"], [18, 1, 1, "", "generate"]], "lore_sa.rule": [[19, 0, 1, "", "Rule"], [20, 0, 1, "", "RuleGetter"], [21, 0, 1, "", "RuleGetterBinary"]], "lore_sa.rule.Rule": [[19, 1, 1, "", "__init__"]], "lore_sa.rule.RuleGetter": [[20, 1, 1, "", "__init__"]], "lore_sa.rule.RuleGetterBinary": [[21, 1, 1, "", "__init__"], [21, 1, 1, "", "get_rule"]], "lore_sa.rule.rule": [[22, 0, 1, "", "Condition"], [23, 0, 1, "", "ConditionEncoder"], [24, 0, 1, "", "NumpyEncoder"], [25, 0, 1, "", "RuleEncoder"], [26, 4, 1, "", "json2cond"], [27, 4, 1, "", "json2rule"]], "lore_sa.rule.rule.Condition": [[22, 1, 1, "", "__init__"]], "lore_sa.rule.rule.ConditionEncoder": [[23, 1, 1, "", "__init__"], [23, 1, 1, "", "default"], [23, 1, 1, "", "encode"], [23, 1, 1, "", "iterencode"]], "lore_sa.rule.rule.NumpyEncoder": [[24, 1, 1, "", "__init__"], [24, 1, 1, "", "default"], [24, 1, 1, "", "encode"], [24, 1, 1, "", "iterencode"]], "lore_sa.rule.rule.RuleEncoder": [[25, 1, 1, "", "__init__"], [25, 1, 1, "", "default"], [25, 1, 1, "", "encode"], [25, 1, 1, "", "iterencode"]], "lore_sa.surrogate": [[28, 0, 1, "", "DecisionTreeSurrogate"], [29, 0, 1, "", "Surrogate"]], "lore_sa.surrogate.DecisionTreeSurrogate": [[28, 1, 1, "", "__init__"], [28, 1, 1, "", "is_leaf"], [28, 1, 1, "", "prune_duplicate_leaves"], [28, 1, 1, "", "prune_index"], [28, 1, 1, "", "train"]], "lore_sa.surrogate.Surrogate": [[29, 1, 1, "", "__init__"]], "lore_sa": [[30, 3, 0, "-", "util"]], "lore_sa.util": [[30, 4, 1, "", "best_fit_distribution"], [30, 4, 1, "", "sigmoid"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:attribute", "3": "py:module", "4": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "attribute", "Python attribute"], "3": ["py", "module", "Python module"], "4": ["py", "function", "Python function"]}, "titleterms": {"tabular": 0, "explan": [0, 11, 12, 13, 14, 15, 16, 31], "exampl": 0, "learn": 0, "explain": 0, "german": 0, "credit": 0, "dataset": [0, 3, 4, 5, 31], "load": 0, "prepar": 0, "data": 0, "random": 0, "forest": 0, "classfier": 0, "predict": 0, "shap": 0, "lore": 0, "lime": 0, "differ": 0, "model": 0, "logist": 0, "regressor": 0, "lore_sa": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], "indic": 1, "tabl": 1, "bbox": [2, 31], "abstractbbox": 2, "tabulardataset": 4, "util": [5, 30, 31], "discret": [6, 7, 31], "rmepdiscret": 7, "encoder_decod": [8, 9, 10, 31], "encdec": 8, "labelenc": 9, "onehotenc": 10, "explanationencod": 12, "imageexplan": 13, "multilabelexplan": 14, "textexplan": 15, "json2explan": 16, "lorem": [17, 31], "neighgen": [18, 31], "randomgener": 18, "rule": [19, 20, 21, 22, 23, 24, 25, 26, 27, 31], "rulegett": 20, "rulegetterbinari": 21, "condit": 22, "conditionencod": 23, "numpyencod": 24, "ruleencod": 25, "json2cond": 26, "json2rul": 27, "surrog": [28, 29, 31], "decisiontreesurrog": 28, "modul": 31, "class": 31, "blackbox": 31, "abstract": 31, "neighborhood": 31, "gener": 31, "function": 31, "encod": 31, "decod": 31}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})