from lore_sa.neighgen.genetic_generator import GeneticGenerator
from lore_sa.util import sigmoid, neuclidean
from scipy.spatial.distance import cdist, cosine

import numpy as np

__all__ = ["GeneticGenerator","GeneticProbaGenerator"]

class GeneticProbaGenerator(GeneticGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2,
                 cxpb=0.5, tournsize=3, halloffame_ratio=0.1, bb_predict_proba=None, random_seed=None, encdec = None, verbose=False):
        super(GeneticProbaGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                                    nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                                    numeric_columns_index=numeric_columns_index,
                                                    ocr=ocr, alpha1=alpha1, alpha2=alpha2, metric=metric, ngen=ngen,
                                                    mutpb=mutpb,cxpb=cxpb, tournsize=tournsize,
                                                    halloffame_ratio=halloffame_ratio, random_seed=random_seed, encdec=encdec)
        self.bb_predict_proba = bb_predict_proba

    def fitness_equal(self, x, x1):
        return self.fitness_equal_proba(x, x1)

    def fitness_notequal(self, x, x1):
        return self.fitness_notequal_proba(x, x1)

    def fitness_equal_proba(self, x, x1):
        x = np.array(x)
        x1 = np.array(x1)
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = sigmoid(feature_similarity_score) if feature_similarity_score < 1.0 else 0.0
        # feature_similarity = sigmoid(feature_similarity_score)

        #y = self.bb_predict_proba(x.reshape(1, -1))[0]
        #y1 = self.bb_predict_proba(x1.reshape(1, -1))[0]
        y = self.apply_bb_predict_proba(x.reshape(1, -1))[0]
        y1 = self.apply_bb_predict_proba(x1.reshape(1, -1))[0]

        # target_similarity_score = np.sum(np.abs(y - y1))
        target_similarity_score = 1.0 - cosine(y, y1)
        target_similarity = sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,

    def fitness_notequal_proba(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = sigmoid(feature_similarity_score)

        #y = self.bb_predict_proba(x.reshape(1, -1))[0]
        #y1 = self.bb_predict_proba(x1.reshape(1, -1))[0]

        y = self.apply_bb_predict_proba(x.reshape(1, -1))[0]
        y1 = self.apply_bb_predict_proba(x1.reshape(1, -1))[0]

        # target_similarity_score = np.sum(np.abs(y - y1))
        target_similarity_score = 1.0 - cosine(y, y1)
        target_similarity = 1.0 - sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,
