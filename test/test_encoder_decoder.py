import unittest
from lore_sa.encoder_decoder import OneHotEnc, LabelEnc
from lore_sa.encoder_decoder.my_target_enc import TargetEnc
import numpy as np

descriptor_dummy = {'categoric': {'col3': {'count': {'America': 1, 'Europe': 1,'Africa':1},
                                        'distinct_values': ['America', 'Europe', 'Africa'],
                                        'index': 2},
                                  'colours': {
                                        'distinct_values': ['White', 'Black', 'Red','Blue','Green'],
                                        'index': 4
                                  }},
                     'numeric': {'col1': {'index': 0,
                                          'max': 2,
                                          'mean': 1.5,
                                          'median': 1.5,
                                          'min': 1,
                                          'q1': 1.25,
                                          'q3': 1.75,
                                          'std': 0.7071067811865476},
                                 'col2': {'index': 1,
                                          'max': 4,
                                          'mean': 3.5,
                                          'median': 3.5,
                                          'min': 3,
                                          'q1': 3.25,
                                          'q3': 3.75,
                                          'std': 0.7071067811865476}},
                    'ordinal': {'education': {'index':3,
                                      'distinct_values':['Elementary','High School','College','Graduate','Post-graduate']}
                    }}

class EncDecTest(unittest.TestCase):

    def test_one_hot_encoder_init(self):
        one_hot_enc = OneHotEnc(descriptor_dummy)
        self.assertEqual(one_hot_enc.type,'one-hot')
        self.assertEqual(one_hot_enc.__str__(),"OneHotEncoder - no features encoded")

    def test_target_encoder_init(self):
        target_enc = TargetEnc(descriptor_dummy)
        self.assertEqual(target_enc.type,'target')
        self.assertEqual(target_enc.__str__(),"TargetEncoder - no features encoded")


    def test_one_hot_encoder_init_with_features_encoder(self):
        one_hot_enc = OneHotEnc(descriptor_dummy)
        encoded = one_hot_enc.encode(np.array([1, 2, "Europe", "Graduate", "Green"]))
        self.assertEqual(one_hot_enc.__str__(),"OneHotEncoder - features encoded: col3,colours")
        self.assertEqual(encoded.tolist(),np.array([1, 2, 0, 1, 0, "Graduate", 0, 0, 0, 0, 1]).tolist())
        self.assertEqual(one_hot_enc.dataset_descriptor['categoric']['col3']['index'],2)
        self.assertEqual(one_hot_enc.dataset_descriptor['categoric']['colours']['index'], 6)
        self.assertEqual(one_hot_enc.dataset_descriptor['numeric']['col1']['index'], 0)
        self.assertEqual(one_hot_enc.dataset_descriptor['numeric']['col2']['index'], 1)
        self.assertEqual(one_hot_enc.dataset_descriptor['ordinal']['education']['index'], 5)

    def test_one_hot_decode_init_with_features_encoder(self):
        one_hot_enc = OneHotEnc(descriptor_dummy)
        decoded = one_hot_enc.decode(np.array([1, 2, 0, 0, 1, "Graduate", 0, 0, 0, 1, 0]))

        self.assertEqual(decoded.tolist(), np.array([1, 2, "Africa", "Graduate", "Blue"]).tolist())


    def test_label_encoder_init_with_features_encoder(self):
        label_enc = LabelEnc(descriptor_dummy)
        encoded = label_enc.encode(np.array([1,2,"Europe","Graduate"]))
        self.assertEqual(label_enc.__str__(),"LabelEncoder - features encoded: education")
        self.assertEqual(encoded[3],'3')
        self.assertEqual(encoded.tolist(),np.array([1,2,"Europe",3]).tolist())


    def test_label_decode(self):
        label_enc = LabelEnc(descriptor_dummy)
        decoded = label_enc.decode(np.array([1,2,"Europe",0]))

        self.assertEqual(decoded[3],'Elementary')
        self.assertEqual(decoded.tolist(),np.array([1,2,"Europe",'Elementary']).tolist())

if __name__ == '__main__':
    unittest.main()
