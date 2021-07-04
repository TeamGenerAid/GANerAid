import os
import shutil
import unittest

import numpy as np
import torch
from GANerAid.ganeraid import GANerAid
import pandas as pd


class GANerAidUnitTests(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gan = GANerAid(self.device)
        self.data = pd.read_csv('../../Breast_cancer_data.csv')

    def test_setup_of_parameters(self):
        lr_d = .2
        lr_g = .2
        hidden_feature_space = 100
        batch_size = 10
        nr_of_rows = 20
        binary_noise = .1

        gan = GANerAid(self.device, lr_d=lr_d, lr_g=lr_g, hidden_feature_space=hidden_feature_space,
                       batch_size=batch_size, nr_of_rows=nr_of_rows, binary_noise=binary_noise)

        self.assertEqual(lr_d, gan.lr_d)
        self.assertEqual(lr_g, gan.lr_g)
        self.assertEqual(hidden_feature_space, gan.hidden_feature_space)
        self.assertEqual(nr_of_rows, gan.nr_of_rows)
        self.assertEqual(binary_noise, gan.binary_noise)

    def test_default_run(self):
        lr_d = 5e-2
        lr_g = 5e-2
        hidden_feature_space = 50
        batch_size = 50
        nr_of_rows = 50
        binary_noise = .4

        gan = GANerAid(self.device, lr_d=lr_d, lr_g=lr_g, hidden_feature_space=hidden_feature_space,
                       batch_size=batch_size, nr_of_rows=nr_of_rows, binary_noise=binary_noise)

        gan.fit(self.data, 5)
        gen_data = gan.generate(self.data.shape[0])

        self.assertEqual(self.data.shape, gen_data.shape)
        self.assertTrue(self.data.dtypes.equals(gen_data.dtypes))
        self.assertTrue(self.data.columns.equals(gen_data.columns))

    def test_run_with_different_parameters(self):
        self.gan.fit(self.data, 5)
        gen_data = self.gan.generate(self.data.shape[0])

        self.assertEqual(self.data.shape, gen_data.shape)
        self.assertTrue(self.data.dtypes.equals(gen_data.dtypes))
        self.assertTrue(self.data.columns.equals(gen_data.columns))

    def test_continue_fit(self):
        # fit once
        self.gan.fit(self.data, 5)
        # fit again
        self.gan.fit(self.data, 5)

        gen_data = self.gan.generate(self.data.shape[0])

        self.assertEqual(self.data.shape, gen_data.shape)
        self.assertTrue(self.data.dtypes.equals(gen_data.dtypes))
        self.assertTrue(self.data.columns.equals(gen_data.columns))

    def test_error_handling(self):
        # test wrong input data
        with self.assertRaises(ValueError):
            self.gan.fit(np.empty((100, 100)))

        # test generate without fit first
        with self.assertRaises(ValueError):
            self.gan.generate(100)

        # test evaluation without fit first
        with self.assertRaises(ValueError):
            self.gan.evaluate(self.data, self.data)

        # test save without fit first
        with self.assertRaises(ValueError):
            self.gan.save("test")

    def test_data_aug(self):
        self.gan.fit(self.data, epochs=5, aug_factor=1)
        # assert that aug factor of 1 doubles the data
        self.assertEqual(self.gan.dataset.shape[0], self.data.shape[0] * 2)

    def test_save_and_load_gan(self):
        self.gan.fit(self.data, epochs=5)
        self.gan.save("gan_checkpoint", "BEST_GAN")
        assert os.path.exists("gan_checkpoint") == 1

        # reload gan
        self.gan = GANerAid.load(self.device, "gan_checkpoint", "BEST_GAN")

        gen_data = self.gan.generate(self.data.shape[0])
        self.assertEqual(self.data.shape, gen_data.shape)

        # cleanup
        shutil.rmtree("gan_checkpoint")


if __name__ == '__main__':
    unittest.main()
