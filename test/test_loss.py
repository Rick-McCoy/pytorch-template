import unittest

import torch
import numpy as np
from hydra import initialize, compose

from model.loss import SimpleLoss


class TestLoss(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.loss = SimpleLoss(cfg)

    def test_loss(self):
        logit_1 = torch.zeros(8, self.cfg.model.num_class)
        target_1 = torch.zeros(8, dtype=torch.int64)
        self.assertAlmostEqual(self.loss(logit_1, target_1).numpy(),
                               np.log(10),
                               places=4)
        logit_2 = torch.full((8, self.cfg.model.num_class), -1e9)
        logit_2[torch.arange(8), torch.arange(8)] = 1e9
        target_2 = torch.arange(8, dtype=torch.int64)
        self.assertAlmostEqual(self.loss(logit_2, target_2).numpy(),
                               0,
                               places=4)
