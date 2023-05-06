import unittest

import torch._dynamo.config
from hydra import compose, initialize

from main import main


class TestMain(unittest.TestCase):
    def test_fast_dev(self):
        torch._dynamo.config.verbose = True
        with initialize(config_path="../config"):
            cfg = compose(config_name="config", overrides=["train.fast_dev_run=True"])
            main(cfg)
