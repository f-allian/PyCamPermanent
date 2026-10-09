from pycam.doas.cfg import doas_worker, process_settings
import pandas as pd
import numpy as np
import unittest

class TestIfitWorker(unittest.TestCase):

    def test_ifit_regression(self):
        # provide specific settings 
        ils_path = './pycam/doas/calibration/2019-07-03_302nm_ILS.txt'

        doas_worker.load_ils(ils_path)
        doas_worker.corr_light_dilution = True

        doas_worker.load_ld_lookup(process_settings["ld_lookup_1"], fit_num=0)
        doas_worker.load_ld_lookup(process_settings["ld_lookup_2"], fit_num=1)

        # Run doas_worker
        doas_worker.start_processing_threadless()
        # Load output(s) from new run
        # Load gold standard output(s)
        expected_output = pd.read_csv("./pycam/tests/test_data/test_spectra/ifit_test_standard/doas_results_2019-05-29T124401_124412.csv")

        # Compare original and gold standard
        self.assertTrue(all(expected_output["Time"] == doas_worker.results.index))
        # 1% relative tolerance: the gold standard was generated in 2023 with older numpy/scipy/Shapely, and fit results
        # differ by up to ~0.6% with current pinned versions
        self.assertTrue(np.allclose(doas_worker.results.values, expected_output["Column density"], rtol = 1e-2, equal_nan = True))
        self.assertTrue(np.allclose(doas_worker.results.fit_errs, expected_output["CD error"], rtol = 1e-2, equal_nan = True))
        # LDF is not compared: none of the test spectra fall inside the light dilution lookup grid, so the LDF column only
        # reflects the fallback branch in IFitWorker.ld_lookup()