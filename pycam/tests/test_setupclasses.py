# -*- coding: utf-8 -*-

"""
pycam test module for setupclasses
"""

import pytest
from pycam.setupclasses import CameraSpecs
import numpy as np

class TestSpecs:
    @pytest.fixture
    def filename(self, tmp_path):
        """Temporary specs file, so tests don't depend on the working directory or modify tracked files.
        The file must already exist: save_specs() checks for it, then overwrites it"""
        path = tmp_path / 'temp_io_test.txt'
        path.touch()
        return str(path)

    def test_cam_specs_io(self, filename):
        """Test basic IO functionality of the CameraSpecs object"""
        # Instantiate CameraSpecs object
        cam_1 = CameraSpecs()

        # Edit value from default
        cam_1.shutter_speed = 1000

        # Edit another value
        cam_1.framerate = 0.5

        # Attempt to save camera specs
        cam_1.save_specs(filename)

        # Reload camera specs
        cam_2 = CameraSpecs(filename)

        # Check cam_1 against cam_2 to ensure load and save has been performed as anticipated
        # assert cam_1.__dict__ == cam_2.__dict__
        for key in cam_1.__dict__:
            if isinstance(cam_1.__dict__[key], np.ndarray):
                assert np.array_equal(cam_1.__dict__[key], cam_2.__dict__[key])
            else:
                assert cam_1.__dict__[key] == cam_2.__dict__[key]

    def test_changing_bool(self, filename):
        """Test changing bool value and saving"""
        # Create first CameraSpecs obj
        cam_1 = CameraSpecs()

        # Change auto_ss to False
        cam_1.auto_ss = False

        # Save camera specs
        cam_1.save_specs(filename)

        # Load new CameraSpecs
        cam_2 = CameraSpecs(filename)

        assert cam_1.auto_ss == cam_2.auto_ss

    def test_incorrect_bool(self, filename):
        """Tests performance if incorrect bool is saved to, and then read from, file"""
        cam_1 = CameraSpecs()

        # Define auto_ss incorrectly
        cam_1.auto_ss = 3

        # Save incorrect definition
        cam_1.save_specs(filename)

        # Load specifications (auto_ss should revert to default - True)
        cam_1.load_specs(filename)

        # Check auto_ss has reverted to default
        assert cam_1.auto_ss == True
