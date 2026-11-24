"""
Tests for the headless CLI setup in process_emission_rates.py
a config that fails to load must stop the run (SystemExit) instead of
processing with the built-in default config.

The workers are mocked where possible
"""
from unittest import mock

import pytest
import yaml

import process_emission_rates as per

IFIT_KEYS = ['species_paths', 'dark_img_dir', 'ILS_path', 'ld_lookup_1', 'ld_lookup_2', 'spec_dir']


def write_config(tmp_path, config):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return path


def full_ifit_config():
    return {
        'species_paths': {'SO2': {'path': 'SO2.txt', 'value': 1.0e16}},
        'dark_img_dir': 'darks',
        'ILS_path': 'ils.txt',
        'ld_lookup_1': 'ld1.npy',
        'ld_lookup_2': 'ld2.npy',
        'spec_dir': 'spectra',
    }


# --- setup_ifit_worker (used by doas, pyplis and watcher) ---------------------

def test_ifit_missing_keys_exits_and_names_them(tmp_path):
    config = full_ifit_config()
    del config['species_paths']
    del config['ILS_path']
    config_path = write_config(tmp_path, config)

    with mock.patch.object(per, 'IFitWorker') as ifit_cls:
        with pytest.raises(SystemExit) as exc:
            per.setup_ifit_worker(str(config_path))

    message = str(exc.value.code)
    assert 'species_paths' in message
    assert 'ILS_path' in message
    assert str(config_path) in message
    ifit_cls.assert_not_called()


def test_ifit_empty_config_exits(tmp_path):
    config_path = tmp_path / "empty.yml"
    config_path.write_text("")

    with mock.patch.object(per, 'IFitWorker') as ifit_cls:
        with pytest.raises(SystemExit) as exc:
            per.setup_ifit_worker(str(config_path))

    for key in IFIT_KEYS:
        assert key in str(exc.value.code)
    ifit_cls.assert_not_called()


def test_ifit_valid_config_continues(tmp_path):
    config_path = write_config(tmp_path, full_ifit_config())

    with mock.patch.object(per, 'IFitWorker') as ifit_cls:
        worker = per.setup_ifit_worker(str(config_path))

    assert worker is ifit_cls.return_value
    # Relative paths in the config are resolved against the config's folder
    worker.load_ils.assert_called_once_with(str(tmp_path / 'ils.txt'))
    worker.load_dir.assert_called_once()


# --- setup_pyplis_worker (used by pyplis and watcher) -------------------------

def test_pyplis_unloadable_config_exits_without_fallback(tmp_path):
    """Real PyplisWorker: a config file that does not exist triggers its fallback,
    which the headless setup must turn into a stop."""
    missing = tmp_path / "does_not_exist.yml"

    with pytest.raises(SystemExit) as exc:
        per.setup_pyplis_worker(str(missing))

    message = str(exc.value.code)
    assert 'could not load config' in message
    assert str(missing) in message


def test_pyplis_fallback_flag_exits_before_setup():
    with mock.patch.object(per, 'PyplisWorker') as pw_cls:
        pw_cls.return_value.load_default_conf_errors = "Problem loading specified default config file"
        with pytest.raises(SystemExit):
            per.setup_pyplis_worker("some_config.yml")

    pw_cls.return_value.load_pcs_from_config.assert_not_called()


def test_pyplis_valid_config_continues():
    with mock.patch.object(per, 'PyplisWorker') as pw_cls, \
            mock.patch.object(per, 'setup_ifit_worker') as setup_ifit:
        instance = pw_cls.return_value
        instance.load_default_conf_errors = None
        instance.config = {'default_cam_geom': 'geom.txt'}
        instance.opt_flow_sett_keys = []

        worker = per.setup_pyplis_worker("some_config.yml")

    assert worker is instance
    assert worker.headless is True
    instance.load_pcs_from_config.assert_called_once()
    setup_ifit.assert_called_once_with("some_config.yml")
