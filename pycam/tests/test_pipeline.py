"""
Tests for the automated pipeline in pipeline/ (run_mvp.py, web_data.py).

These use small fake files in a temporary folder. PyCam itself is not run and nothing
is copied over the network
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import pytest
import yaml

# The pipeline scripts import each other as plain modules (from sources import ...)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline'))

import run_mvp
import web_data

DATE = datetime.date(2026, 11, 24)


# Helpers

def make_template(folder, **changes):
    """Write a template whose path settings all point at files that exist. Returns its path."""
    data_dir = folder / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ['dark_img', 'dark_spec', 'cell_cal']:
        (data_dir / name).mkdir(exist_ok=True)
    for name in ['bg_A.npy', 'bg_B.npy', 'line_0.txt', 'reg.txt', 'dil.txt', 'ld1.npy', 'ld2.npy',
                 'ils.txt', 'geom.txt', 'so2.txt', 'o3.txt']:
        (data_dir / name).touch()

    config = {
        'img_dir': './old_images',
        'spec_dir': './old_spectra',
        'transfer_dir': './old_transfer',
        'dark_img_dir': 'data/dark_img',
        'dark_spec_dir': 'data/dark_spec',
        'cell_cal_dir': 'data/cell_cal',
        'bg_A_path': 'data/bg_A.npy',
        'bg_B_path': 'data/bg_B.npy',
        'pcs_lines': ['data/line_0.txt'],
        'img_registration': 'data/reg.txt',
        'dil_lines': ['data/dil.txt'],
        'ld_lookup_1': 'data/ld1.npy',
        'ld_lookup_2': 'data/ld2.npy',
        'ILS_path': 'data/ils.txt',
        'default_cam_geom': 'data/geom.txt',
        'species_paths': {'SO2': {'path': 'data/so2.txt'}, 'O3': {'path': 'data/o3.txt'}},
        'cal_type_int': 1,
        'img_buff_size': 1500,
    }
    config.update(changes)
    template = folder / 'template.yml'
    with open(template, 'w') as f:
        yaml.safe_dump(config, f)
    return template


def make_args(dry_run=False, date=None, max_hours=None):
    return argparse.Namespace(dry_run=dry_run, date=date, max_hours=max_hours)


def image_names(date, hour, minutes):
    """Image file names spread over the given number of minutes of one hour."""
    start = datetime.datetime.combine(date, datetime.time(int(hour)))
    return [f"{start + datetime.timedelta(minutes=m):%Y-%m-%dT%H%M%S}_fltrA_1ss_Plume.png"
            for m in range(0, minutes + 1, 5)]


class FakeSource:
    """Stands in for LocalSource / SshSource: hours and minutes of data are given up front."""

    def __init__(self, minutes_per_hour, problem=None):
        self.minutes_per_hour = minutes_per_hour   # {'12': 60, ...}
        self.problem = problem

    def describe(self):
        return 'fake source'

    def check(self):
        return self.problem

    def list_hours(self, date):
        return sorted(self.minutes_per_hour)

    def list_files(self, date, hour):
        return image_names(date, hour, self.minutes_per_hour[hour])


def write_emission_file(path, times, flux):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Emission rates', '# datetime, flux (kg/s), flux_err (kg/s)']
    for t in times:
        lines.append(f"{t:%Y-%m-%d %H:%M:%S}, {flux}, 0.1")
    path.write_text('\n'.join(lines) + '\n')


def write_calibration_file(path, times, r2_values):
    lines = ['header line'] * 5 + ['timepoint,col density (doas),r-squared']
    for t, r2 in zip(times, r2_values):
        lines.append(f"{t:%Y-%m-%d %H:%M:%S},1e17,{r2}")
    path.write_text('\n'.join(lines) + '\n')


def make_processed_hour(folder, hour, flux, r2_values):
    """A fake pyplis Processed_ folder for one hour with all three flow types."""
    times = [datetime.datetime.combine(DATE, datetime.time(hour, m)) for m in (0, 10, 20)]
    processed = folder / f"Processed_2026-11-24T{hour:02d}0000"
    for flow in web_data.FLOW_TYPES:
        write_emission_file(processed / 'line_0' / f"20261124_EmissionRates_flow_{flow}.txt", times, flux)
    write_calibration_file(processed / 'full_calibration.csv', times, r2_values)
    return processed


@pytest.fixture
def pipeline_root(tmp_path, monkeypatch):
    """A pipeline folder with one site, its clock set to 18:00 UTC on DATE and processing faked."""
    template = make_template(tmp_path / 'template')
    site = {'name': 'testsite', 'template': str(template), 'source': {'type': 'local'},
            'min_minutes': 45, 'grace_minutes': 10, 'max_hours_per_run': 24}
    root = tmp_path / 'root'
    root.mkdir()
    paths = {'root': root, 'code': tmp_path / 'code', 'python': sys.executable}

    monkeypatch.setattr(run_mvp, 'now_utc', lambda: datetime.datetime(2026, 11, 24, 18, 0))

    processed = []

    def fake_process_hour(site, source, date, hour, paths):
        processed.append(hour)
        return {'files': 10, 'doas_rows': 5, 'emission_files': 3, 'rows_line0': 5}

    monkeypatch.setattr(run_mvp, 'process_hour', fake_process_hour)
    return site, paths, processed


def use_source(monkeypatch, source):
    monkeypatch.setattr(run_mvp, 'make_source', lambda settings: source)


# Choosing which hours to process

def test_full_hours_are_processed_and_marked_done(pipeline_root, monkeypatch):
    site, paths, processed = pipeline_root
    use_source(monkeypatch, FakeSource({'12': 60, '13': 60}))

    updated = run_mvp.run_site(site, paths, make_args(date='2026-11-24'))

    assert processed == ['12', '13']
    assert updated == {DATE}
    state_dir = paths['root'] / 'state' / 'testsite'
    assert (state_dir / '2026-11-24_12.done').exists()
    assert (state_dir / '2026-11-24_13.done').exists()


def test_short_hour_is_skipped(pipeline_root, monkeypatch):
    site, paths, processed = pipeline_root
    use_source(monkeypatch, FakeSource({'11': 30, '12': 60}))

    run_mvp.run_site(site, paths, make_args(date='2026-11-24'))

    assert processed == ['12']
    skipped = paths['root'] / 'state' / 'testsite' / '2026-11-24_11.skipped'
    assert json.loads(skipped.read_text())['minutes'] == 30


def test_hours_with_a_marker_are_left_alone(pipeline_root, monkeypatch):
    site, paths, processed = pipeline_root
    use_source(monkeypatch, FakeSource({'12': 60, '13': 60, '14': 60}))
    state_dir = paths['root'] / 'state' / 'testsite'
    state_dir.mkdir(parents=True)
    (state_dir / '2026-11-24_12.done').touch()
    (state_dir / '2026-11-24_13.failed').touch()

    run_mvp.run_site(site, paths, make_args(date='2026-11-24'))

    assert processed == ['14']


def test_hour_still_being_recorded_waits(pipeline_root, monkeypatch):
    site, paths, processed = pipeline_root
    use_source(monkeypatch, FakeSource({'16': 60, '17': 60}))
    # 17:00 to 18:00 has ended, but not the 10 minute grace period after it
    monkeypatch.setattr(run_mvp, 'now_utc', lambda: datetime.datetime(2026, 11, 24, 18, 5))

    run_mvp.run_site(site, paths, make_args(date='2026-11-24'))

    assert processed == ['16']
    assert not (paths['root'] / 'state' / 'testsite' / '2026-11-24_17.done').exists()


def test_max_hours_per_run(pipeline_root, monkeypatch):
    site, paths, processed = pipeline_root
    use_source(monkeypatch, FakeSource({'12': 60, '13': 60, '14': 60}))
    site['max_hours_per_run'] = 2

    run_mvp.run_site(site, paths, make_args(date='2026-11-24'))
    assert processed == ['12', '13']

    # The next run carries on where this one stopped
    run_mvp.run_site(site, paths, make_args(date='2026-11-24'))
    assert processed == ['12', '13', '14']


def test_dry_run_writes_nothing(pipeline_root, monkeypatch):
    site, paths, processed = pipeline_root
    use_source(monkeypatch, FakeSource({'11': 30, '12': 60}))

    run_mvp.run_site(site, paths, make_args(dry_run=True, date='2026-11-24'))

    assert processed == []
    assert not (paths['root'] / 'state').exists()
    assert not (paths['root'] / 'status.log').exists()


def test_failed_hour_is_marked_and_the_next_hour_still_runs(pipeline_root, monkeypatch):
    site, paths, processed = pipeline_root
    use_source(monkeypatch, FakeSource({'12': 60, '13': 60}))

    def process_hour(site, source, date, hour, paths):
        processed.append(hour)
        if hour == '12':
            raise run_mvp.ProcessingError("DOAS results are empty")
        return {'files': 10}

    monkeypatch.setattr(run_mvp, 'process_hour', process_hour)

    updated = run_mvp.run_site(site, paths, make_args(date='2026-11-24'))

    assert processed == ['12', '13']
    assert updated == {DATE}
    failed = paths['root'] / 'state' / 'testsite' / '2026-11-24_12.failed'
    assert json.loads(failed.read_text())['error'] == "DOAS results are empty"
    assert 'FAILED' in (paths['root'] / 'status.log').read_text()


def test_unreachable_source_skips_the_site(pipeline_root, monkeypatch):
    site, paths, processed = pipeline_root
    use_source(monkeypatch, FakeSource({'12': 60}, problem="cannot connect to pi@host"))

    updated = run_mvp.run_site(site, paths, make_args(date='2026-11-24'))

    assert processed == []
    assert updated == set()
    assert not (paths['root'] / 'state').exists()
    assert 'cannot connect' in (paths['root'] / 'status.log').read_text()


def test_dates_to_check_uses_lookback_days(monkeypatch):
    monkeypatch.setattr(run_mvp, 'now_utc', lambda: datetime.datetime(2026, 11, 24, 18, 0))

    assert run_mvp.dates_to_check({'lookback_days': 2}, None) == [datetime.date(2026, 11, 23), DATE]
    assert run_mvp.dates_to_check({}, None) == [DATE]
    assert run_mvp.dates_to_check({'lookback_days': 2}, '2026-11-01') == [datetime.date(2026, 11, 1)]


def test_data_minutes():
    names = image_names(DATE, '12', 50) + ['notes.txt', '2026-11-24T120000_spec.json']
    assert run_mvp.data_minutes(names, ['.png', '.npy']) == 50
    assert run_mvp.data_minutes(['2026-11-24T120000_fltrA.png'], ['.png']) == 0


# Template check and run config

def test_template_check_passes_when_all_files_exist(tmp_path):
    site = {'template': str(make_template(tmp_path))}
    assert run_mvp.check_template(site, tmp_path / 'root', tmp_path / 'code') is None


def test_template_check_reports_missing_files(tmp_path):
    site = {'template': str(make_template(tmp_path, ILS_path='data/no_such_ils.txt'))}
    problem = run_mvp.check_template(site, tmp_path / 'root', tmp_path / 'code')
    assert 'ILS_path' in problem
    assert 'no_such_ils.txt' in problem


def test_template_check_reports_missing_species_file(tmp_path):
    species = {'SO2': {'path': 'data/so2.txt'}, 'O3': {'path': 'data/missing_o3.txt'}}
    site = {'template': str(make_template(tmp_path, species_paths=species))}
    problem = run_mvp.check_template(site, tmp_path / 'root', tmp_path / 'code')
    assert 'missing_o3.txt' in problem


def test_cal_series_path_only_needed_for_cal_type_3(tmp_path):
    site = {'template': str(make_template(tmp_path, cal_series_path='data/no_such_series.csv'))}
    assert run_mvp.check_template(site, tmp_path / 'root', tmp_path / 'code') is None

    site = {'template': str(make_template(tmp_path, cal_type_int=3,
                                          cal_series_path='data/no_such_series.csv'))}
    problem = run_mvp.check_template(site, tmp_path / 'root', tmp_path / 'code')
    assert 'cal_series_path' in problem


def test_template_check_reports_empty_template(tmp_path):
    template = tmp_path / 'template.yml'
    template.write_text('')
    problem = run_mvp.check_template({'template': str(template)}, tmp_path / 'root', tmp_path / 'code')
    assert 'cannot be read' in problem


def test_run_config_points_data_folders_at_the_hour(tmp_path):
    site = {'template': str(make_template(tmp_path)), 'overrides': {'img_buff_size': 500}}
    raw_dir = tmp_path / 'work' / 'raw'

    config = run_mvp.build_run_config(site, raw_dir, tmp_path / 'code')

    for key in ['img_dir', 'spec_dir', 'transfer_dir']:
        assert config[key] == str(raw_dir)
    assert config['img_buff_size'] == 500


def test_run_config_makes_paths_absolute(tmp_path):
    site = {'template': str(make_template(tmp_path))}
    config = run_mvp.build_run_config(site, tmp_path / 'raw', tmp_path / 'code')

    assert config['ILS_path'] == str((tmp_path / 'data' / 'ils.txt').resolve())
    assert config['pcs_lines'] == [str((tmp_path / 'data' / 'line_0.txt').resolve())]
    assert config['species_paths']['SO2']['path'] == str((tmp_path / 'data' / 'so2.txt').resolve())


def test_relative_paths_prefer_the_code_folder(tmp_path):
    # PyCam looks in its own folder first, so a file there wins over one next to the template
    site = {'template': str(make_template(tmp_path / 'template'))}
    code_dir = tmp_path / 'code'
    (code_dir / 'data').mkdir(parents=True)
    (code_dir / 'data' / 'ils.txt').touch()

    config = run_mvp.build_run_config(site, tmp_path / 'raw', code_dir)

    assert config['ILS_path'] == str((code_dir / 'data' / 'ils.txt').resolve())
    assert config['ld_lookup_1'] == str((tmp_path / 'template' / 'data' / 'ld1.npy').resolve())


# Checking the PyCam output

def test_combine_doas_results(tmp_path):
    doas_dir = tmp_path / 'Processed_spec_1'
    doas_dir.mkdir()
    (doas_dir / 'a.csv').write_text('time,column\n1,10\n2,20\n')
    (doas_dir / 'b.csv').write_text('time,column\n3,30\n')

    rows = run_mvp.combine_doas_results(doas_dir, tmp_path / 'all.csv')

    assert rows == 3
    assert len((tmp_path / 'all.csv').read_text().splitlines()) == 4


def test_combine_doas_results_fails_without_results(tmp_path):
    doas_dir = tmp_path / 'Processed_spec_1'
    doas_dir.mkdir()
    with pytest.raises(run_mvp.ProcessingError):
        run_mvp.combine_doas_results(doas_dir, tmp_path / 'all.csv')

    (doas_dir / 'a.csv').write_text('time,column\n')
    with pytest.raises(run_mvp.ProcessingError):
        run_mvp.combine_doas_results(doas_dir, tmp_path / 'all.csv')


def test_check_emission_files(tmp_path):
    processed = make_processed_hour(tmp_path, 12, flux=1.5, r2_values=[0.9, 0.9, 0.9])
    n_files, rows_line0 = run_mvp.check_emission_files(processed)
    assert n_files == 3
    assert rows_line0 == 3


def test_empty_emission_file_is_a_failure(tmp_path):
    processed = make_processed_hour(tmp_path, 12, flux=1.5, r2_values=[0.9, 0.9, 0.9])
    write_emission_file(processed / 'line_0' / '20261124_EmissionRates_flow_histo.txt', [], 1.5)

    with pytest.raises(run_mvp.ProcessingError, match='flow_histo'):
        run_mvp.check_emission_files(processed)


def test_missing_line_0_is_a_failure(tmp_path):
    with pytest.raises(run_mvp.ProcessingError, match='line_0'):
        run_mvp.check_emission_files(tmp_path)


# Dashboard JSON

def test_build_day_merges_hours(tmp_path):
    hour_12 = make_processed_hour(tmp_path / '12', 12, flux=1.0, r2_values=[0.9, 0.9, 0.9])
    hour_13 = make_processed_hour(tmp_path / '13', 13, flux=3.0, r2_values=[0.9, 0.9, 0.9])

    day = web_data.build_day([hour_12, hour_13], DATE)

    assert day['date'] == '2026-11-24'
    # 2 hours x 3 times x 3 flow types
    assert len(day['time_series_data']) == 18
    histo = day['summary_stats']['all_data']['histo']
    assert histo['min'] == 1.0
    assert histo['max'] == 3.0
    assert histo['mean'] == 2.0


def test_build_day_r2_quality(tmp_path):
    processed = make_processed_hour(tmp_path, 12, flux=1.0, r2_values=[0.9, 0.4, ''])

    day = web_data.build_day([processed], DATE)

    glob_records = [r for r in day['time_series_data'] if r['type'] == 'glob']
    assert [r['r2_quality'] for r in glob_records] == ['Excellent', 'Acceptable', 'nan']
    assert glob_records[2]['r2'] is None
    # Only the Excellent point counts as good data
    assert day['summary_stats']['good_data']['glob']['mean'] == 1.0


def test_build_day_without_data_returns_none(tmp_path):
    assert web_data.build_day([], DATE) is None
    assert web_data.build_day([tmp_path], DATE) is None


def test_build_day_output_is_valid_json(tmp_path):
    processed = make_processed_hour(tmp_path, 12, flux=1.0, r2_values=['', '', ''])
    day = web_data.build_day([processed], DATE)
    # allow_nan=False fails on NaN or inf, which browsers can't parse
    json.dumps(day, allow_nan=False)


def test_update_web_data_only_uses_done_hours(tmp_path):
    root = tmp_path
    site = {'name': 'testsite'}
    day_dir = root / 'results' / 'testsite' / '2026-11-24'
    make_processed_hour(day_dir / '12', 12, flux=1.0, r2_values=[0.9, 0.9, 0.9])
    make_processed_hour(day_dir / '13', 13, flux=100.0, r2_values=[0.9, 0.9, 0.9])
    state_dir = root / 'state' / 'testsite'
    state_dir.mkdir(parents=True)
    (state_dir / '2026-11-24_12.done').touch()
    (state_dir / '2026-11-24_13.failed').touch()

    web_file = run_mvp.update_web_data(site, DATE, root)

    day = json.loads(web_file.read_text())
    assert web_file.name == 'testsite_data_2026-11-24.json'
    assert day['summary_stats']['all_data']['histo']['max'] == 1.0
    assert json.loads((root / 'web' / 'testsite_dates.json').read_text()) == ['2026-11-24']


# Publishing to the website folder

def make_day_file(folder, date='2026-11-24'):
    folder.mkdir(parents=True, exist_ok=True)
    day_file = folder / f'testsite_data_{date}.json'
    day_file.write_text(json.dumps({'date': date, 'time_series_data': []}))
    return day_file


def test_publish_copies_the_day_and_keeps_old_dates(tmp_path):
    website = tmp_path / 'website'
    website.mkdir()
    (website / 'testsite_dates.json').write_text(json.dumps(['2025-12-06', '2025-12-05']))
    (website / 'testsite_data_2025-12-06.json').write_text('old file')
    day_file = make_day_file(tmp_path / 'web')

    published = web_data.publish_day(day_file, website, 'testsite', tmp_path / 'backup')

    assert published == website / 'testsite_data_2026-11-24.json'
    assert json.loads(published.read_text())['date'] == '2026-11-24'
    dates = json.loads((website / 'testsite_dates.json').read_text())
    assert dates == ['2026-11-24', '2025-12-06', '2025-12-05']
    assert (website / 'testsite_data_2025-12-06.json').read_text() == 'old file'


def test_publish_twice_adds_no_duplicate(tmp_path):
    website = tmp_path / 'website'
    website.mkdir()
    day_file = make_day_file(tmp_path / 'web')

    web_data.publish_day(day_file, website, 'testsite', tmp_path / 'backup')
    web_data.publish_day(day_file, website, 'testsite', tmp_path / 'backup')

    assert json.loads((website / 'testsite_dates.json').read_text()) == ['2026-11-24']


def test_publish_backs_up_the_original_date_list_once(tmp_path):
    website = tmp_path / 'website'
    website.mkdir()
    (website / 'testsite_dates.json').write_text(json.dumps(['2025-12-06']))
    backup = tmp_path / 'backup' / 'testsite_dates.original.json'

    web_data.publish_day(make_day_file(tmp_path / 'web', '2026-11-24'), website, 'testsite', tmp_path / 'backup')
    assert json.loads(backup.read_text()) == ['2025-12-06']

    # A later publish must not overwrite the original with our changed list
    web_data.publish_day(make_day_file(tmp_path / 'web', '2026-11-25'), website, 'testsite', tmp_path / 'backup')
    assert json.loads(backup.read_text()) == ['2025-12-06']


def test_publish_to_missing_folder_raises(tmp_path):
    day_file = make_day_file(tmp_path / 'web')
    with pytest.raises(FileNotFoundError):
        web_data.publish_day(day_file, tmp_path / 'no_such_website', 'testsite', tmp_path / 'backup')


def test_update_website_dry_run_publishes_nothing(tmp_path, monkeypatch):
    website = tmp_path / 'website'
    website.mkdir()
    site = {'name': 'testsite', 'publish_dir': str(website)}
    monkeypatch.setattr(run_mvp, 'update_web_data', lambda site, date, root: pytest.fail("should not run"))

    run_mvp.update_website(site, {DATE}, tmp_path, dry_run=True)

    assert list(website.iterdir()) == []


def test_update_website_failure_is_logged_not_raised(tmp_path):
    site = {'name': 'testsite', 'publish_dir': str(tmp_path / 'no_such_website')}
    make_processed_hour(tmp_path / 'results' / 'testsite' / '2026-11-24' / '12', 12, 1.0, [0.9, 0.9, 0.9])
    (tmp_path / 'state' / 'testsite').mkdir(parents=True)
    (tmp_path / 'state' / 'testsite' / '2026-11-24_12.done').touch()

    run_mvp.update_website(site, {DATE}, tmp_path, dry_run=False)

    assert 'web data FAILED' in (tmp_path / 'status.log').read_text()


# Lock file

def test_second_run_cannot_take_the_lock(tmp_path):
    lock_file = tmp_path / 'run.lock'
    assert run_mvp.take_lock(lock_file)
    assert not run_mvp.take_lock(lock_file)


def test_old_lock_is_removed(tmp_path):
    lock_file = tmp_path / 'run.lock'
    lock_file.write_text('left behind by a crashed run')
    seven_hours_ago = time.time() - 7 * 3600
    os.utime(lock_file, (seven_hours_ago, seven_hours_ago))

    assert run_mvp.take_lock(lock_file)
    assert 'pid' in lock_file.read_text()


# Site files

def test_load_sites_ignores_templates_and_disabled_sites(tmp_path):
    sites_dir = tmp_path / 'sites'
    sites_dir.mkdir()
    (sites_dir / 'a.yml').write_text(yaml.safe_dump({'name': 'a', 'source': {'type': 'local'}}))
    (sites_dir / 'b.yml').write_text(yaml.safe_dump({'name': 'b', 'source': {}, 'enabled': False}))
    (sites_dir / 'a_template.yml').write_text(yaml.safe_dump({'img_dir': './data'}))

    assert [s['name'] for s in run_mvp.load_sites(tmp_path, None)] == ['a']
    assert run_mvp.load_sites(tmp_path, 'other') == []
