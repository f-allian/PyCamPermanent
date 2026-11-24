#!/usr/bin/env python3
"""
Automated headless processing of PyCam data, one hour at a time.

Meant to be run by cron every 15 minutes. For every site file in <root>/sites/ it finds
finished hours that haven't been processed yet, copies the raw data, runs
process_emission_rates.py (doas, then pyplis), checks the output and updates the
dashboard JSON for that day. See pipeline/sites/example_site.yml for the site settings.

Folders under --root:
    sites/      one .yml file per site
    work/       raw data of the hour being processed (deleted afterwards)
    results/    <site>/<date>/<hour>/ with the DOAS and pyplis output
    state/      <site>/<date>_<hour>.done, .failed or .skipped
    logs/       <site>/<date>_<hour>.log with the full PyCam output
    web/        dashboard JSON files
    status.log  one line per processed or skipped hour

Examples:
    python pipeline/run_mvp.py --root /root/pycam_mvp
    python pipeline/run_mvp.py --root /root/pycam_mvp --dry-run
    python pipeline/run_mvp.py --root /root/pycam_mvp --date 2026-11-24 --max-hours 1
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from sources import SourceError, make_source
from web_data import build_day, write_day

# Config keys that hold file or folder paths (same list as path_params in so2_camera_processor.py)
PATH_KEYS = ['img_dir', 'dark_img_dir', 'transfer_dir', 'spec_dir', 'dark_spec_dir', 'bg_A_path', 'bg_B_path',
             'cell_cal_dir', 'pcs_lines', 'img_registration', 'dil_lines', 'ld_lookup_1', 'ld_lookup_2',
             'ILS_path', 'default_cam_geom', 'cal_series_path', 'species_paths']

# These point at the raw data and are set to the hour's work folder for every run
DATA_KEYS = ['img_dir', 'spec_dir', 'transfer_dir']

STEP_TIMEOUT = 3 * 60 * 60   # seconds allowed for one doas or pyplis run
STALE_LOCK_HOURS = 6


class ProcessingError(Exception):
    pass


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def log(message):
    print(f"{now_utc():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def take_lock(lock_file):
    """Create the lock file so two cron runs never process at the same time."""
    if lock_file.exists():
        age_hours = (time.time() - lock_file.stat().st_mtime) / 3600
        if age_hours < STALE_LOCK_HOURS:
            return False
        log(f"Removing old lock file ({age_hours:.1f} hours old)")
        lock_file.unlink()

    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        # O_EXCL makes this fail if another run created the file in the meantime
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, 'w') as f:
        f.write(f"pid {os.getpid()}, started {now_utc():%Y-%m-%d %H:%M:%S}\n")
    return True


def make_absolute(path, template_dir, code_dir):
    """
    Resolve a relative config path the same way PyCam does: relative to the code folder if
    it exists there, otherwise relative to the template's folder.
    """
    if not isinstance(path, str) or path.strip() == '':
        return path
    path = path.strip()
    if os.path.isabs(path):
        return path
    if (code_dir / path).exists():
        return str((code_dir / path).resolve())
    return str((template_dir / path).resolve())


def build_run_config(site, raw_dir, code_dir):
    """Load the site's template, make every path absolute and point the data folders at raw_dir."""
    template = Path(site['template']).resolve()
    with open(template) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ProcessingError(f"Template is empty or not valid YAML: {template}")

    config.update(site.get('overrides') or {})

    for key in PATH_KEYS:
        if key not in config or key in DATA_KEYS:
            continue
        value = config[key]
        if key == 'species_paths':
            for species in value.values():
                species['path'] = make_absolute(species['path'], template.parent, code_dir)
        elif isinstance(value, list):
            config[key] = [make_absolute(v, template.parent, code_dir) for v in value]
        else:
            config[key] = make_absolute(value, template.parent, code_dir)

    for key in DATA_KEYS:
        config[key] = str(raw_dir)
    return config


def find_missing_paths(config):
    """List the path settings in a run config that are not set or don't exist."""
    missing = []
    for key in PATH_KEYS:
        if key in DATA_KEYS:
            continue
        # cal_series_path is only used with cal_type_int 3
        if key == 'cal_series_path' and config.get('cal_type_int') != 3:
            continue
        if key not in config:
            missing.append(f"{key} (not set)")
            continue

        value = config[key]
        if key == 'species_paths':
            paths = [species['path'] for species in value.values()]
        elif isinstance(value, list):
            paths = value
        else:
            paths = [value]

        for path in paths:
            if not path or not os.path.exists(path):
                missing.append(f"{key}: {path}")
    return missing


def check_template(site, root, code_dir):
    """Check the site's template before any data is copied. Returns an error message or None."""
    try:
        config = build_run_config(site, root / 'work', code_dir)
    except (OSError, yaml.YAMLError, KeyError, AttributeError, ProcessingError) as e:
        return f"template cannot be read: {e}"
    missing = find_missing_paths(config)
    if missing:
        return "template paths not found: " + "; ".join(missing)
    return None


def data_minutes(file_names, extensions):
    """Minutes between the first and last data file, from the time in the file names."""
    times = []
    for name in file_names:
        if Path(name).suffix.lower() not in extensions:
            continue
        match = re.match(r'^(\d{4}-\d{2}-\d{2}T\d{6})', name)
        if match:
            times.append(datetime.datetime.strptime(match.group(1), '%Y-%m-%dT%H%M%S'))
    if len(times) < 2:
        return 0.0
    return (max(times) - min(times)).total_seconds() / 60


def run_pycam(args, code_dir, env, log_file, step):
    log_file.write(f"\n===== {step}: {' '.join(args)}\n")
    log_file.flush()
    try:
        result = subprocess.run(args, cwd=code_dir, env=env, stdout=log_file, stderr=subprocess.STDOUT,
                                timeout=STEP_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise ProcessingError(f"{step} took longer than {STEP_TIMEOUT} s")
    if result.returncode != 0:
        raise ProcessingError(f"{step} exited with code {result.returncode} (see log)")


def newest_folder(parent, prefix):
    folders = sorted(p for p in parent.glob(prefix + '*') if p.is_dir())
    if not folders:
        return None
    return folders[-1]


def combine_doas_results(doas_dir, combined_file):
    """Join all DOAS result CSVs into one file for pyplis. Returns the number of rows."""
    csv_files = sorted(doas_dir.glob('*.csv'))
    if not csv_files:
        raise ProcessingError(f"DOAS wrote no results in {doas_dir}")
    results = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    if results.empty:
        raise ProcessingError("DOAS results are empty")
    results.to_csv(combined_file, index=False)
    return len(results)


def count_rows(emission_file):
    with open(emission_file) as f:
        return sum(1 for line in f if re.match(r'^\d{4}-\d{2}-\d{2}', line))


def check_emission_files(processed_dir):
    """Every EmissionRates file must have data. Returns (number of files, rows on line_0)."""
    if not (processed_dir / 'line_0').is_dir():
        raise ProcessingError(f"pyplis wrote no line_0 folder in {processed_dir}")

    files = sorted(processed_dir.glob('line_*/*_EmissionRates_*.txt'))
    if not files:
        raise ProcessingError(f"pyplis wrote no emission rate files in {processed_dir}")

    empty = [f.name for f in files if count_rows(f) == 0]
    if empty:
        raise ProcessingError(f"empty emission rate files: {', '.join(empty)}")

    rows_line0 = max(count_rows(f) for f in (processed_dir / 'line_0').glob('*_EmissionRates_*.txt'))
    return len(files), rows_line0


def process_hour(site, source, date, hour, paths):
    """Copy, process and check one hour. Returns a dict of numbers for status.log."""
    tag = f"{date}_{hour}"
    work_dir = paths['root'] / 'work' / site['name'] / tag
    raw_dir = work_dir / 'raw'
    results_dir = paths['root'] / 'results' / site['name'] / str(date) / hour
    log_path = paths['root'] / 'logs' / site['name'] / f"{tag}.log"

    if work_dir.exists():
        shutil.rmtree(work_dir)   # left over from an interrupted run
    raw_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    extensions = site_extensions(site)
    env = os.environ.copy()
    for key, value in (site.get('env') or {}).items():
        env[key] = str(value)

    script = str(paths['code'] / 'process_emission_rates.py')
    python = paths['python']

    with open(log_path, 'a') as log_file:
        log_file.write(f"\n##### {now_utc():%Y-%m-%d %H:%M:%S} {site['name']} {tag}\n")

        source.fetch(date, hour, raw_dir, extensions)
        n_files = len(list(raw_dir.iterdir()))
        if n_files == 0:
            raise ProcessingError("no files were copied")

        config_file = work_dir / 'run_config.yml'
        with open(config_file, 'w') as f:
            yaml.safe_dump(build_run_config(site, raw_dir, paths['code']), f, sort_keys=False)

        run_pycam([python, script, 'doas', '--config_path', str(config_file)],
                  paths['code'], env, log_file, 'doas')
        doas_dir = newest_folder(raw_dir, 'Processed_spec_')
        if doas_dir is None:
            raise ProcessingError("DOAS wrote no Processed_spec_ folder")
        doas_file = work_dir / 'doas_results.csv'
        doas_rows = combine_doas_results(doas_dir, doas_file)

        run_pycam([python, script, 'pyplis', '--config_path', str(config_file),
                   '--doas_results', str(doas_file), '--output_directory', str(results_dir)],
                  paths['code'], env, log_file, 'pyplis')
        # 'Processed_2' matches Processed_<time> but not Processed_spec_<time>
        pyplis_dir = newest_folder(results_dir, 'Processed_2')
        if pyplis_dir is None:
            raise ProcessingError("pyplis wrote no Processed_ folder")
        n_emission_files, rows_line0 = check_emission_files(pyplis_dir)

        shutil.move(str(doas_dir), str(results_dir / doas_dir.name))

    shutil.rmtree(work_dir)
    return {'files': n_files, 'doas_rows': doas_rows, 'emission_files': n_emission_files,
            'rows_line0': rows_line0}


def site_extensions(site):
    return [ext.lower() for ext in site.get('extensions', ['.png', '.npy'])]


def dates_to_check(site, date_arg):
    if date_arg:
        return [datetime.date.fromisoformat(date_arg)]
    today = now_utc().date()
    days = site.get('lookback_days', 1)
    return [today - datetime.timedelta(days=n) for n in reversed(range(days))]


def marker_exists(state_dir, tag):
    for ending in ['.done', '.failed', '.skipped']:
        if (state_dir / (tag + ending)).exists():
            return True
    return False


def write_marker(state_dir, tag, ending, info):
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / (tag + ending), 'w') as f:
        json.dump(info, f, indent=2)


def write_status(root, text):
    with open(root / 'status.log', 'a') as f:
        f.write(f"{now_utc():%Y-%m-%d %H:%M:%S} {text}\n")


def hour_is_finished(date, hour, grace_minutes):
    hour_end = datetime.datetime.combine(date, datetime.time(int(hour))) + datetime.timedelta(hours=1)
    return now_utc() >= hour_end + datetime.timedelta(minutes=grace_minutes)


def run_site(site, paths, args):
    """Process the waiting hours of one site. Returns the dates that got new results."""
    name = site['name']
    root = paths['root']
    state_dir = root / 'state' / name
    min_minutes = site.get('min_minutes', 35)
    grace_minutes = site.get('grace_minutes', 10)
    max_hours = args.max_hours or site.get('max_hours_per_run', 1)

    problem = check_template(site, root, paths['code'])
    if problem is None:
        source = make_source(site['source'])
        problem = source.check()
    if problem:
        # Nothing copied and no markers written, so the hours are picked up once this is fixed
        log(f"[{name}] skipped: {problem}")
        if not args.dry_run:
            write_status(root, f"{name} skipped: {problem}")
        return set()

    log(f"[{name}] template OK, data from {source.describe()}")
    updated_dates = set()
    hours_processed = 0

    for date in dates_to_check(site, args.date):
        try:
            hours = source.list_hours(date)
        except SourceError as e:
            log(f"[{name}] cannot list {date}: {e}")
            continue
        if not hours:
            log(f"[{name}] no data for {date}")

        for hour in hours:
            tag = f"{date}_{hour}"
            if marker_exists(state_dir, tag):
                if args.dry_run:
                    log(f"[{name}] {tag} already handled (has a marker in {state_dir})")
                continue
            if not hour_is_finished(date, hour, grace_minutes):
                log(f"[{name}] {tag} still being recorded")
                continue
            if hours_processed >= max_hours:
                log(f"[{name}] {tag} waiting (max {max_hours} hour(s) per run)")
                continue

            try:
                minutes = data_minutes(source.list_files(date, hour), site_extensions(site))
            except SourceError as e:
                log(f"[{name}] cannot list {tag}: {e}")
                continue

            if minutes < min_minutes:
                log(f"[{name}] {tag} skip: {minutes:.1f} min of data (minimum {min_minutes})")
                if not args.dry_run:
                    write_marker(state_dir, tag, '.skipped', {'minutes': round(minutes, 1)})
                    write_status(root, f"{name} {tag} skipped, {minutes:.1f} min of data")
                continue

            if args.dry_run:
                log(f"[{name}] {tag} would be processed ({minutes:.1f} min of data)")
                hours_processed += 1
                continue

            log(f"[{name}] {tag} processing ({minutes:.1f} min of data)")
            start = time.time()
            try:
                result = process_hour(site, source, date, hour, paths)
                error = None
            except (ProcessingError, SourceError) as e:
                result, error = None, str(e)
            except Exception as e:
                # Unexpected error: record it and carry on with the next hour
                result, error = None, f"{type(e).__name__}: {e}"
            seconds = int(time.time() - start)
            hours_processed += 1

            if error:
                # Keep the run config and log for debugging, but not the raw copy (the Pi keeps that)
                shutil.rmtree(root / 'work' / name / tag / 'raw', ignore_errors=True)
                write_marker(state_dir, tag, '.failed', {'error': error, 'seconds': seconds})
                write_status(root, f"{name} {tag} FAILED after {seconds} s: {error}")
                log(f"[{name}] {tag} FAILED: {error}")
            else:
                write_marker(state_dir, tag, '.done', dict(result, seconds=seconds))
                numbers = ', '.join(f"{key} {value}" for key, value in result.items())
                write_status(root, f"{name} {tag} ok in {seconds} s ({numbers})")
                log(f"[{name}] {tag} ok in {seconds} s ({numbers})")
                updated_dates.add(date)

    return updated_dates


def update_web_data(site, date, root):
    """Rebuild the dashboard JSON for one day from all hours that finished successfully."""
    state_dir = root / 'state' / site['name']
    day_dir = root / 'results' / site['name'] / str(date)

    processed_dirs = []
    for hour_dir in sorted(day_dir.glob('[0-9][0-9]')):
        # A failed hour can leave partial output behind, so only use hours marked done
        if not (state_dir / f"{date}_{hour_dir.name}.done").exists():
            continue
        pyplis_dir = newest_folder(hour_dir, 'Processed_2')
        if pyplis_dir is not None:
            processed_dirs.append(pyplis_dir)

    day_data = build_day(processed_dirs, date)
    if day_data is None:
        return None
    return write_day(root / 'web', site.get('web_prefix', site['name']), day_data)


def load_sites(root, only_site):
    sites = []
    for path in sorted((root / 'sites').glob('*.yml')):
        with open(path) as f:
            site = yaml.safe_load(f)
        # Templates live in sites/ too; site files are the ones with a 'name' and a 'source'
        if not isinstance(site, dict) or 'name' not in site or 'source' not in site:
            continue
        if not site.get('enabled', True):
            continue
        if only_site and site['name'] != only_site:
            continue
        sites.append(site)
    return sites


def main():
    parser = argparse.ArgumentParser(description="Automated headless PyCam processing, one hour at a time")
    parser.add_argument('--root', required=True, help="pipeline folder with sites/, results/, state/ etc.")
    parser.add_argument('--site', help="only process this site")
    parser.add_argument('--date', help="process this UTC date (YYYY-MM-DD) instead of the last lookback_days")
    parser.add_argument('--max-hours', type=int, help="maximum hours to process per site in this run")
    parser.add_argument('--dry-run', action='store_true',
                        help="check templates and list what would be processed, without copying anything")
    parser.add_argument('--code-dir', default=str(Path(__file__).resolve().parent.parent),
                        help="PyCamPermanent folder with process_emission_rates.py")
    parser.add_argument('--python', default=sys.executable, help="python of the PyCam environment")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = {'root': root, 'code': Path(args.code_dir).resolve(), 'python': args.python}

    lock_file = root / 'run.lock'
    if not take_lock(lock_file):
        log("Another run is still going, exiting")
        return
    try:
        sites = load_sites(root, args.site)
        if not sites:
            log(f"No site files in {root / 'sites'}")
        for site in sites:
            updated_dates = run_site(site, paths, args)
            for date in sorted(updated_dates):
                try:
                    web_file = update_web_data(site, date, root)
                except Exception as e:
                    write_status(root, f"{site['name']} {date} web data FAILED: {e}")
                    log(f"[{site['name']}] web data for {date} failed: {e}")
                    continue
                if web_file:
                    log(f"[{site['name']}] web data updated: {web_file}")
    finally:
        lock_file.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
