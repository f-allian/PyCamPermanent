"""
Where the raw data for a site comes from.

The Pi stores one folder per day with one sub-folder per hour (00 to 23). Both sources
below work with that layout: LocalSource reads a folder on this machine (used for
testing), SshSource reads the Pi over SSH and copies with rsync.
"""
import re
import shutil
import subprocess
from pathlib import Path


class SourceError(Exception):
    pass


def day_folder(day_path, date):
    """Fill in the {date}, {yyyy}, {mm} and {dd} placeholders of a site's day_path."""
    return day_path.format(date=date.isoformat(),
                           yyyy=date.strftime('%Y'),
                           mm=date.strftime('%m'),
                           dd=date.strftime('%d'))


def is_hour_name(name):
    return re.fullmatch(r'\d{2}', name) is not None


class LocalSource:
    def __init__(self, settings):
        self.day_path = settings['day_path']

    def describe(self):
        return f"local folder {self.day_path}"

    def check(self):
        """Nothing to check for a local folder."""
        return None

    def list_hours(self, date):
        folder = Path(day_folder(self.day_path, date))
        if not folder.is_dir():
            return []
        hours = [p.name for p in folder.iterdir() if p.is_dir() and is_hour_name(p.name)]
        return sorted(hours)

    def list_files(self, date, hour):
        folder = Path(day_folder(self.day_path, date)) / hour
        return sorted(p.name for p in folder.iterdir() if p.is_file())

    def fetch(self, date, hour, dest, extensions):
        folder = Path(day_folder(self.day_path, date)) / hour
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in extensions:
                shutil.copy2(path, dest / path.name)


class SshSource:
    def __init__(self, settings):
        self.host = settings['host']
        self.day_path = settings['day_path']
        self.bwlimit = settings.get('bwlimit_kbps')
        self.timeout = settings.get('timeout_s', 3600)

    def describe(self):
        return f"{self.host}:{self.day_path}"

    def _ssh(self, command):
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', self.host, command]
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise SourceError(f"ssh to {self.host} timed out")

    def check(self):
        """Return an error message if we can't reach the host or it has no rsync."""
        result = self._ssh('command -v rsync')
        if result.returncode == 255:
            return f"cannot connect to {self.host}: {result.stderr.strip()}"
        if result.returncode != 0:
            return f"rsync is not installed on {self.host}"
        return None

    def _ls(self, folder):
        # ls returns 2 when the folder doesn't exist, which just means "no data yet"
        result = self._ssh(f"ls -1 '{folder}'")
        if result.returncode == 2:
            return []
        if result.returncode != 0:
            raise SourceError(f"listing {folder} failed: {result.stderr.strip()}")
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def list_hours(self, date):
        names = self._ls(day_folder(self.day_path, date))
        return sorted(name for name in names if is_hour_name(name))

    def list_files(self, date, hour):
        return self._ls(f"{day_folder(self.day_path, date)}/{hour}")

    def fetch(self, date, hour, dest, extensions):
        remote = f"{self.host}:{day_folder(self.day_path, date)}/{hour}/"
        cmd = ['rsync', '-a', '--partial', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=30']
        if self.bwlimit:
            cmd.append(f'--bwlimit={self.bwlimit}')
        for ext in sorted(extensions):
            cmd.append(f'--include=*{ext}')
        cmd += ['--exclude=*', remote, f'{dest}/']

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise SourceError(f"rsync timed out after {self.timeout} s")
        if result.returncode != 0:
            raise SourceError(f"rsync failed: {result.stderr.strip()}")


def make_source(settings):
    if settings['type'] == 'local':
        return LocalSource(settings)
    if settings['type'] == 'ssh':
        return SshSource(settings)
    raise ValueError(f"Unknown source type '{settings['type']}', use 'local' or 'ssh'")
