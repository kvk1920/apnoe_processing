from clrprint import clrprint
import contextlib
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import typing as tp
from scipy.integrate import simps
from scipy.signal import welch
import scipy.stats as sps
import sys
from tqdm import tqdm
from sortedcontainers.sortedset import SortedSet
from operator import itemgetter


################################################################################
# Configuration #
################################################################################
# NOTE: This script should be used from terminal or IDLE (but not via
#       double-click) Apnoe file must be named as Apnoe.txt, and neuron files
#       must be named as "Neuron*.txt".
# NOTE: All durations are measured in seconds.
################################################################################
# Bootstrap #
################################################################################
# Duration of left apnoe part.
APNOE_LEFT_DURATION: float = 5.0
APNOE_RIGHT_DURATION: float = 5.0
# Shift of sliding window which is used in frequency curve calculating (in
# seconds).
SLIDING_WINDOW_SHIFT: float = 0.001
# Duration of sliding window for frequency curve calculating (in seconds).
SLIDING_WINDOW_DURATION: float = 1.0
# Minimal allowed time between different elements of bootstrap sample (in
# seconds).
BOOTSTRAP_MIN_CLOSENESS: float = 1.0
# Bootstrap sample size.
BOOTSTRAP_SIZE: int = 10000
# Neediness of plot of found min/max frequency.
MINMAX_PLOT_IS_NEEDED = False
# Total apnoe duration.
APNOE_DURATION = APNOE_LEFT_DURATION + APNOE_RIGHT_DURATION
# Number of points in neuron frequency curve.
CURVE_SIZE = int(round((APNOE_DURATION - SLIDING_WINDOW_DURATION) /
                       SLIDING_WINDOW_SHIFT))
################################################################################
# File names #
################################################################################
APNOE_FILE_PATTERN = '* Apnoe.txt'
NEURON_FILE_PATTERNS = ['* neuron*.txt', '* All_neuron*.txt']
EEG_FILE_PATTERN = '* EEG_sleep.txt'
ROOT = Path(__file__).parent
################################################################################
# Sleep detection #
################################################################################
USE_EEG_SLEEP_DETECTION = False
EEG_FREQ: int = 200  # Number of EEG measures per second.

################################################################################

################################################################################
# Implementation
################################################################################
# Logging
################################################################################


def fmt_path(path: Path) -> str:
    return f'"{path.relative_to(ROOT)}"'


def ok(msg: str) -> None:
    clrprint('[ OK ]', clr='green', end='\t')
    print(msg)


def fail(msg: str) -> None:
    clrprint('[FAIL]', clr='red', end='\t')
    print(msg)
    exit(1)


def info(msg: str) -> None:
    clrprint('[INFO]', clr='blue', end='\t')
    print(msg)


def warn(msg: str) -> None:
    clrprint('[WARN]', clr='yellow', end='\t')
    print(msg)
################################################################################
# Data loading
################################################################################


@dataclass
class EEG:
    start_time: float
    eeg: np.ndarray


@dataclass
class Data:
    apnoes: np.ndarray
    neurons: tp.Dict[str, np.ndarray]
    eeg: tp.Optional[EEG] = None


def validate_file_prefixes(paths: tp.List[Path]) -> None:
    prefixes = {str(path.relative_to(ROOT)).split(maxsplit=1)[0]
                for path in paths}
    if len(prefixes) > 1:
        prefixes = ', '.join(map(lambda s: f'"{s}"', prefixes))
        fail(f'Files have different prefixes: {prefixes}')


def find_files() -> tp.Tuple[Path, tp.List[Path], tp.Optional[Path]]:
    info('Locating files...')
    info(f'Root: {ROOT}')

    info(f'Apnoe file pattern: "{APNOE_FILE_PATTERN}"')
    apnoe_files = list(ROOT.glob(APNOE_FILE_PATTERN))

    neuron_files: tp.List[Path] = []
    for neuron_file_pattern in NEURON_FILE_PATTERNS:
        info(f'Neuron file pattern: "{neuron_file_pattern}"')
        neuron_files += list(ROOT.glob(neuron_file_pattern))

    info(f'EEG file pattern: "{EEG_FILE_PATTERN}"')
    eeg_files = list(ROOT.glob(EEG_FILE_PATTERN))

    validate_file_prefixes(apnoe_files + neuron_files + eeg_files)

    def file_found(kind: str, path: Path) -> None:
        ok(f'{kind} file found: {fmt_path(path)}')

    def file_not_found(kind: str, warn_only: bool = False) -> None:
        log_func = warn if warn_only else fail
        log_func(f'{kind} file not found')

    def multiple_files_found(kind: str, paths: tp.List[Path]) -> None:
        paths = ', '.join(map(fmt_path, paths))
        fail(f'Multiple {kind} files found: {paths}')

    if not apnoe_files:
        file_not_found('Apnoe')
    if len(apnoe_files) > 1:
        multiple_files_found('apnoe', apnoe_files)
    apnoe_file = apnoe_files[0]
    file_found('Apnoe', apnoe_file)

    if not neuron_files:
        file_not_found('Neuron')
    for neuron_file in neuron_files:
        file_found('Neuron', neuron_file)
    info(f'Neuron files found: {len(neuron_files)}')

    if len(eeg_files) > 1:
        multiple_files_found('EEG', eeg_files)
    if eeg_files:
        eeg_file = eeg_files[0]
        file_found('EEG', eeg_file)
    else:
        file_not_found('EEG', warn_only=not USE_EEG_SLEEP_DETECTION)
        eeg_file = None
    ok('Files located')
    return apnoe_file, neuron_files, eeg_file


def load_everything() -> Data:
    apnoe_file, neuron_files, eeg_file = find_files()

    info('Loading files...')

    def load_file(path: Path, *, required: bool,
                  skip_rows: int = 5) -> tp.Optional[np.ndarray]:
        content = None
        with contextlib.suppress(Exception):
            content = np.loadtxt(str(path), skiprows=skip_rows, dtype='float')
        if len(content.shape) == 0:
            content.shape = (1,)
        if content is not None and len(content) > 0 and len(content.shape) == 1:
            ok(f'{fmt_path(path)} loaded')
            return content
        if required:
            fail(f'{fmt_path(path)} not loaded')
        warn(f'{fmt_path(path)} not loaded')
        return None

    apnoe_data = load_file(apnoe_file, required=True)
    apnoe_data.sort()
    info(f'Apnoes loaded: {len(apnoe_data)}')
    info(f'Apnoes: {", ".join(map(str, apnoe_data))}')

    neurons: tp.Dict[str, np.ndarray] = {}
    for neuron_file in neuron_files:
        neuron_data = load_file(neuron_file, required=False)
        if neuron_data is not None:
            neuron_name = neuron_file.relative_to(ROOT).stem. \
                split(maxsplit=1)[1]
            neurons[neuron_name] = neuron_data
    if not neurons:
        fail('No neuron loaded')
    info(f'Neurons loaded: {len(neurons)}')
    info(f'Neurons: {", ".join(neurons.keys())}')

    eeg = None
    if eeg_file:
        eeg_content = load_file(eeg_file, required=USE_EEG_SLEEP_DETECTION,
                                skip_rows=7)
        if eeg_content is not None:
            with open(eeg_file) as f:
                for _ in range(7):
                    line = next(f)
            eeg = EEG(start_time=float(line.split()[1]), eeg=eeg_content)
            info(f'EEG start time: {eeg.start_time}')

    ok('Files loaded')
    return Data(apnoes=apnoe_data, neurons=neurons, eeg=eeg)


################################################################################
# Bootstrap
################################################################################
class SampleValidator:
    def __init__(self):
        self.ranges = SortedSet(key=itemgetter(0))

    def prohibit_range(self, begin: float, end: float):
        while True:
            left_pos = self.ranges.bisect_right((end, end))
            if left_pos == 0:
                self.ranges.add((begin, end))
                break
            left_pos -= 1
            left_begin, left_end = self.ranges[left_pos]
            # left_begin <= end
            if left_end < begin:  # Disjoint ranges.
                self.ranges.add((begin, end))
                break
            self.ranges.pop(left_pos)
            begin = min(begin, left_begin)
            end = max(end, left_end)

    def check_point(self, point: float) -> bool:
        left_pos = self.ranges.bisect_right((point, point))
        if left_pos == 0:
            return True
        left_pos -= 1
        return self.ranges[left_pos][1] < point


def generate_sample(apnoe_starts: np.ndarray, apnoe_duration: float,
                    max_time: float) -> np.ndarray:
    validator = SampleValidator()
    validator.prohibit_range(float('-inf'), 0.)
    validator.prohibit_range(max_time, float('inf'))
    for apnoe in apnoe_starts:
        validator.prohibit_range(apnoe - apnoe_duration, apnoe + apnoe_duration)

    generator = sps.uniform(0, max_time - apnoe_duration)

    size = 0
    sample = np.zeros(BOOTSTRAP_SIZE, dtype='float')

    i = 0
    values = np.array([], dtype='float')

    def get_point() -> float:
        nonlocal i, size, values
        if i == len(values):
            i = 0
            values = generator.rvs(size=BOOTSTRAP_SIZE - size)
        i += 1
        return values[i - 1]

    info('Bootstrapping sample...')
    info(f'Apnoe duration: {apnoe_duration}')
    info(f'Max time: {max_time}')
    for _ in range(BOOTSTRAP_SIZE):
        good = False
        for retry in range(50):
            p = get_point()
            if validator.check_point(p) or True:
                sample[size] = p
                size += 1
                validator.prohibit_range(p - apnoe_duration, p + apnoe_duration)
                good = True
                break
        if not good:
            continue
    if size == BOOTSTRAP_SIZE:
        ok('Sample ready')
    elif size == 0:
        fail('Bootstrap failed')
    else:
        warn(f'Bootstrapped sample is small: {size} instead of '
             f'{BOOTSTRAP_SIZE}')
    return sample[:size]
################################################################################
# Neuron processing
################################################################################


@dataclass
class CurveStats:
    min_value: float
    max_value: float
    min_pos: int
    max_pos: int

    @property
    def min_time(self) -> float:
        return self.min_pos * SLIDING_WINDOW_SHIFT

    @property
    def max_time(self) -> float:
        return self.max_pos * SLIDING_WINDOW_SHIFT

    @property
    def dif(self) -> float:
        return self.max_value - self.min_value


def calc_freq(neuron: np.ndarray, start: float, end: float) -> np.ndarray:
    starts = np.arange(start, end - 1, SLIDING_WINDOW_SHIFT)
    ends = starts + SLIDING_WINDOW_DURATION
    return (np.searchsorted(neuron, ends) - np.searchsorted(neuron, starts)) / \
           (ends - starts)


def find_min_max(data: np.ndarray) -> CurveStats:
    min_pos = int(np.argmin(data))
    max_pos = int(np.argmax(data))
    return CurveStats(data[min_pos], data[max_pos], min_pos, max_pos)


result = open('result.txt', 'w')


def process_neuron(neuron: np.ndarray, apnoes: np.ndarray, eeg: EEG):
    activity_in_apnoes = [calc_freq(neuron, apnoe - APNOE_LEFT_DURATION,
                                    apnoe + APNOE_RIGHT_DURATION)
                          for apnoe in apnoes]
    for i, activity in enumerate(activity_in_apnoes):
        info(f'Processing apnoe {i}...')
        apnoe_stats = find_min_max(activity)
        info(f'Time of minimum: {apnoe_stats.min_time - APNOE_LEFT_DURATION}')
        info(f'Time of maximum: {apnoe_stats.max_time - APNOE_LEFT_DURATION}')
        apnoe_duration = abs(apnoe_stats.min_time - apnoe_stats.max_time)
        bootstrapped_sample = generate_sample(apnoes +
                                              min(apnoe_stats.min_time,
                                                  apnoe_stats.max_time),
                                              apnoe_duration,
                                              float(neuron.max(initial=.0)) + 1)
        bootstrapped_stats = np.array([
            find_min_max(calc_freq(neuron, begin, begin + apnoe_duration)).dif
            for begin in bootstrapped_sample
        ])
        bootstrapped_stats.sort()
        num_less = bootstrapped_stats.searchsorted(apnoe_stats.dif)
        p_value = 1. - num_less / BOOTSTRAP_SIZE
        info(f'p-value = {p_value:.2f}')
        print(f'{p_value:.2f}', file=result)
        info(f'min: {bootstrapped_stats[0]:.2f}')
        info(f'max: {bootstrapped_stats[-1]:.2f}')
        info(f'apnoe: {apnoe_stats.dif}')
        ok('Apnoe processing finished')


################################################################################


def main():
    data = load_everything()
    for neuron_name, neuron in data.neurons.items():
        info(f'Processing {neuron_name}...')
        process_neuron(neuron, data.apnoes, data.eeg)
        ok(f'Neuron processing finished: {neuron_name}')


if __name__ == '__main__':
    # colorama.init()
    np.random.seed(42)
    main()
