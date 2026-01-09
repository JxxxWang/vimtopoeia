import hashlib
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import click
import h5py
import hdf5plugin
import librosa
import numpy as np
import rootutils
from loguru import logger
from pedalboard import VST3Plugin
from pyloudnorm import Meter
import pyloudnorm as pyln
from tqdm import trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation import load_plugin, param_specs, render_params  # noqa
from data_generation.param_spec import ParamSpec  # noqa

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

_worker_plugin = None

def worker_init(plugin_path: str):
    global _worker_plugin
    try:
        _worker_plugin = load_plugin(plugin_path)
    except Exception as e:
        logger.error(f"Failed to load plugin: {e}")
        raise

def worker_generate_sample(
    velocity,
    signal_duration_seconds,
    sample_rate,
    channels,
    min_loudness,
    param_spec,
    preset_path,
):
    return generate_sample(
        _worker_plugin,
        velocity,
        signal_duration_seconds,
        sample_rate,
        channels,
        min_loudness,
        param_spec,
        preset_path,
    )


@dataclass
class VSTDataSample:
    target_synth_params: dict[str, float]
    reference_synth_params: dict[str, float]
    note_params: dict[str, float]

    sample_rate: float
    channels: int

    param_spec: ParamSpec

    target_audio: np.ndarray
    reference_audio: np.ndarray
    
    # Removed Mel Spectrograms (Requirement 2: Raw Data Only)
    # target_mel_spec: np.ndarray
    # reference_mel_spec: np.ndarray

    target_param_array: np.ndarray = None
    reference_param_array: np.ndarray = None

    def __post_init__(self):
        self.target_param_array = self.param_spec.encode(
            self.target_synth_params, self.note_params
        )
        self.reference_param_array = self.param_spec.encode(
            self.reference_synth_params, self.note_params
        )

def generate_sample(
    plugin: VST3Plugin,
    velocity: int,
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    min_loudness: float,
    param_spec: ParamSpec,
    preset_path: str,
) -> VSTDataSample:
    while True:
        logger.debug("sampling params")
        target_synth_params, ref_synth_params, note_params = param_spec.sample_pair()

        # Render Target
        logger.debug("rendering target")
        target_output = render_params(
            plugin,
            target_synth_params,
            note_params["pitch"],
            velocity,
            note_params["note_start_and_end"],
            signal_duration_seconds,
            sample_rate,
            channels,
            # preset_path=preset_path,
            preset_path=None,
        )

        # Render Reference
        logger.debug("rendering reference")
        ref_output = render_params(
            plugin,
            ref_synth_params,
            note_params["pitch"],
            velocity,
            note_params["note_start_and_end"],
            signal_duration_seconds,
            sample_rate,
            channels,
            # preset_path=preset_path,
            preset_path=None,
        )

        meter = Meter(sample_rate)
        target_loudness = meter.integrated_loudness(target_output.T)
        ref_loudness = meter.integrated_loudness(ref_output.T)

        def normalize_if_needed(audio, current_loudness, target_l):
            if current_loudness > -70.0 and current_loudness < target_l:
                try:
                    norm = pyln.normalize.loudness(audio.T, current_loudness, target_l).T
                    if np.max(np.abs(norm)) < 1.0:
                        return norm, target_l
                except Exception:
                    pass
            return audio, current_loudness

        target_output, target_loudness = normalize_if_needed(target_output, target_loudness, min_loudness)
        ref_output, ref_loudness = normalize_if_needed(ref_output, ref_loudness, min_loudness)

        logger.debug(
            f"target loudness: {target_loudness}, ref loudness: {ref_loudness}"
        )
        if target_loudness < min_loudness or ref_loudness < min_loudness:
            logger.debug("loudness too low, skipping")
            continue

        break

    # Removed Spectrogram Calculation (Requirement 2)
    # logger.debug("making spectrograms")
    # target_spectrogram = make_spectrogram(target_output.T, sample_rate)
    # ref_spectrogram = make_spectrogram(ref_output.T, sample_rate)

    return VSTDataSample(
        target_synth_params=target_synth_params,
        reference_synth_params=ref_synth_params,
        note_params=note_params,
        target_audio=target_output.T,
        reference_audio=ref_output.T,
        sample_rate=sample_rate,
        channels=channels,
        param_spec=param_spec,
    )


def save_samples(
    samples: List[VSTDataSample],
    target_audio_dataset: h5py.Dataset,
    ref_audio_dataset: h5py.Dataset,
    # target_mel_dataset: h5py.Dataset,
    # ref_mel_dataset: h5py.Dataset,
    target_param_dataset: h5py.Dataset,
    ref_param_dataset: h5py.Dataset,
    start_idx: int,
) -> None:
    logger.info(f"Saving {len(samples)} samples...")

    target_audios = np.stack([s.target_audio.T for s in samples], axis=0)
    ref_audios = np.stack([s.reference_audio.T for s in samples], axis=0)

    # target_mels = np.stack([s.target_mel_spec for s in samples], axis=0)
    # ref_mels = np.stack([s.reference_mel_spec for s in samples], axis=0)

    target_params = np.stack([s.target_param_array for s in samples], axis=0)
    ref_params = np.stack([s.reference_param_array for s in samples], axis=0)

    n = len(samples)
    target_audio_dataset[start_idx : start_idx + n, :, :] = target_audios
    ref_audio_dataset[start_idx : start_idx + n, :, :] = ref_audios

    # target_mel_dataset[start_idx : start_idx + n, :, :] = target_mels
    # ref_mel_dataset[start_idx : start_idx + n, :, :] = ref_mels

    target_param_dataset[start_idx : start_idx + n, :] = target_params
    ref_param_dataset[start_idx : start_idx + n, :] = ref_params

    logger.info(f"{len(samples)} samples written!")


def get_first_unwritten_idx(dataset: h5py.Dataset) -> int:
    num_rows, *_ = dataset.shape
    for i in range(num_rows):
        row = dataset[num_rows - i - 1]
        if not np.all(row == 0):
            return num_rows - i
        logger.debug(f"Row {num_rows - i - 1} is empty...")

    return 0


def create_dataset_and_get_first_unwritten_idx(
    h5py_file: h5py.File,
    name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    compression: Any,
) -> Tuple[h5py.Dataset, int]:
    logger.info(f"Looking for dataset {name}...")
    if name in h5py_file:
        logger.info(f"Found dataset {name}, looking for first unwritten row.")
        dataset = h5py_file[name]
        return dataset, get_first_unwritten_idx(dataset)

    dataset = h5py_file.create_dataset(
        name, shape=shape, dtype=dtype, compression=compression
    )
    return dataset, 0


def create_datasets_and_get_start_idx(
    hdf5_file: h5py.File,
    num_samples: int,
    channels: int,
    sample_rate: float,
    signal_duration_seconds: float,
    num_params: int,
):
    # audio_shape = (num_samples, channels, int(sample_rate * signal_duration_seconds))
    # mel_shape = (num_samples, 2, 128, int(signal_duration_seconds * 100) + 1)
    param_shape = (num_samples, num_params)

    target_audio_ds, ta_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file, 
        "target_audio", 
        (num_samples, channels, sample_rate * signal_duration_seconds), 
        dtype = np.float16, 
        compression = hdf5plugin.Blosc2()
    )
    ref_audio_ds, ra_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file, 
        "reference_audio", 
        (num_samples, channels, sample_rate * signal_duration_seconds), 
        np.float16, 
        compression=hdf5plugin.Blosc2()
    )

    # target_mel_ds, tm_idx = create_dataset_and_get_first_unwritten_idx(
    #     hdf5_file, "target_mel_spec", mel_shape, np.float32, hdf5plugin.Blosc2()
    # )
    # ref_mel_ds, rm_idx = create_dataset_and_get_first_unwritten_idx(
    #     hdf5_file, "reference_mel_spec", mel_shape, np.float32, hdf5plugin.Blosc2()
    # )

    target_param_ds, tp_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file, 
        "target_param_array", 
        (num_samples, num_params), 
        np.float32, 
        hdf5plugin.Blosc2()
    )
    ref_param_ds, rp_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file, 
        "reference_param_array", 
        (num_samples, num_params), 
        np.float32, 
        hdf5plugin.Blosc2()
    )

    start_idx = min(ta_idx, ra_idx, tp_idx, rp_idx)
    return (
        target_audio_ds,
        ref_audio_ds,
        # target_mel_ds,
        # ref_mel_ds,
        target_param_ds,
        ref_param_ds,
        min(ta_idx, ra_idx, start_idx, tp_idx, rp_idx)
    )


def make_dataset(
    hdf5_file: h5py.File,
    num_samples: int,
    plugin_path: str,
    preset_path: str,
    sample_rate: float,
    channels: int,
    velocity: int,
    signal_duration_seconds: float,
    min_loudness: float,
    param_spec: ParamSpec,
    sample_batch_size: int,
) -> None:

    target_audio_ds, ref_audio_ds, target_param_ds, ref_param_ds, start_idx = (
        create_datasets_and_get_start_idx(
            hdf5_file=hdf5_file,
            num_samples=num_samples,
            channels=channels,
            sample_rate=sample_rate,
            signal_duration_seconds=signal_duration_seconds,
            num_params=len(param_spec),
        )
    )

    # Set attributes
    target_audio_ds.attrs["velocity"] = velocity
    target_audio_ds.attrs["signal_duration_seconds"] = signal_duration_seconds
    target_audio_ds.attrs["sample_rate"] = sample_rate
    target_audio_ds.attrs["channels"] = channels
    target_audio_ds.attrs["min_loudness"] = min_loudness

    # plugin = load_plugin(plugin_path) # moved to worker

    sample_batch = []
    current_idx = start_idx
    
    num_workers = min(multiprocessing.cpu_count(), 8)
    samples_to_generate = num_samples - start_idx
    
    if samples_to_generate <= 0:
        logger.info("Dataset completion check: All samples already generated.")
        return

    logger.info(f"Starting parallel generation: {samples_to_generate} samples, {num_workers} workers.")

    with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_init, initargs=(plugin_path,)) as executor:
        futures = []
        for _ in range(samples_to_generate):
            futures.append(executor.submit(
                worker_generate_sample,
                velocity,
                signal_duration_seconds,
                sample_rate,
                channels,
                min_loudness,
                param_spec,
                preset_path,
            ))
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating samples"):
            try:
                sample = future.result()
                sample_batch.append(sample)
                
                if len(sample_batch) >= sample_batch_size:
                    save_samples(
                        sample_batch,
                        target_audio_ds,
                        ref_audio_ds,
                        target_param_ds,
                        ref_param_ds,
                        current_idx,
                    )
                    current_idx += len(sample_batch)
                    sample_batch = []
            except Exception as e:
                logger.error(f"Sample generation failed: {e}")

    if len(sample_batch) > 0:
        save_samples(
            sample_batch,
            target_audio_ds,
            ref_audio_ds,
            target_param_ds,
            ref_param_ds,
            current_idx,
        )


@click.command()
@click.argument("data_file", type=str, required=True)
@click.argument("num_samples", type=int, required=True)
@click.option("--plugin_path", "-p", type=str, default="/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")
@click.option("--preset_path", "-r", type=str, default=None)
@click.option("--sample_rate", "-s", type=float, default=44100.0)
@click.option("--channels", "-c", type=int, default=2)
@click.option("--velocity", "-v", type=int, default=100)
@click.option("--signal_duration_seconds", "-d", type=float, default=4.0)
@click.option("--min_loudness", "-l", type=float, default=-30.0)
@click.option("--param_spec", "-t", type=str, default="surge_simple")
@click.option("--sample_batch_size", "-b", type=int, default=32)
def main(
    data_file: str,
    num_samples: int,
    plugin_path: str = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3",
    preset_path: str = None,
    sample_rate: float = 44100.0,
    channels: int = 2,
    velocity: int = 100,
    signal_duration_seconds: float = 4.0,
    min_loudness: float = -30.0,
    param_spec: str = "surge_simple",
    sample_batch_size: int = 32,
):
    param_spec = param_specs[param_spec]
    with h5py.File(data_file, "a") as f:
        make_dataset(
            f,
            num_samples,
            plugin_path,
            preset_path,
            sample_rate,
            channels,
            velocity,
            signal_duration_seconds,
            min_loudness,
            param_spec,
            sample_batch_size,
        )


if __name__ == "__main__":
    main()
