"""
Generate a robust Keyword Spotting (KWS) dataset.

Classes:
    wake_word/
    unknown/
    noise/

Wake-word samples:
    - Edge TTS voices
    - Optional real recordings of the wake word
    - Praat pseudo-speaker transformations
    - Rate and pitch variations
    - Noise, gain and reverb augmentation

Unknown samples:
    - Automatically downloaded LibriSpeech-PC text corpus
    - Random general speech sentences
    - Edge TTS voices
    - Rate and pitch variations
    - Praat pseudo-speaker transformations
    - Noise, gain and reverb augmentation

Noise samples:
    - MS-SNSD real noise corpus when available
    - Synthetic noise fallback

The unknown class does NOT contain manually written sentences.

Example:

    python generate_wakeword_dataset.py ^
        --wake-word "Hey Ruby" ^
        --out-dir dataset ^
        --my-voice-dir my_voice ^
        --wake-clips 900 ^
        --unknown-clips 900 ^
        --noise-clips 400 ^
        --pseudo-speakers-per-recording 25

Linux/macOS:

    python generate_wakeword_dataset.py \
        --wake-word "Hey Ruby" \
        --out-dir dataset \
        --my-voice-dir my_voice \
        --wake-clips 900 \
        --unknown-clips 900 \
        --noise-clips 400 \
        --pseudo-speakers-per-recording 25
"""

import argparse
import asyncio
import csv
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000
TARGET_SEC = 1.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_SEC)

LIBRISPEECH_PC_URL = (
    "https://www.openslr.org/resources/145/manifests.tar.gz"
)

LIBRISPEECH_PC_ARCHIVE = "librispeech_pc_manifests.tar.gz"

MS_SNSD_REPO = "https://github.com/microsoft/MS-SNSD.git"


# ============================================================
# Edge TTS voices
# ============================================================

VOICES = [
    # US English
    "en-US-AriaNeural",
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-US-DavisNeural",
    "en-US-AmberNeural",
    "en-US-AnaNeural",
    "en-US-BrandonNeural",
    "en-US-ChristopherNeural",
    "en-US-CoraNeural",
    "en-US-ElizabethNeural",
    "en-US-EricNeural",
    "en-US-JacobNeural",
    "en-US-JaneNeural",
    "en-US-JasonNeural",
    "en-US-MichelleNeural",
    "en-US-MonicaNeural",
    "en-US-NancyNeural",
    "en-US-RogerNeural",
    "en-US-SaraNeural",
    "en-US-SteffanNeural",

    # UK English
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-GB-LibbyNeural",
    "en-GB-ThomasNeural",
    "en-GB-AbbiNeural",
    "en-GB-BellaNeural",
    "en-GB-ElliotNeural",
    "en-GB-EthanNeural",
    "en-GB-HollieNeural",
    "en-GB-MaisieNeural",

    # Australian English
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    "en-AU-AnnetteNeural",
    "en-AU-CarlyNeural",
    "en-AU-DarrenNeural",
    "en-AU-ElsieNeural",
    "en-AU-JoanneNeural",
    "en-AU-KenNeural",
    "en-AU-KimNeural",

    # Indian English
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",

    # Canadian English
    "en-CA-ClaraNeural",
    "en-CA-LiamNeural",

    # Irish English
    "en-IE-EmilyNeural",
    "en-IE-ConnorNeural",

    # South African English
    "en-ZA-LeahNeural",
    "en-ZA-LukeNeural",

    # New Zealand English
    "en-NZ-MitchellNeural",
    "en-NZ-MollyNeural",

    # Nigerian English
    "en-NG-EzinneNeural",
    "en-NG-AbeoNeural",

    # Kenyan English
    "en-KE-AsiliaNeural",
    "en-KE-ChilembaNeural",

    # Philippines English
    "en-PH-RosaNeural",
    "en-PH-JamesNeural",
]


RATE_VARIATIONS = [
    "-20%",
    "-10%",
    "+0%",
    "+10%",
    "+20%",
]


PITCH_VARIATIONS = [
    "-30Hz",
    "-10Hz",
    "+0Hz",
    "+10Hz",
    "+30Hz",
]


# ============================================================
# Dependency checks
# ============================================================

def check_dependencies():
    missing = []

    try:
        import edge_tts
    except ImportError:
        missing.append("edge-tts")

    try:
        import librosa
    except ImportError:
        missing.append("librosa")

    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")

    try:
        import parselmouth
    except ImportError:
        missing.append("praat-parselmouth")

    try:
        import scipy
    except ImportError:
        missing.append("scipy")

    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")

    if missing:
        print("\nMissing Python packages:")
        for package in missing:
            print(f"  - {package}")

        print("\nInstall them with:")
        print("  pip install " + " ".join(missing))
        sys.exit(1)

    if shutil.which("ffmpeg") is None:
        print("\nWARNING: ffmpeg was not found in PATH.")
        print("Some audio formats may not work.")
        print(
            "Install FFmpeg and make sure it is available "
            "from the command line."
        )


check_dependencies()

import edge_tts
import librosa
import parselmouth
import soundfile as sf

from scipy.signal import fftconvolve
from tqdm import tqdm


# ============================================================
# Utility functions
# ============================================================

def safe_filename(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")

    if len(text) > 50:
        text = text[:50]

    return text or "speech"


def run_command(command, cwd=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout

    except subprocess.CalledProcessError as exc:
        print("\nCommand failed:")
        print(" ".join(command))
        print(exc.stderr)
        return None


# ============================================================
# Edge TTS
# ============================================================

async def synth_one(
    text,
    voice,
    output_mp3,
    rate="+0%",
    pitch="+0Hz",
):
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )

    await communicate.save(output_mp3)


def synthesize(
    text,
    voice,
    output_mp3,
    rate="+0%",
    pitch="+0Hz",
):
    asyncio.run(
        synth_one(
            text=text,
            voice=voice,
            output_mp3=output_mp3,
            rate=rate,
            pitch=pitch,
        )
    )


# ============================================================
# Audio loading / conversion
# ============================================================

def mp3_to_wav(input_file, output_file):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_file),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(output_file),
    ]

    result = run_command(command)

    if result is None:
        raise RuntimeError(
            f"FFmpeg failed for {input_file}"
        )


def any_to_wav(input_file, output_file):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_file),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(output_file),
    ]

    result = run_command(command)

    if result is None:
        raise RuntimeError(
            f"FFmpeg failed for {input_file}"
        )


def load_wav_mono(path):
    audio, sr = sf.read(
        path,
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1,
        )

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
        )

    return audio.astype(np.float32)


def write_wav_mono(path, audio):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1.0:
        audio = audio / peak

    sf.write(
        path,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16",
    )


# ============================================================
# Audio normalization / trimming
# ============================================================

def trim_and_pad_to_length(audio):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if len(audio) > TARGET_SAMPLES:

        max_start = (
            len(audio)
            - TARGET_SAMPLES
        )

        start = random.randint(
            0,
            max_start,
        )

        audio = audio[
            start:start + TARGET_SAMPLES
        ]

    elif len(audio) < TARGET_SAMPLES:

        missing = (
            TARGET_SAMPLES
            - len(audio)
        )

        left = random.randint(
            0,
            missing,
        )

        right = (
            missing - left
        )

        audio = np.pad(
            audio,
            (left, right),
            mode="constant",
        )

    return audio.astype(
        np.float32
    )


def normalize_audio(
    audio,
    target_peak=0.90,
):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak < 1e-8:
        return audio

    return audio * (
        target_peak / peak
    )


# ============================================================
# Praat pseudo-speaker transformation
# ============================================================

def pseudo_speaker_shift(
    audio,
    pitch_floor,
    pitch_ceiling,
    formant_shift,
):
    """
    Use Praat Change Gender to create
    pseudo-speaker variation.
    """

    try:
        sound = parselmouth.Sound(
            audio,
            sampling_frequency=SAMPLE_RATE,
        )

        result = parselmouth.praat.call(
            sound,
            "Change gender",
            75,
            600,
            pitch_floor,
            pitch_ceiling,
            formant_shift,
            1.0,
            1.0,
        )

        shifted = result.values[0]

        return shifted.astype(
            np.float32
        )

    except Exception:
        return audio


def random_pseudo_speaker_params():
    pitch_floor = random.uniform(
        65,
        100,
    )

    pitch_ceiling = random.uniform(
        400,
        600,
    )

    formant_shift = random.uniform(
        -25,
        25,
    )

    return (
        pitch_floor,
        pitch_ceiling,
        formant_shift,
    )


# ============================================================
# Reverb
# ============================================================

def make_synthetic_rir():
    rir_length = random.randint(
        int(0.08 * SAMPLE_RATE),
        int(0.35 * SAMPLE_RATE),
    )

    rir = np.zeros(
        rir_length,
        dtype=np.float32,
    )

    rir[0] = 1.0

    number_of_reflections = random.randint(
        5,
        20,
    )

    for _ in range(
        number_of_reflections
    ):
        delay = random.randint(
            1,
            rir_length - 1,
        )

        amplitude = random.uniform(
            0.05,
            0.35,
        )

        rir[delay] += amplitude

    decay = np.exp(
        -np.linspace(
            0,
            random.uniform(
                2.0,
                5.0,
            ),
            rir_length,
        )
    )

    rir *= decay

    return rir


def apply_reverb(audio):

    if random.random() > 0.35:
        return audio

    rir = make_synthetic_rir()

    reverberated = fftconvolve(
        audio,
        rir,
        mode="full",
    )

    reverberated = reverberated[
        :len(audio)
    ]

    dry_gain = random.uniform(
        0.65,
        0.90,
    )

    wet_gain = 1.0 - dry_gain

    result = (
        dry_gain * audio
        + wet_gain * reverberated
    )

    return result.astype(
        np.float32
    )


# ============================================================
# Noise
# ============================================================

def try_download_noise_corpus(
    cache_dir,
):
    """
    Clone Microsoft MS-SNSD if possible.
    """

    cache_dir = Path(
        cache_dir
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    repo_dir = (
        cache_dir / "MS-SNSD"
    )

    if repo_dir.exists():
        return repo_dir

    print(
        "\nDownloading MS-SNSD noise corpus..."
    )

    result = run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            MS_SNSD_REPO,
            str(repo_dir),
        ]
    )

    if result is None:
        print(
            "Could not download MS-SNSD."
        )
        return None

    return repo_dir


def find_noise_files(noise_root):

    if noise_root is None:
        return []

    patterns = [
        "**/*.wav",
        "**/*.flac",
        "**/*.mp3",
    ]

    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(
                str(
                    noise_root / pattern
                ),
                recursive=True,
            )
        )

    return files


def load_random_real_noise_chunk(
    noise_files,
):
    if not noise_files:
        return None

    path = random.choice(
        noise_files
    )

    try:
        audio, sr = sf.read(
            path,
            dtype="float32",
        )

        if audio.ndim > 1:
            audio = np.mean(
                audio,
                axis=1,
            )

        if sr != SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=SAMPLE_RATE,
            )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if len(audio) < TARGET_SAMPLES:

            repeats = int(
                np.ceil(
                    TARGET_SAMPLES
                    / len(audio)
                )
            )

            audio = np.tile(
                audio,
                repeats,
            )

        max_start = (
            len(audio)
            - TARGET_SAMPLES
        )

        start = (
            random.randint(
                0,
                max_start,
            )
            if max_start > 0
            else 0
        )

        return audio[
            start:start + TARGET_SAMPLES
        ]

    except Exception:
        return None


def synthetic_noise_clip():

    noise_type = random.choice(
        [
            "white",
            "pink",
            "hum",
            "impulse",
        ]
    )

    if noise_type == "white":

        noise = np.random.normal(
            0,
            1,
            TARGET_SAMPLES,
        )

    elif noise_type == "pink":

        white = np.random.normal(
            0,
            1,
            TARGET_SAMPLES,
        )

        spectrum = np.fft.rfft(
            white
        )

        freqs = np.fft.rfftfreq(
            TARGET_SAMPLES,
        )

        freqs[0] = 1.0

        spectrum /= np.sqrt(
            freqs
        )

        noise = np.fft.irfft(
            spectrum,
            n=TARGET_SAMPLES,
        )

    elif noise_type == "hum":

        t = (
            np.arange(
                TARGET_SAMPLES
            )
            / SAMPLE_RATE
        )

        freq = random.choice(
            [
                50,
                60,
                100,
                120,
            ]
        )

        noise = (
            np.sin(
                2 * np.pi * freq * t
            )
            + 0.3
            * np.sin(
                2 * np.pi * freq * 2 * t
            )
        )

    else:

        noise = np.zeros(
            TARGET_SAMPLES,
            dtype=np.float32,
        )

        number = random.randint(
            3,
            20,
        )

        for _ in range(number):

            index = random.randint(
                0,
                TARGET_SAMPLES - 1,
            )

            width = random.randint(
                1,
                30,
            )

            end = min(
                index + width,
                TARGET_SAMPLES,
            )

            noise[
                index:end
            ] = random.uniform(
                -1,
                1,
            )

    noise = np.asarray(
        noise,
        dtype=np.float32,
    )

    return normalize_audio(
        noise,
        random.uniform(
            0.05,
            0.5,
        ),
    )


def mix_noise(
    audio,
    noise,
):
    if noise is None:
        return audio

    noise = trim_and_pad_to_length(
        noise
    )

    snr_db = random.uniform(
        5,
        25,
    )

    signal_power = np.mean(
        audio ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    if noise_power < 1e-10:
        return audio

    desired_noise_power = (
        signal_power
        / (10 ** (snr_db / 10))
    )

    scale = np.sqrt(
        desired_noise_power
        / noise_power
    )

    return (
        audio + noise * scale
    ).astype(np.float32)


# ============================================================
# Speech augmentation
# ============================================================

def augment_speech_clip(
    audio,
    noise_files=None,
    use_pseudo_speaker=False,
):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if use_pseudo_speaker:

        if random.random() < 0.75:

            (
                pitch_floor,
                pitch_ceiling,
                formant_shift,
            ) = random_pseudo_speaker_params()

            audio = pseudo_speaker_shift(
                audio,
                pitch_floor,
                pitch_ceiling,
                formant_shift,
            )

    gain_db = random.uniform(
        -6,
        6,
    )

    gain = 10 ** (
        gain_db / 20
    )

    audio = audio * gain

    audio = apply_reverb(
        audio
    )

    if random.random() < 0.75:

        if noise_files:

            noise = load_random_real_noise_chunk(
                noise_files
            )

        else:

            noise = synthetic_noise_clip()

        audio = mix_noise(
            audio,
            noise,
        )

    audio = trim_and_pad_to_length(
        audio
    )

    audio = normalize_audio(
        audio,
        target_peak=random.uniform(
            0.75,
            0.95,
        ),
    )

    return audio


# ============================================================
# LibriSpeech-PC text corpus
# ============================================================

def download_file(
    url,
    output_path,
):
    output_path = Path(
        output_path
    )

    if output_path.exists():

        print(
            f"Using existing file: "
            f"{output_path}"
        )

        return output_path

    print(
        f"\nDownloading:\n{url}"
    )

    urllib.request.urlretrieve(
        url,
        output_path,
    )

    return output_path


def download_librispeech_pc(
    cache_dir,
):
    """
    Download the lightweight LibriSpeech-PC
    manifest archive.

    The archive contains text manifests only,
    not audio.
    """

    cache_dir = Path(
        cache_dir
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_path = (
        cache_dir
        / LIBRISPEECH_PC_ARCHIVE
    )

    download_file(
        LIBRISPEECH_PC_URL,
        archive_path,
    )

    extract_dir = (
        cache_dir
        / "librispeech_pc"
    )

    if not extract_dir.exists():

        print(
            "\nExtracting LibriSpeech-PC manifests..."
        )

        extract_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with tarfile.open(
            archive_path,
            "r:gz",
        ) as tar:

            tar.extractall(
                extract_dir
            )

    return extract_dir


def extract_text_from_json_object(
    obj,
):
    """
    Recursively search a JSON object
    for likely transcript/text fields.
    """

    texts = []

    if isinstance(obj, dict):

        for key in [
            "text",
            "text_raw",
            "text_normalized",
            "transcription",
            "transcript",
        ]:

            value = obj.get(
                key
            )

            if isinstance(
                value,
                str,
            ):
                texts.append(
                    value
                )

        for value in obj.values():

            if isinstance(
                value,
                (dict, list),
            ):

                texts.extend(
                    extract_text_from_json_object(
                        value
                    )
                )

    elif isinstance(obj, list):

        for item in obj:

            texts.extend(
                extract_text_from_json_object(
                    item
                )
            )

    return texts


def load_manifest_file(
    manifest_path,
):
    """
    Load either a standard JSON file
    or a JSON Lines (JSONL) file.

    LibriSpeech-PC manifests are JSONL files,
    where every line contains one JSON object.
    """

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as f:

        content = f.read().strip()

    if not content:
        return []

    # --------------------------------------------------------
    # First try standard JSON.
    # --------------------------------------------------------

    try:
        return json.loads(
            content
        )

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Fall back to JSONL.
    # --------------------------------------------------------

    records = []

    for line_number, line in enumerate(
        content.splitlines(),
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        try:

            record = json.loads(
                line
            )

            records.append(
                record
            )

        except json.JSONDecodeError as exc:

            print(
                f"Warning: invalid JSON on "
                f"{manifest_path}, line "
                f"{line_number}: {exc}"
            )

    return records


def clean_corpus_sentence(
    text,
):
    """
    Clean a sentence before passing
    it to TTS.
    """

    if not isinstance(
        text,
        str,
    ):
        return None

    text = text.strip()

    if not text:
        return None

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if ch.isprintable()
    )

    text = text.strip()

    words = text.split()

    if len(words) < 3:
        return None

    if len(words) > 25:
        return None

    if len(text) > 180:
        return None

    return text


def load_librispeech_sentences(
    manifest_dir,
):
    """
    Find all JSON manifests and extract
    transcript text.

    Supports both:
        - Standard JSON
        - JSON Lines / JSONL
    """

    json_files = glob.glob(
        str(
            Path(manifest_dir)
            / "**/*.json"
        ),
        recursive=True,
    )

    if not json_files:

        raise RuntimeError(
            "No JSON manifests found in "
            f"{manifest_dir}"
        )

    print(
        f"\nFound {len(json_files)} JSON "
        "manifest(s)."
    )

    sentences = []

    for json_file in tqdm(
        json_files,
        desc="Reading corpus manifests",
    ):

        try:

            data = load_manifest_file(
                json_file
            )

            texts = (
                extract_text_from_json_object(
                    data
                )
            )

            for text in texts:

                cleaned = (
                    clean_corpus_sentence(
                        text
                    )
                )

                if cleaned:
                    sentences.append(
                        cleaned
                    )

        except Exception as exc:

            print(
                f"Warning: could not read "
                f"{json_file}: {exc}"
            )

    # Remove duplicates while preserving order.
    sentences = list(
        dict.fromkeys(
            sentences
        )
    )

    if not sentences:

        raise RuntimeError(
            "No usable sentences were found "
            "in the LibriSpeech-PC manifests."
        )

    print(
        f"Loaded {len(sentences):,} unique "
        "general speech sentences."
    )

    return sentences


# ============================================================
# TTS voice validation
# ============================================================

async def fetch_valid_voice_names():

    voices = await edge_tts.list_voices()

    return {
        voice["ShortName"]
        for voice in voices
    }


def filter_live_voices():

    print(
        "\nChecking Edge TTS voices..."
    )

    try:

        available = asyncio.run(
            fetch_valid_voice_names()
        )

    except Exception as exc:

        print(
            f"Could not query Edge TTS voices: "
            f"{exc}"
        )

        return VOICES

    valid = [
        voice
        for voice in VOICES
        if voice in available
    ]

    print(
        f"Available configured voices: "
        f"{len(valid)}/{len(VOICES)}"
    )

    if not valid:

        raise RuntimeError(
            "No configured Edge TTS voices "
            "are available."
        )

    return valid


# ============================================================
# TTS pool
# ============================================================

def build_tts_pool(
    texts,
    num_samples,
    voices,
):
    """
    Create random TTS jobs.

    Each job contains:
        text
        voice
        rate
        pitch
    """

    pool = []

    for _ in range(
        num_samples
    ):

        text = random.choice(
            texts
        )

        voice = random.choice(
            voices
        )

        rate = random.choice(
            RATE_VARIATIONS
        )

        pitch = random.choice(
            PITCH_VARIATIONS
        )

        pool.append(
            {
                "text": text,
                "voice": voice,
                "rate": rate,
                "pitch": pitch,
            }
        )

    return pool


# ============================================================
# Manifest
# ============================================================

class ManifestWriter:

    def __init__(
        self,
        path,
    ):
        self.path = path

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        self.file = open(
            path,
            "a",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.writer(
            self.file
        )

    def write(
        self,
        filename,
        label,
        text="",
    ):
        self.writer.writerow(
            [
                filename,
                label,
                text,
            ]
        )

        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================
# Existing file counting
# ============================================================

def count_existing(
    directory,
):

    if not os.path.exists(
        directory
    ):
        return 0

    return len(
        glob.glob(
            os.path.join(
                directory,
                "*.wav",
            )
        )
    )


# ============================================================
# Process and save
# ============================================================

def process_and_save(
    audio,
    output_path,
    noise_files=None,
    use_pseudo_speaker=False,
):

    audio = augment_speech_clip(
        audio,
        noise_files=noise_files,
        use_pseudo_speaker=use_pseudo_speaker,
    )

    write_wav_mono(
        output_path,
        audio,
    )


# ============================================================
# Wake word generation
# ============================================================

def generate_wake_word_dataset(
    args,
    voices,
    noise_files,
    manifest_writer,
    combined_writer,
):

    output_dir = Path(
        args.out_dir
    )

    wake_dir = (
        output_dir
        / "wake_word"
    )

    wake_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = count_existing(
        wake_dir
    )

    remaining = max(
        args.wake_clips - existing,
        0,
    )

    print(
        "\nWake-word dataset"
    )

    print(
        f"Existing : {existing}"
    )

    print(
        f"Target   : {args.wake_clips}"
    )

    print(
        f"Generate : {remaining}"
    )

    if remaining == 0:
        return

    wake_word = args.wake_word

    # --------------------------------------------------------
    # Own voice recordings
    # --------------------------------------------------------

    own_files = []

    if args.my_voice_dir:

        patterns = [
            "*.wav",
            "*.mp3",
            "*.m4a",
            "*.ogg",
            "*.flac",
        ]

        for pattern in patterns:

            own_files.extend(
                glob.glob(
                    os.path.join(
                        args.my_voice_dir,
                        pattern,
                    )
                )
            )

    own_variants = []

    for path in own_files:

        own_variants.append(
            (
                path,
                False,
            )
        )

        for _ in range(
            args.pseudo_speakers_per_recording
        ):

            own_variants.append(
                (
                    path,
                    True,
                )
            )

    # --------------------------------------------------------
    # Process own recordings first
    # --------------------------------------------------------

    generated = 0

    for path, pseudo in tqdm(
        own_variants,
        desc="Processing own wake-word recordings",
    ):

        if generated >= remaining:
            break

        try:

            if path.lower().endswith(
                ".wav"
            ):

                audio = load_wav_mono(
                    path
                )

            else:

                with tempfile.TemporaryDirectory() as tmp:

                    wav_path = (
                        Path(tmp)
                        / "input.wav"
                    )

                    any_to_wav(
                        path,
                        wav_path,
                    )

                    audio = load_wav_mono(
                        wav_path
                    )

            audio = trim_and_pad_to_length(
                audio
            )

            output_path = (
                wake_dir
                / (
                    f"wake_{existing + generated:05d}"
                    "_own.wav"
                )
            )

            process_and_save(
                audio,
                output_path,
                noise_files,
                use_pseudo_speaker=pseudo,
            )

            manifest_writer.write(
                output_path.name,
                "wake_word",
                wake_word,
            )

            combined_writer.write(
                output_path.name,
                "wake_word",
                wake_word,
            )

            generated += 1

        except Exception as exc:

            print(
                f"\nWarning: failed to process "
                f"{path}: {exc}"
            )

    # --------------------------------------------------------
    # Generate remaining wake-word samples using Edge TTS
    # --------------------------------------------------------

    remaining_tts = (
        remaining - generated
    )

    if remaining_tts <= 0:
        return

    print(
        f"\nGenerating {remaining_tts} "
        "wake-word TTS samples..."
    )

    tts_pool = build_tts_pool(
        [wake_word],
        remaining_tts,
        voices,
    )

    for job in tqdm(
        tts_pool,
        desc="Generating wake-word TTS",
    ):

        output_index = (
            existing
            + generated
        )

        try:

            with tempfile.TemporaryDirectory() as tmp:

                mp3_path = (
                    Path(tmp)
                    / "speech.mp3"
                )

                wav_path = (
                    Path(tmp)
                    / "speech.wav"
                )

                synthesize(
                    job["text"],
                    job["voice"],
                    mp3_path,
                    job["rate"],
                    job["pitch"],
                )

                mp3_to_wav(
                    mp3_path,
                    wav_path,
                )

                audio = load_wav_mono(
                    wav_path
                )

            output_path = (
                wake_dir
                / (
                    f"wake_{output_index:05d}_"
                    f"{safe_filename(job['voice'])}.wav"
                )
            )

            process_and_save(
                audio,
                output_path,
                noise_files,
                use_pseudo_speaker=True,
            )

            manifest_writer.write(
                output_path.name,
                "wake_word",
                wake_word,
            )

            combined_writer.write(
                output_path.name,
                "wake_word",
                wake_word,
            )

            generated += 1

        except Exception as exc:

            print(
                f"\nWarning: wake TTS generation "
                f"failed: {exc}"
            )

    print(
        f"\nWake-word generation complete: "
        f"{generated} new samples."
    )


# ============================================================
# Unknown generation
# ============================================================

def generate_unknown_dataset(
    args,
    voices,
    noise_files,
    unknown_sentences,
    manifest_writer,
    combined_writer,
):

    output_dir = Path(
        args.out_dir
    )

    unknown_dir = (
        output_dir
        / "unknown"
    )

    unknown_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = count_existing(
        unknown_dir
    )

    remaining = max(
        args.unknown_clips - existing,
        0,
    )

    print(
        "\nUnknown dataset"
    )

    print(
        f"Existing : {existing}"
    )

    print(
        f"Target   : {args.unknown_clips}"
    )

    print(
        f"Generate : {remaining}"
    )

    if remaining == 0:
        return

    print(
        "\nUsing random sentences from "
        "LibriSpeech-PC."
    )

    tts_pool = build_tts_pool(
        unknown_sentences,
        remaining,
        voices,
    )

    generated = 0

    for job in tqdm(
        tts_pool,
        desc="Generating unknown speech",
    ):

        output_index = (
            existing
            + generated
        )

        try:

            with tempfile.TemporaryDirectory() as tmp:

                mp3_path = (
                    Path(tmp)
                    / "speech.mp3"
                )

                wav_path = (
                    Path(tmp)
                    / "speech.wav"
                )

                synthesize(
                    job["text"],
                    job["voice"],
                    mp3_path,
                    job["rate"],
                    job["pitch"],
                )

                mp3_to_wav(
                    mp3_path,
                    wav_path,
                )

                audio = load_wav_mono(
                    wav_path
                )

            output_path = (
                unknown_dir
                / (
                    f"unknown_{output_index:05d}_"
                    f"{safe_filename(job['voice'])}.wav"
                )
            )

            process_and_save(
                audio,
                output_path,
                noise_files,
                use_pseudo_speaker=True,
            )

            manifest_writer.write(
                output_path.name,
                "unknown",
                job["text"],
            )

            combined_writer.write(
                output_path.name,
                "unknown",
                job["text"],
            )

            generated += 1

        except Exception as exc:

            print(
                f"\nWarning: unknown TTS generation "
                f"failed: {exc}"
            )

    print(
        f"\nUnknown generation complete: "
        f"{generated} new samples."
    )


# ============================================================
# Noise generation
# ============================================================

def generate_noise_dataset(
    args,
    noise_files,
    manifest_writer,
    combined_writer,
):

    output_dir = Path(
        args.out_dir
    )

    noise_dir = (
        output_dir
        / "noise"
    )

    noise_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = count_existing(
        noise_dir
    )

    remaining = max(
        args.noise_clips - existing,
        0,
    )

    print(
        "\nNoise dataset"
    )

    print(
        f"Existing : {existing}"
    )

    print(
        f"Target   : {args.noise_clips}"
    )

    print(
        f"Generate : {remaining}"
    )

    if remaining == 0:
        return

    for i in tqdm(
        range(remaining),
        desc="Generating noise",
    ):

        output_index = (
            existing + i
        )

        if (
            noise_files
            and random.random() < 0.8
        ):

            audio = (
                load_random_real_noise_chunk(
                    noise_files
                )
            )

            if audio is None:
                audio = synthetic_noise_clip()

        else:

            audio = synthetic_noise_clip()

        audio = trim_and_pad_to_length(
            audio
        )

        audio = normalize_audio(
            audio,
            random.uniform(
                0.2,
                0.9,
            ),
        )

        output_path = (
            noise_dir
            / f"noise_{output_index:05d}.wav"
        )

        write_wav_mono(
            output_path,
            audio,
        )

        manifest_writer.write(
            output_path.name,
            "noise",
            "",
        )

        combined_writer.write(
            output_path.name,
            "noise",
            "",
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a robust KWS dataset "
            "with wake-word, unknown and noise classes."
        )
    )

    parser.add_argument(
        "--wake-word",
        required=True,
        help=(
            "Wake word/phrase, e.g. "
            "'Hey Ruby'."
        ),
    )

    parser.add_argument(
        "--out-dir",
        default="dataset",
        help="Output dataset directory.",
    )

    parser.add_argument(
        "--my-voice-dir",
        default=None,
        help=(
            "Directory containing your own "
            "wake-word recordings."
        ),
    )

    parser.add_argument(
        "--wake-clips",
        type=int,
        default=900,
        help="Total number of wake-word clips.",
    )

    parser.add_argument(
        "--unknown-clips",
        type=int,
        default=900,
        help="Total number of unknown speech clips.",
    )

    parser.add_argument(
        "--noise-clips",
        type=int,
        default=400,
        help="Total number of noise clips.",
    )

    parser.add_argument(
        "--pseudo-speakers-per-recording",
        type=int,
        default=25,
        help=(
            "Number of pseudo-speaker variants "
            "per own recording."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--cache-dir",
        default=".dataset_cache",
        help=(
            "Cache directory for downloaded "
            "corpus/noise data."
        ),
    )

    parser.add_argument(
        "--skip-noise-download",
        action="store_true",
        help=(
            "Do not attempt to download MS-SNSD. "
            "Use synthetic noise only."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    output_dir = Path(
        args.out_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir = Path(
        args.cache_dir
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Voices
    # --------------------------------------------------------

    voices = filter_live_voices()

    # --------------------------------------------------------
    # LibriSpeech-PC
    # --------------------------------------------------------

    print(
        "\nPreparing general speech corpus..."
    )

    librispeech_dir = (
        download_librispeech_pc(
            cache_dir
        )
    )

    unknown_sentences = (
        load_librispeech_sentences(
            librispeech_dir
        )
    )

    # --------------------------------------------------------
    # Noise corpus
    # --------------------------------------------------------

    noise_files = []

    if not args.skip_noise_download:

        noise_root = (
            try_download_noise_corpus(
                cache_dir
            )
        )

        if noise_root:

            noise_files = find_noise_files(
                noise_root
            )

    print(
        f"\nReal noise files found: "
        f"{len(noise_files)}"
    )

    # --------------------------------------------------------
    # Manifest files
    # --------------------------------------------------------

    manifest_dir = (
        output_dir
        / "manifests"
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    wake_manifest = ManifestWriter(
        str(
            manifest_dir
            / "wake_word.csv"
        )
    )

    unknown_manifest = ManifestWriter(
        str(
            manifest_dir
            / "unknown.csv"
        )
    )

    noise_manifest = ManifestWriter(
        str(
            manifest_dir
            / "noise.csv"
        )
    )

    combined_manifest = ManifestWriter(
        str(
            manifest_dir
            / "dataset.csv"
        )
    )

    try:

        # ----------------------------------------------------
        # Wake word
        # ----------------------------------------------------

        generate_wake_word_dataset(
            args,
            voices,
            noise_files,
            wake_manifest,
            combined_manifest,
        )

        # ----------------------------------------------------
        # Unknown
        # ----------------------------------------------------

        generate_unknown_dataset(
            args,
            voices,
            noise_files,
            unknown_sentences,
            unknown_manifest,
            combined_manifest,
        )

        # ----------------------------------------------------
        # Noise
        # ----------------------------------------------------

        generate_noise_dataset(
            args,
            noise_files,
            noise_manifest,
            combined_manifest,
        )

    finally:

        wake_manifest.close()
        unknown_manifest.close()
        noise_manifest.close()
        combined_manifest.close()

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    wake_count = count_existing(
        output_dir / "wake_word"
    )

    unknown_count = count_existing(
        output_dir / "unknown"
    )

    noise_count = count_existing(
        output_dir / "noise"
    )

    total = (
        wake_count
        + unknown_count
        + noise_count
    )

    print("\n")
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Wake word : {wake_count}"
    )

    print(
        f"Unknown   : {unknown_count}"
    )

    print(
        f"Noise     : {noise_count}"
    )

    print(
        f"Total     : {total}"
    )

    print("=" * 60)

    print(
        "\nDataset structure:"
    )

    print(
        f"{output_dir}/"
    )

    print(
        "├── wake_word/"
    )

    print(
        "├── unknown/"
    )

    print(
        "├── noise/"
    )

    print(
        "└── manifests/"
    )

    print(
        "    ├── wake_word.csv"
    )

    print(
        "    ├── unknown.csv"
    )

    print(
        "    ├── noise.csv"
    )

    print(
        "    └── dataset.csv"
    )

    print(
        "\nYou can resume generation by running "
        "the same command again."
    )


if __name__ == "__main__":
    main()