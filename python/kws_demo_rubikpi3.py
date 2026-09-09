"""
Keyword Spotting (KWS) ONNX Inference Demo.

Runs a KWS ONNX model using Edge Impulse-compatible MFCC preprocessing.
Supports live microphone inference and WAV file inference at 16 kHz or
48 kHz.

Usage:

    # Live microphone inference
    python kws_demo_rubikpi3.py --model model.onnx

    # Inference on a 16 kHz WAV file
    python kws_demo_rubikpi3.py --model model.onnx --wav test.wav

    # Inference on a 48 kHz WAV file
    python kws_demo_rubikpi3.py --model model.onnx --wav_48k test.wav

    # Set wake-word detection threshold
    python kws_demo_rubikpi3.py --model model.onnx --threshold 0.8

    # Quantized model with input scale and zero point
    python kws_demo_rubikpi3.py --model model.onnx \
        --input-scale 0.02 \
        --input-zero-point -128

Model classes:
    0 - Wake word
    1 - Noise
    2 - Unknown
"""

import argparse
import queue
import time
from collections import deque

import numpy as np
from scipy.fft import dct
# SpeechPy 2.4 compatibility with NumPy 2.x
if not hasattr(np.lib, "pad"):
        np.lib.pad = np.pad

from scipy.signal import resample_poly

import sounddevice as sd
import speechpy
import onnxruntime as ort
import traceback


import soundfile as sf


# ============================================================
# Audio / Edge Impulse MFCC configuration
# ============================================================

# Physical microphone capture rate.
# RUBIK Pi 3 headset capture device supports 48 kHz.
MIC_SAMPLE_RATE = 48000
MIC_DEVICE = 2

# Model / DSP sample rate.
# Edge Impulse model expects 16 kHz audio.
SAMPLE_RATE = 16000

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

NUM_COEFFICIENTS = 13

FRAME_LENGTH = 0.020       # 20 ms
FRAME_STRIDE = 0.020       # 20 ms

NUM_FILTERS = 32
FFT_LENGTH = 256

LOW_FREQUENCY = 0
HIGH_FREQUENCY = 8000      # sample_rate / 2

PREEMPHASIS_COEFF = 0.98
PREEMPHASIS_SHIFT = 1

CMVN_WINDOW = 101

EXPECTED_FEATURES = 650

CLASS_NAMES = [
    "Wake word",
    "Noise",
    "Unknown",
]


# ============================================================
# Audio capture
# ============================================================

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)

    # Mono audio
    audio = indata[:, 0].copy()

    audio_queue.put(audio)


def resample_to_model_rate(audio):
    """
    Convert microphone audio from 48 kHz to the
    16 kHz sample rate expected by the model.
    """

    audio = np.asarray(audio, dtype=np.float32)

    # 48000 Hz -> 16000 Hz
    # 48000 / 3 = 16000
    audio_16k = resample_poly(
        audio,
        up=1,
        down=3
    )

    return audio_16k.astype(np.float32)


# ============================================================
# MFCC / Edge Impulse feature extraction
# ============================================================

def preemphasis(signal):
    """Edge Impulse pre-emphasis: y[0]=x[0], y[i]=x[i]-0.98*x[i-1]."""
    signal = np.asarray(signal, dtype=np.float32)
    output = np.empty_like(signal)
    output[0] = signal[0]
    output[1:] = signal[1:] - PREEMPHASIS_COEFF * signal[:-1]
    return output


def frame_signal(signal):
    """Frame the 1-second signal exactly as the Edge Impulse v4 pipeline."""
    frame_length = int(round(SAMPLE_RATE * FRAME_LENGTH))
    frame_stride = int(round(SAMPLE_RATE * FRAME_STRIDE))

    num_frames = int(np.floor((len(signal) - frame_length) / frame_stride)) + 1

    frames = np.zeros((num_frames, frame_length), dtype=np.float32)

    for i in range(num_frames):
        start = i * frame_stride
        end = start + frame_length
        if end <= len(signal):
            frames[i] = signal[start:end]
        else:
            available = len(signal) - start
            if available > 0:
                frames[i, :available] = signal[start:]

    return frames


def hz_to_mel(hz):
    return 1127.0 * np.log(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def calculate_mel_bins():
    # Edge Impulse v4 uses max_bin = FFT_LENGTH and
    # floor((max_bin + 1) * Hz / sample_rate).
    high_frequency = HIGH_FREQUENCY
    if high_frequency == 0:
        high_frequency = SAMPLE_RATE / 2.0

    low_mel = hz_to_mel(LOW_FREQUENCY)
    high_mel = hz_to_mel(high_frequency)

    mel_points = np.linspace(low_mel, high_mel, NUM_FILTERS + 2)
    hz_points = mel_to_hz(mel_points)

    max_bin = FFT_LENGTH
    bins = np.floor((max_bin + 1) * hz_points / SAMPLE_RATE).astype(np.int32)

    return hz_points, bins


def power_spectrum(frame):
    fft_output = np.fft.rfft(frame, n=FFT_LENGTH)
    magnitude = np.abs(fft_output)
    power = (1.0 / FFT_LENGTH) * (magnitude * magnitude)
    return power.astype(np.float32)


def mel_filterbank_energy(power, bins):
    num_power_bins = FFT_LENGTH // 2 + 1
    energies = np.zeros(NUM_FILTERS, dtype=np.float32)

    frame_energy = np.sum(power, dtype=np.float64)

    for i in range(NUM_FILTERS):
        left = int(bins[i])
        middle = int(bins[i + 1])
        right = int(bins[i + 2])

        value = 0.0

        # Match the generated Edge Impulse implementation:
        # explicitly add the middle bin first.
        if middle < num_power_bins:
            value += power[middle]

        for bin_index in range(left + 1, right):
            if bin_index < middle:
                denominator = middle - left
                if denominator != 0:
                    value += ((bin_index - left) / denominator) * power[bin_index]
            elif bin_index > middle:
                denominator = right - middle
                if denominator != 0:
                    value += ((right - bin_index) / denominator) * power[bin_index]

        energies[i] = value

    energies[energies == 0] = 1e-10
    if frame_energy == 0:
        frame_energy = 1e-10

    return energies, frame_energy


def calculate_mfcc(frames, bins):
    num_frames = frames.shape[0]
    mfcc = np.zeros((num_frames, NUM_COEFFICIENTS), dtype=np.float32)

    for frame_index in range(num_frames):
        frame = frames[frame_index]

        power = power_spectrum(frame)
        mfe, energy = mel_filterbank_energy(power, bins)

        log_mfe = np.log(mfe)

        # Edge Impulse uses DCT-II with orthonormal normalization.
        dct_output = dct(log_mfe, type=2, norm="ortho").astype(np.float32)

        # Edge Impulse replaces the DC coefficient with log(frame_energy).
        dct_output[0] = np.log(energy)

        mfcc[frame_index] = dct_output[:NUM_COEFFICIENTS]

    return mfcc


def cmvnw(features):
    """Edge Impulse CMVN: 101-frame symmetric window, variance normalization enabled."""
    features = np.asarray(features, dtype=np.float32)
    num_frames, num_coeffs = features.shape

    pad_size = (CMVN_WINDOW - 1) // 2

    # Mean normalization.
    padded = np.pad(
        features,
        ((pad_size, pad_size), (0, 0)),
        mode="symmetric",
    )

    mean_normalized = np.zeros_like(features)

    for i in range(num_frames):
        window = padded[i:i + CMVN_WINDOW]
        mean = np.mean(window, axis=0)
        mean_normalized[i] = features[i] - mean

    # Variance normalization.
    padded = np.pad(
        mean_normalized,
        ((pad_size, pad_size), (0, 0)),
        mode="symmetric",
    )

    normalized = np.zeros_like(features)

    for i in range(num_frames):
        window = padded[i:i + CMVN_WINDOW]
        std = np.std(window, axis=0)
        normalized[i] = mean_normalized[i] / (std + 1e-10)

    return normalized


def extract_features(audio):
    """
    Convert 1 second of 16 kHz audio into the 650 Edge Impulse features.

    This intentionally mirrors test_mfcc_v2.py instead of using SpeechPy's
    MFCC implementation, so the live demo uses the same preprocessing path
    that produced the validated 650-value feature vector.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) != WINDOW_SAMPLES:
        raise ValueError(
            f"Expected {WINDOW_SAMPLES} samples, got {len(audio)}"
        )

    # 1. Edge Impulse pre-emphasis.
    preemphasized = preemphasis(audio)

    # 2. 20 ms / 20 ms framing -> 50 frames for 1 second.
    frames = frame_signal(preemphasized)

    if frames.shape != (50, 320):
        raise RuntimeError(
            f"Unexpected frame shape: {frames.shape}. Expected (50, 320)."
        )

    # 3. Edge Impulse v4 mel filterbank + power spectrum + DCT-II.
    _, mel_bins = calculate_mel_bins()
    mfcc = calculate_mfcc(frames, mel_bins)

    if mfcc.shape != (50, 13):
        raise RuntimeError(
            f"Unexpected MFCC shape: {mfcc.shape}. Expected (50, 13)."
        )

    # 4. Edge Impulse CMVN with variance normalization enabled.
    normalized = cmvnw(mfcc)

    # 5. Frame-major flatten: frame0[13], frame1[13], ...
    features = normalized.astype(np.float32).reshape(-1)

    if features.shape != (EXPECTED_FEATURES,):
        raise RuntimeError(
            f"Unexpected feature shape: {features.shape}. "
            f"Expected ({EXPECTED_FEATURES},)"
        )

    return features


_shape_printed = False


def print_once_shape(mfcc):
    global _shape_printed

    if not _shape_printed:
        print(f"\nMFCC shape: {mfcc.shape}")
        print(f"Expected features: {mfcc.size}")
        _shape_printed = True


# ============================================================
# ONNX helpers
# ============================================================

def get_onnx_dtype(onnx_type):
    """
    Convert ONNX tensor type to NumPy dtype.
    """

    if onnx_type == "tensor(float)":
        return np.float32

    if onnx_type == "tensor(float16)":
        return np.float16

    if onnx_type == "tensor(int8)":
        return np.int8

    if onnx_type == "tensor(uint8)":
        return np.uint8

    if onnx_type == "tensor(int16)":
        return np.int16

    raise RuntimeError(
        f"Unsupported ONNX input type: {onnx_type}"
    )


def inspect_model(session):
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print("\n==============================")
    print("ONNX MODEL")
    print("==============================")

    print("Input name :", input_info.name)
    print("Input shape:", input_info.shape)
    print("Input type :", input_info.type)

    print("Output name :", output_info.name)
    print("Output shape:", output_info.shape)
    print("Output type :", output_info.type)

    return input_info, output_info


# ============================================================
# Quantization
# ============================================================

def quantize_input(features, scale, zero_point, dtype):
    """
    Quantize FP32 features to INT8/UINT8.
    """

    q = np.round(features / scale + zero_point)

    if dtype == np.int8:
        q = np.clip(q, -128, 127)

    elif dtype == np.uint8:
        q = np.clip(q, 0, 255)

    else:
        raise RuntimeError(
            f"Unsupported quantization dtype: {dtype}"
        )

    return q.astype(dtype)


def prepare_input(features, input_info, scale=None, zero_point=None):
    """
    Prepare input according to ONNX model input dtype.
    """

    dtype = get_onnx_dtype(input_info.type)

    # --------------------------------------------------------
    # FP32
    # --------------------------------------------------------

    if dtype == np.float32:
        return features.astype(np.float32)

    # --------------------------------------------------------
    # FP16
    # --------------------------------------------------------

    if dtype == np.float16:
        return features.astype(np.float16)

    # --------------------------------------------------------
    # INT8 / UINT8
    # --------------------------------------------------------

    if dtype in (np.int8, np.uint8):

        if scale is None or zero_point is None:
            raise RuntimeError(
                "\nModel expects integer input:\n"
                f"    {input_info.type}\n\n"
                "But input quantization parameters were not supplied.\n"
                "Run the script with:\n\n"
                "    --input-scale <scale> "
                "--input-zero-point <zero_point>\n"
            )

        return quantize_input(
            features,
            scale,
            zero_point,
            dtype
        )

    raise RuntimeError(
        f"Unsupported model input dtype: {input_info.type}"
    )


# ============================================================
# ONNX inference
# ============================================================

def run_inference(
    session,
    input_info,
    output_info,
    features,
    input_scale=None,
    input_zero_point=None
):

    input_data = prepare_input(
        features,
        input_info,
        input_scale,
        input_zero_point
    )


    # ONNX input normally needs [1, 650]
    input_data = input_data.reshape(1, -1)

    outputs = session.run(
        [output_info.name],
        {
            input_info.name: input_data
        }
    )

    output = outputs[0]

    # Remove batch dimension
    output = np.squeeze(output)

    return output


# ============================================================
# Output processing
# ============================================================

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def process_output(output, output_info):

    output = np.asarray(output)

    # --------------------------------------------------------
    # If output is already probabilities
    # --------------------------------------------------------

    if output.ndim == 1 and len(output) == 3:

        # Check whether output already looks like probabilities
        if (
            np.all(output >= 0)
            and
            np.all(output <= 1)
            and
            abs(np.sum(output) - 1.0) < 0.05
        ):
            probabilities = output.astype(np.float32)

        else:
            probabilities = softmax(output.astype(np.float32))

    else:
        raise RuntimeError(
            f"Unexpected output shape: {output.shape}"
        )

    predicted_index = int(np.argmax(probabilities))

    return probabilities, predicted_index

def run_wav_48k(model_path,
    wav_path,
    input_scale=None,
    input_zero_point=None):
    """
    Test a 48 kHz WAV file.

    Pipeline:
        48 kHz WAV
            ↓
        mono
            ↓
        1-second / 48000 samples
            ↓
        resample 48 kHz -> 16 kHz
            ↓
        16000 samples
            ↓
        Edge Impulse DSP
            ↓
        ONNX inference
    """

    import soundfile as sf
    print("Inside run wav 48")
    print(f"wav_path is {wav_path}")

    # --------------------------------------------------
    # Load WAV
    # --------------------------------------------------
    audio_48k, sample_rate = sf.read(
        wav_path,
        dtype="float32"
    )

    print(f"WAV sample rate : {sample_rate}")
    print(f"WAV shape       : {audio_48k.shape}")

    # --------------------------------------------------
    # Verify sample rate
    # --------------------------------------------------
    if sample_rate != 48000:
        raise ValueError(
            f"Expected a 48 kHz WAV, but got {sample_rate} Hz"
        )

    # --------------------------------------------------
    # Convert stereo -> mono
    # --------------------------------------------------
    if audio_48k.ndim > 1:
        audio_48k = np.mean(audio_48k, axis=1)

    # --------------------------------------------------
    # Need at least 1 second
    # --------------------------------------------------
    if len(audio_48k) < 48000:
        raise ValueError(
            f"WAV is too short: {len(audio_48k)} samples. "
            f"Need at least 48000 samples."
        )

    # --------------------------------------------------
    # Take first 1 second
    # --------------------------------------------------
    audio_48k = audio_48k[:48000]

    print(f"Using {len(audio_48k)} samples at 48 kHz")

    # --------------------------------------------------
    # Resample 48 kHz -> 16 kHz
    # --------------------------------------------------
    audio = resample_to_model_rate(audio_48k)

    # Make absolutely sure we have exactly 16000
    audio = audio[:WINDOW_SAMPLES]

    if len(audio) != WINDOW_SAMPLES:
        raise ValueError(
            f"Resampling produced {len(audio)} samples, "
            f"expected {WINDOW_SAMPLES}"
        )

    print(f"Resampled to {len(audio)} samples at 16 kHz")

    # ------------------------------------------------
    # MFCC preprocessing
    # ------------------------------------------------

    features = extract_features(audio)

    print("Feature shape   :", features.shape)

    # ------------------------------------------------
    # ONNX
    # ------------------------------------------------

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_info, output_info = inspect_model(session)

    output = run_inference(
        session,
        input_info,
        output_info,
        features,
        input_scale,
        input_zero_point
    )

    probabilities, predicted_index = process_output(
        output,
        output_info
    )

    print("\n==============================")
    print("RESULT")
    print("==============================")
    print(f"Wake    : {probabilities[0]:.4f}")
    print(f"Noise   : {probabilities[1]:.4f}")
    print(f"Unknown : {probabilities[2]:.4f}")
    print(f"Result  : {CLASS_NAMES[predicted_index]}")

    print("=" * 60)

def run_wav(
    model_path,
    wav_path,
    input_scale=None,
    input_zero_point=None
):
    print("\nLoading WAV...")

    audio, sample_rate = sf.read(
        wav_path,
        dtype="float32"
    )

    print("WAV sample rate :", sample_rate)
    print("WAV samples     :", len(audio))

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz WAV, got {sample_rate} Hz"
        )

    # If stereo, convert to mono
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Need exactly 1 second
    if len(audio) < WINDOW_SAMPLES:
        raise ValueError(
            f"WAV is too short. "
            f"Need {WINDOW_SAMPLES} samples, got {len(audio)}"
        )

    audio = audio[:WINDOW_SAMPLES]

    print("Using first 1 second of WAV")
    print("Audio shape     :", audio.shape)
    print("Audio min       :", np.min(audio))
    print("Audio max       :", np.max(audio))
    print("Audio RMS       :", np.sqrt(np.mean(audio ** 2)))

    # ------------------------------------------------
    # MFCC preprocessing
    # ------------------------------------------------

    features = extract_features(audio)

    print("Feature shape   :", features.shape)

    # ------------------------------------------------
    # ONNX
    # ------------------------------------------------

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_info, output_info = inspect_model(session)

    output = run_inference(
        session,
        input_info,
        output_info,
        features,
        input_scale,
        input_zero_point
    )

    probabilities, predicted_index = process_output(
        output,
        output_info
    )

    print("\n==============================")
    print("RESULT")
    print("==============================")
    print(f"Wake    : {probabilities[0]:.4f}")
    print(f"Noise   : {probabilities[1]:.4f}")
    print(f"Unknown : {probabilities[2]:.4f}")
    print(f"Result  : {CLASS_NAMES[predicted_index]}")

# ============================================================
# Live KWS
# ============================================================

def run_live(
    model_path,
    input_scale=None,
    input_zero_point=None,
    threshold=0.70,
    inference_interval=0.25
):
    print("\nLoading ONNX model...")
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    input_info, output_info = inspect_model(session)

    print("\nStarting microphone...")
    print(f"Microphone  : device {MIC_DEVICE}")
    print(f"Mic rate    : {MIC_SAMPLE_RATE} Hz")
    print(f"Model rate  : {SAMPLE_RATE} Hz")
    print(f"Window      : {WINDOW_SECONDS} sec")
    print(f"Threshold   : {threshold}")
    print(f"Interval    : {inference_interval} sec")

    print("\nPress Ctrl+C to stop.\n")

    # Rolling 1-second audio buffer at microphone rate
    mic_window_samples = int(
        MIC_SAMPLE_RATE * WINDOW_SECONDS
    )

    audio_buffer = deque(
        maxlen=mic_window_samples
    )

    last_inference = 0
    last_wake_detection = 0

    with sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=1024,
        device=MIC_DEVICE,
        callback=audio_callback
    ):
        while True:
            try:
                while True:
                    chunk = audio_queue.get_nowait()

                    # Raw microphone audio is 48 kHz
                    audio_buffer.extend(chunk)

            except queue.Empty:
                pass

            # Wait until we have a complete
            # 1-second window at 48 kHz
            if len(audio_buffer) < mic_window_samples:
                time.sleep(0.01)
                continue

            current_time = time.time()

            if (
                current_time - last_inference
                <
                inference_interval
            ):
                time.sleep(0.01)
                continue

            last_inference = current_time

            # ------------------------------------------------
            # 1. Get 1-second 48 kHz microphone audio
            # ------------------------------------------------

            audio_48k = np.asarray(
                audio_buffer,
                dtype=np.float32
            )

            # ------------------------------------------------
            # 2. Resample 48 kHz -> 16 kHz
            # ------------------------------------------------

            audio = resample_to_model_rate(
                audio_48k
            )

            # Make sure we have exactly
            # 16000 samples for the model
            audio = audio[:WINDOW_SAMPLES]

            # ------------------------------------------------
            # 3. Extract Edge Impulse MFCC features
            # ------------------------------------------------

            try:
                features = extract_features(audio)
            except Exception as e:
                print("\nFeature extraction error:")
                print(e)
                traceback.print_exc()
                continue

            # ------------------------------------------------
            # 4. ONNX inference
            # ------------------------------------------------

            try:
                output = run_inference(
                    session,
                    input_info,
                    output_info,
                    features,
                    input_scale,
                    input_zero_point
                )
            except Exception as e:
                print("\nONNX inference error:")
                print(e)
                continue

            # ------------------------------------------------
            # 5. Process output
            # ------------------------------------------------

            try:
                probabilities, predicted_index = process_output(
                    output,
                    output_info
                )
            except Exception as e:
                print("\nOutput processing error:")
                print(e)
                continue

            wake_prob = probabilities[0]
            #print(f"wake_prob is {wake_prob}\n")

            if (
                predicted_index == 0
                and
                wake_prob >= threshold
            ):
                print(
                    "\r"
                    f"Wake: {probabilities[0]:.3f}   "
                    f"Noise: {probabilities[1]:.3f}   "
                    f"Unknown: {probabilities[2]:.3f}   "
                    f"predicted_index: {predicted_index}  "
                    f"    -> {CLASS_NAMES[predicted_index]}",
                    end="",
                    flush=True
                )

            if (
                predicted_index == 0
                and
                wake_prob >= threshold
            ):
                #if current_time - last_wake_detection > 1.0:
                print("\n\n================================")
                print("       WAKE WORD DETECTED!")
                print("================================\n")
                    #last_wake_detection = current_time


def main():

    parser = argparse.ArgumentParser(
        description="Live Edge Impulse KWS ONNX demo"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to ONNX model"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Wake word confidence threshold"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Inference interval in seconds"
    )

    # Required only for integer-input ONNX models
    parser.add_argument(
        "--input-scale",
        type=float,
        default=None,
        help="Input quantization scale for INT8/UINT8 model"
    )

    parser.add_argument(
        "--input-zero-point",
        type=int,
        default=None,
        help="Input quantization zero point for INT8/UINT8 model"
    )
    
    parser.add_argument(
    "--wav",
    type=str,
    default=None,
    help="Run inference on a 16 kHz WAV file instead of microphone"
    )

    parser.add_argument(
        "--wav_48k",
        type=str,
        default=None,
        help="Run inference on a 48 kHz WAV file",
    )

    args = parser.parse_args()

    if args.wav:
        run_wav(
            model_path=args.model,
            wav_path=args.wav,
            input_scale=args.input_scale,
            input_zero_point=args.input_zero_point
        )
    elif args.wav_48k:
        print("Invoking run wav 48")
        run_wav_48k(
            model_path=args.model,
            wav_path=args.wav_48k,
            input_scale=args.input_scale,
            input_zero_point=args.input_zero_point
        )
   
    else:
        run_live(
            model_path=args.model,
            input_scale=args.input_scale,
            input_zero_point=args.input_zero_point,
            threshold=args.threshold,
            inference_interval=args.interval
        )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n\nStopped.")

    except Exception as e:
        print("\nERROR:")
        print(e)
