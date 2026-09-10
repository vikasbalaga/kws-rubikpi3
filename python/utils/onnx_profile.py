import onnxruntime as ort
import numpy as np
import json
import os
import sys
import time
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

NUM_WARMUP_RUNS = 20
NUM_PROFILE_RUNS = 100
TOP_N_OPERATORS = 10

INPUT_FEATURES = 650

QNN_BACKEND_PATH = "/usr/lib/libQnnHtp.so"


# ============================================================
# COMMAND LINE
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print("  python onnx_profile.py <model.onnx>")
    print()
    print("Examples:")
    print("  python onnx_profile.py kws_model_fp32.onnx")
    print("  python onnx_profile.py kws_model_qnn_uint16_uint8.onnx")
    sys.exit(1)


MODEL_PATH = sys.argv[1]


if not os.path.isfile(MODEL_PATH):
    print(f"ERROR: Model not found: {MODEL_PATH}")
    sys.exit(1)


# ============================================================
# CREATE PROFILE FILE NAME
# ============================================================

model_name = os.path.splitext(
    os.path.basename(MODEL_PATH)
)[0]

PROFILE_OUTPUT = f"{model_name}_profile.json"


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("ONNX RUNTIME MODEL PROFILING")
print("=" * 80)

print(f"\nModel:")
print(f"  {MODEL_PATH}")


# ============================================================
# ONNX RUNTIME INFORMATION
# ============================================================

print("\nONNX Runtime:")
print(f"  Version: {ort.__version__}")


available_providers = ort.get_available_providers()

print("\nAvailable Execution Providers:")

for provider in available_providers:
    print(f"  - {provider}")


# ============================================================
# CREATE SESSION
#
# Priority:
#
#     QNNExecutionProvider
#             ↓
#     CPUExecutionProvider
#
# QNN gets first chance to execute supported subgraphs.
# Unsupported parts automatically fall back to CPU.
#
# If QNN session creation itself fails, we create a
# CPU-only session.
# ============================================================

print()
print("=" * 80)
print("CREATING INFERENCE SESSION")
print("=" * 80)


session_options = ort.SessionOptions()

# Enable ORT profiling
session_options.enable_profiling = True

# Enable graph optimizations
session_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)


qnn_available = (
    "QNNExecutionProvider" in available_providers
)


session = None
using_qnn = False


# ------------------------------------------------------------
# Try QNN first
# ------------------------------------------------------------

if qnn_available:

    print("\nAttempting QNN HTP + CPU fallback...")

    providers = [
        (
            "QNNExecutionProvider",
            {
                "backend_path": QNN_BACKEND_PATH,
            }
        ),
        "CPUExecutionProvider",
    ]

    try:

        session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=session_options,
            providers=providers
        )

        using_qnn = True

        print("\nQNN session created successfully.")

    except Exception as e:

        print("\nWARNING: QNN session creation failed.")
        print(f"Reason: {e}")

        print("\nFalling back to CPUExecutionProvider...")

        # Create a fresh SessionOptions because profiling state
        # should belong to the new session.
        session_options = ort.SessionOptions()

        session_options.enable_profiling = True

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=session_options,
            providers=[
                "CPUExecutionProvider"
            ]
        )

        using_qnn = False

else:

    print("\nQNNExecutionProvider is not available.")

    print("Using CPUExecutionProvider.")

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=session_options,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    using_qnn = False


# ============================================================
# SESSION INFORMATION
# ============================================================

print()
print("=" * 80)
print("SESSION INFORMATION")
print("=" * 80)

print("\nExecution Providers Used By Session:")

for provider in session.get_providers():
    print(f"  - {provider}")


print("\nProvider Options:")

try:

    provider_options = session.get_provider_options()

    for provider, options in provider_options.items():

        print(f"\n  {provider}:")

        if options:

            for key, value in options.items():
                print(f"    {key}: {value}")

        else:
            print("    No provider-specific options")

except Exception as e:

    print(f"  Unable to get provider options: {e}")


# ============================================================
# MODEL INPUT INFORMATION
# ============================================================

inputs = session.get_inputs()

print()
print("=" * 80)
print("MODEL INPUT")
print("=" * 80)

for inp in inputs:

    print(f"\nName : {inp.name}")
    print(f"Type : {inp.type}")
    print(f"Shape: {inp.shape}")


# ============================================================
# CREATE TEST INPUT
# ============================================================

input_name = inputs[0].name


# ------------------------------------------------------------
# Determine input shape from model
# ------------------------------------------------------------

input_shape = inputs[0].shape

print("\nInput shape used for benchmark:")

print(f"  {input_shape}")


# ------------------------------------------------------------
# For your KWS model this should be [1, 650].
# ------------------------------------------------------------

if len(input_shape) == 2:

    batch_size = 1

    feature_size = input_shape[1]

    if isinstance(feature_size, int):

        input_features = feature_size

    else:

        input_features = INPUT_FEATURES

    input_data = np.random.randn(
        batch_size,
        input_features
    ).astype(np.float32)

else:

    print("\nWARNING:")
    print("Unexpected input rank.")
    print("Falling back to [1, 650].")

    input_data = np.random.randn(
        1,
        INPUT_FEATURES
    ).astype(np.float32)


# ============================================================
# WARMUP
# ============================================================

print()
print("=" * 80)
print("WARM-UP")
print("=" * 80)

print(
    f"\nRunning {NUM_WARMUP_RUNS} warm-up iterations..."
)


for _ in range(NUM_WARMUP_RUNS):

    session.run(
        None,
        {
            input_name: input_data
        }
    )


print("Warm-up completed.")


# ============================================================
# MEASURE OVERALL INFERENCE TIME
# ============================================================

print()
print("=" * 80)
print("INFERENCE BENCHMARK")
print("=" * 80)


start_time = time.perf_counter()


for _ in range(NUM_PROFILE_RUNS):

    session.run(
        None,
        {
            input_name: input_data
        }
    )


end_time = time.perf_counter()


total_time_sec = (
    end_time - start_time
)


average_time_ms = (
    total_time_sec /
    NUM_PROFILE_RUNS
) * 1000


throughput = 1.0 / (
    total_time_sec /
    NUM_PROFILE_RUNS
)


print(
    f"\nIterations       : "
    f"{NUM_PROFILE_RUNS}"
)

print(
    f"Total time       : "
    f"{total_time_sec:.6f} sec"
)

print(
    f"Average latency  : "
    f"{average_time_ms:.4f} ms"
)

print(
    f"Throughput       : "
    f"{throughput:.2f} inferences/sec"
)


# ============================================================
# END PROFILING
# ============================================================

print()
print("Stopping ONNX Runtime profiler...")


profile_path = session.end_profiling()


print("Profile generated:")
print(f"  {profile_path}")


# ============================================================
# LOAD PROFILE
# ============================================================

try:

    with open(profile_path, "r") as f:
        profile_data = json.load(f)

except Exception as e:

    print(f"\nERROR: Could not read profile JSON: {e}")
    sys.exit(1)


# ============================================================
# COPY PROFILE TO SIMPLE NAME
# ============================================================

try:

    with open(PROFILE_OUTPUT, "w") as f:

        json.dump(
            profile_data,
            f,
            indent=2
        )

    print("\nProfile copied to:")
    print(f"  {PROFILE_OUTPUT}")

except Exception as e:

    print(
        f"\nWARNING: "
        f"Could not copy profile JSON: {e}"
    )


# ============================================================
# EXTRACT NODE EVENTS
# ============================================================

node_events = []


for event in profile_data:

    if event.get("cat") != "Node":
        continue


    args = event.get("args", {})


    provider = (
        args.get("provider")
        or args.get("Provider")
        or args.get("execution_provider")
        or args.get("ExecutionProvider")
        or "Unknown"
    )


    op_name = (
        args.get("op_name")
        or args.get("op")
        or args.get("operator")
        or event.get("name", "Unknown")
    )


    duration_us = event.get("dur", 0)


    try:

        duration_us = float(duration_us)

    except (TypeError, ValueError):

        duration_us = 0.0


    node_events.append(
        {
            "name": str(op_name),
            "provider": str(provider),
            "duration_us": duration_us
        }
    )


# ============================================================
# PROVIDER STATISTICS
# ============================================================

provider_stats = defaultdict(
    lambda: {
        "nodes": 0,
        "time_us": 0.0
    }
)


for event in node_events:

    provider = event["provider"]

    provider_stats[provider]["nodes"] += 1

    provider_stats[provider]["time_us"] += (
        event["duration_us"]
    )


total_node_time_us = sum(
    event["duration_us"]
    for event in node_events
)


# ============================================================
# PROVIDER DISTRIBUTION TABLE
# ============================================================

print()
print("=" * 80)
print("EXECUTION PROVIDER DISTRIBUTION")
print("=" * 80)


if not provider_stats:

    print("\nNo Node events found in profile.")

else:

    print()

    print(
        f"{'Execution Provider':35s}"
        f"{'Nodes':>10s}"
        f"{'Time (ms)':>15s}"
        f"{'Time (%)':>12s}"
    )

    print("-" * 80)


    for provider, stats in sorted(
        provider_stats.items(),
        key=lambda x: x[1]["time_us"],
        reverse=True
    ):

        percentage = 0.0

        if total_node_time_us > 0:

            percentage = (
                stats["time_us"]
                / total_node_time_us
            ) * 100


        print(
            f"{provider:35s}"
            f"{stats['nodes']:10d}"
            f"{stats['time_us'] / 1000:15.4f}"
            f"{percentage:12.2f}"
        )


# ============================================================
# OPERATOR STATISTICS PER PROVIDER
# ============================================================

operator_stats = defaultdict(
    lambda: defaultdict(
        lambda: {
            "count": 0,
            "time_us": 0.0
        }
    )
)


for event in node_events:

    provider = event["provider"]
    op_name = event["name"]

    operator_stats[provider][op_name]["count"] += 1

    operator_stats[provider][op_name]["time_us"] += (
        event["duration_us"]
    )


# ============================================================
# OPERATOR TABLE FOR EACH EXECUTION PROVIDER
# ============================================================

print()
print("=" * 80)
print("OPERATOR DISTRIBUTION BY EXECUTION PROVIDER")
print("=" * 80)


for provider in sorted(operator_stats.keys()):

    print()
    print("-" * 80)

    print(f"Execution Provider: {provider}")

    print("-" * 80)


    provider_operators = operator_stats[provider]


    sorted_operators = sorted(
        provider_operators.items(),
        key=lambda x: x[1]["time_us"],
        reverse=True
    )


    print()

    print(
        f"{'Operator':35s}"
        f"{'Count':>10s}"
        f"{'Time (ms)':>15s}"
        f"{'Time (%)':>12s}"
    )

    print("-" * 80)


    provider_total_time_us = sum(
        stats["time_us"]
        for stats in provider_operators.values()
    )


    for op_name, stats in sorted_operators:

        percentage = 0.0

        if provider_total_time_us > 0:

            percentage = (
                stats["time_us"]
                / provider_total_time_us
            ) * 100


        print(
            f"{op_name:35s}"
            f"{stats['count']:10d}"
            f"{stats['time_us'] / 1000:15.4f}"
            f"{percentage:12.2f}"
        )


# ============================================================
# TOP SLOWEST OPERATORS OVERALL
# ============================================================

print()
print("=" * 80)
print("TOP SLOWEST OPERATORS OVERALL")
print("=" * 80)


overall_operator_stats = defaultdict(
    lambda: {
        "count": 0,
        "time_us": 0.0,
        "providers": defaultdict(int)
    }
)


for event in node_events:

    op_name = event["name"]
    provider = event["provider"]

    overall_operator_stats[op_name]["count"] += 1

    overall_operator_stats[op_name]["time_us"] += (
        event["duration_us"]
    )

    overall_operator_stats[op_name]["providers"][
        provider
    ] += 1


sorted_overall = sorted(
    overall_operator_stats.items(),
    key=lambda x: x[1]["time_us"],
    reverse=True
)


print()

print(
    f"{'Rank':>5s}"
    f"{'Operator':35s}"
    f"{'Count':>10s}"
    f"{'Time (ms)':>15s}"
    f"{'Provider':>20s}"
)

print("-" * 90)


for index, (op_name, stats) in enumerate(
    sorted_overall[:TOP_N_OPERATORS],
    start=1
):

    provider_text = ", ".join(
        f"{provider} ({count})"
        for provider, count
        in stats["providers"].items()
    )


    print(
        f"{index:5d}"
        f"{op_name:35s}"
        f"{stats['count']:10d}"
        f"{stats['time_us'] / 1000:15.4f}"
        f"{provider_text:>20s}"
    )


# ============================================================
# CPU FALLBACK ANALYSIS
# ============================================================

cpu_events = []


for event in node_events:

    provider = event["provider"].lower()

    if "cpu" in provider:

        cpu_events.append(event)


print()
print("=" * 80)
print("CPU FALLBACK ANALYSIS")
print("=" * 80)


if not cpu_events:

    print("\nNo CPU node execution detected.")

else:

    cpu_time_us = sum(
        event["duration_us"]
        for event in cpu_events
    )


    cpu_percentage = 0.0

    if total_node_time_us > 0:

        cpu_percentage = (
            cpu_time_us
            / total_node_time_us
        ) * 100


    print(
        f"\nCPU nodes       : "
        f"{len(cpu_events)}"
    )

    print(
        f"CPU node time   : "
        f"{cpu_time_us / 1000:.4f} ms"
    )

    print(
        f"CPU percentage  : "
        f"{cpu_percentage:.2f}%"
    )


# ============================================================
# QNN ANALYSIS
# ============================================================

qnn_events = []


for event in node_events:

    provider = event["provider"].lower()

    if "qnn" in provider:

        qnn_events.append(event)


print()
print("=" * 80)
print("QNN EXECUTION ANALYSIS")
print("=" * 80)


if not qnn_events:

    print("\nNo QNN node execution detected.")

else:

    qnn_time_us = sum(
        event["duration_us"]
        for event in qnn_events
    )


    qnn_percentage = 0.0

    if total_node_time_us > 0:

        qnn_percentage = (
            qnn_time_us
            / total_node_time_us
        ) * 100


    print(
        f"\nQNN nodes       : "
        f"{len(qnn_events)}"
    )

    print(
        f"QNN node time   : "
        f"{qnn_time_us / 1000:.4f} ms"
    )

    print(
        f"QNN percentage  : "
        f"{qnn_percentage:.2f}%"
    )


# ============================================================
# QUANTIZATION / DEQUANTIZATION ANALYSIS
# ============================================================

QUANTIZATION_OPS = {
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear"
}


quant_events = []


for event in node_events:

    op_name = event["name"]

    for quant_op in QUANTIZATION_OPS:

        if quant_op.lower() in op_name.lower():

            quant_events.append(event)

            break


print()
print("=" * 80)
print("QUANTIZATION / DEQUANTIZATION OVERHEAD")
print("=" * 80)


if not quant_events:

    print(
        "\nNo QuantizeLinear/"
        "DequantizeLinear nodes detected."
    )

else:

    quant_time_us = sum(
        event["duration_us"]
        for event in quant_events
    )


    quant_percentage = 0.0

    if total_node_time_us > 0:

        quant_percentage = (
            quant_time_us
            / total_node_time_us
        ) * 100


    print(
        f"\nQuantization nodes : "
        f"{len(quant_events)}"
    )

    print(
        f"Q/DQ execution     : "
        f"{quant_time_us / 1000:.4f} ms"
    )

    print(
        f"Q/DQ percentage    : "
        f"{quant_percentage:.2f}%"
    )


    print()

    print(
        f"{'Operator':35s}"
        f"{'Provider':30s}"
        f"{'Time (ms)':>15s}"
    )

    print("-" * 85)


    for event in sorted(
        quant_events,
        key=lambda x: x["duration_us"],
        reverse=True
    ):

        print(
            f"{event['name']:35s}"
            f"{event['provider']:30s}"
            f"{event['duration_us'] / 1000:15.4f}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)


print(f"\nModel:")
print(f"  {MODEL_PATH}")


print("\nSession backend:")

if using_qnn:

    print("  QNNExecutionProvider + CPU fallback")

else:

    print("  CPUExecutionProvider")


print("\nAverage inference latency:")
print(
    f"  {average_time_ms:.4f} ms"
)


print("\nThroughput:")
print(
    f"  {throughput:.2f} inferences/sec"
)


print("\nExecution provider summary:")


for provider, stats in sorted(
    provider_stats.items(),
    key=lambda x: x[1]["time_us"],
    reverse=True
):

    percentage = 0.0

    if total_node_time_us > 0:

        percentage = (
            stats["time_us"]
            / total_node_time_us
        ) * 100


    print(
        f"  {provider:35s}"
        f"{stats['nodes']:6d} nodes   "
        f"{percentage:7.2f}%"
    )


print("\nProfile JSON:")
print(f"  {PROFILE_OUTPUT}")


print()
print("=" * 80)
print("PROFILING COMPLETE")
print("=" * 80)
print()