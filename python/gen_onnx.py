"""
gen_onnx.py

Convert a Keras (.keras/.h5) or TFLite (.tflite) model to ONNX format
and validate the converted model.

The script:
- Detects the input model type automatically.
- Converts Keras or TFLite models to ONNX.
- Validates the generated ONNX model.
- Inspects ONNX inputs and outputs.
- Runs sample inference using ONNX Runtime.
- Compares Keras vs ONNX outputs for Keras models.
- Compares TFLite vs ONNX outputs for TFLite models.
- Reports the generated ONNX model size.

Usage:
    python gen_onnx.py --input <input_model> --output <output_model>

Examples:
    python gen_onnx.py --input kws_model.keras --output kws_model.onnx

    python gen_onnx.py --input kws_model.tflite --output kws_model.onnx

    python gen_onnx.py \
        --input models/kws_model.keras \
        --output models/kws_model.onnx

Supported input formats:
    .keras
    .h5
    .tflite

Requirements:
    tensorflow
    tf2onnx
    onnx
    onnxruntime
    numpy
"""

import os
import argparse

import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert Keras/TFLite model to ONNX FP32 and validate the conversion."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input .keras, .h5, or .tflite model"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where the output ONNX model will be saved"
    )

    return parser.parse_args()


def main():

    # ============================================================
    # Parse command-line arguments
    # ============================================================

    args = parse_arguments()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input model not found: {input_path}"
        )

    # Create output directory if it does not exist
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Edge Impulse KWS — Keras/TFLite → ONNX FP32")
    print("=" * 70)

    print("\nInput model :", input_path)
    print("Output model:", output_path)

    # ============================================================
    # Print package versions
    # ============================================================

    print("\nPackage versions:")
    print("TensorFlow :", tf.__version__)
    print("tf2onnx    :", tf2onnx.__version__)
    print("ONNX       :", onnx.__version__)
    print("ONNXRuntime:", ort.__version__)

    # ============================================================
    # Detect model type
    # ============================================================

    extension = os.path.splitext(input_path)[1].lower()

    if extension in [".keras", ".h5"]:
        model_type = "keras"

    elif extension == ".tflite":
        model_type = "tflite"

    else:
        raise ValueError(
            f"Unsupported file type: {extension}\n"
            "Supported formats: .keras, .h5, .tflite"
        )

    print("\nDetected model type:", model_type)

    # ============================================================
    # Load and inspect Keras model
    # ============================================================

    keras_model = None

    if model_type == "keras":

        print("\nLoading Keras model...")

        keras_model = tf.keras.models.load_model(
            input_path,
            compile=False
        )

        print("Keras model loaded successfully")

        print("\nInput shape :", keras_model.input_shape)
        print("Output shape:", keras_model.output_shape)

        print("\nModel summary:")
        keras_model.summary()

    # ============================================================
    # Load and inspect TFLite model
    # ============================================================

    tflite_interpreter = None

    if model_type == "tflite":

        print("\nLoading TFLite model...")

        tflite_interpreter = tf.lite.Interpreter(
            model_path=input_path
        )

        tflite_interpreter.allocate_tensors()

        tflite_input_details = (
            tflite_interpreter.get_input_details()
        )

        tflite_output_details = (
            tflite_interpreter.get_output_details()
        )

        print("TFLite model loaded successfully")

        print("\nInputs:")

        for i, inp in enumerate(tflite_input_details):

            print(f"\nInput {i}")
            print("  Name        :", inp["name"])
            print("  Shape       :", inp["shape"])
            print("  Shape sig.  :", inp["shape_signature"])
            print("  Data type   :", inp["dtype"])
            print("  Quantization:", inp.get("quantization"))

        print("\nOutputs:")

        for i, out in enumerate(tflite_output_details):

            print(f"\nOutput {i}")
            print("  Name        :", out["name"])
            print("  Shape       :", out["shape"])
            print("  Data type   :", out["dtype"])
            print("  Quantization:", out.get("quantization"))

    # ============================================================
    # Convert Keras → ONNX
    # ============================================================

    if model_type == "keras":

        print("\n" + "=" * 70)
        print("Converting Keras → ONNX")
        print("=" * 70)

        # --------------------------------------------------------
        # Keras 3 / tf2onnx compatibility fix
        # --------------------------------------------------------

        if not hasattr(keras_model, "output_names"):

            keras_model.output_names = [
                keras_model.outputs[0].name.split(":")[0]
            ]

        print(
            "Model output names:",
            keras_model.output_names
        )

        # --------------------------------------------------------
        # Create input signature
        # --------------------------------------------------------

        input_signature = [
            tf.TensorSpec(
                shape=keras_model.input_shape,
                dtype=tf.float32,
                name="input"
            )
        ]

        # --------------------------------------------------------
        # Convert Keras → ONNX
        # --------------------------------------------------------

        onnx_model, _ = tf2onnx.convert.from_keras(
            keras_model,
            input_signature=input_signature,
            opset=15,
            output_path=output_path
        )

        print("\nKeras → ONNX conversion successful")
        print("ONNX file:", output_path)

    # ============================================================
    # Convert TFLite → ONNX
    # ============================================================

    if model_type == "tflite":

        print("\n" + "=" * 70)
        print("Converting TFLite → ONNX")
        print("=" * 70)

        onnx_model, _ = tf2onnx.convert.from_tflite(
            input_path,
            opset=15,
            output_path=output_path
        )

        print("\nTFLite → ONNX conversion successful")
        print("ONNX file:", output_path)

    # ============================================================
    # Validate ONNX model
    # ============================================================

    print("\n" + "=" * 70)
    print("Validating ONNX model")
    print("=" * 70)

    onnx_model = onnx.load(output_path)

    onnx.checker.check_model(onnx_model)

    print("ONNX model is valid")

    # ============================================================
    # Create ONNX Runtime session
    # ============================================================

    print("\nCreating ONNX Runtime session...")

    session = ort.InferenceSession(
        output_path,
        providers=["CPUExecutionProvider"]
    )

    onnx_inputs = session.get_inputs()
    onnx_outputs = session.get_outputs()

    # ============================================================
    # Inspect ONNX input/output
    # ============================================================

    print("\nInputs:")

    for i, inp in enumerate(onnx_inputs):

        print(f"\nInput {i}")
        print("  Name :", inp.name)
        print("  Shape:", inp.shape)
        print("  Type :", inp.type)

    print("\nOutputs:")

    for i, out in enumerate(onnx_outputs):

        print(f"\nOutput {i}")
        print("  Name :", out.name)
        print("  Shape:", out.shape)
        print("  Type :", out.type)

    # ============================================================
    # Create sample input
    # ============================================================

    print("\n" + "=" * 70)
    print("Creating sample input")
    print("=" * 70)

    # For the Edge Impulse KWS model this is a flattened MFCC
    # feature vector, not raw audio.
    #
    # Example:
    #
    #     [None, 637]
    #
    # produces:
    #
    #     [1, 637]

    onnx_input = onnx_inputs[0]

    input_name = onnx_input.name
    input_shape = onnx_input.shape

    print("Input type:", onnx_input.type)

    if len(input_shape) != 2:

        raise ValueError(
            f"Expected a 2D input such as [None, 637], "
            f"but received {input_shape}"
        )

    input_length = input_shape[1]

    if not isinstance(input_length, int):

        raise ValueError(
            f"Could not determine input length "
            f"from ONNX shape: {input_shape}"
        )

    # Determine NumPy dtype
    if onnx_input.type == "tensor(int8)":

        dtype = np.int8

    elif onnx_input.type == "tensor(uint8)":

        dtype = np.uint8

    elif onnx_input.type == "tensor(float16)":

        dtype = np.float16

    else:

        dtype = np.float32

    # Generate random sample input
    sample_input = np.random.randn(
        1,
        input_length
    ).astype(dtype)

    print("\nInput name :", input_name)
    print("Input shape:", input_shape)
    print("Sample input shape:", sample_input.shape)
    print("Sample input dtype:", sample_input.dtype)

    # ============================================================
    # Run ONNX model
    # ============================================================

    print("\n" + "=" * 70)
    print("Running ONNX inference")
    print("=" * 70)

    onnx_outputs_result = session.run(
        None,
        {
            input_name: sample_input
        }
    )

    print("ONNX inference successful")

    for i, output in enumerate(onnx_outputs_result):

        print(f"\nOutput {i}")
        print("Shape :", output.shape)
        print("Values:")
        print(output)

    # ============================================================
    # Compare Keras vs ONNX
    # ============================================================

    if model_type == "keras":

        print("\n" + "=" * 70)
        print("Comparing Keras vs ONNX")
        print("=" * 70)

        keras_output = keras_model(
            sample_input,
            training=False
        ).numpy()

        onnx_output = session.run(
            None,
            {
                input_name: sample_input
            }
        )[0]

        print("\nKeras output:")
        print(keras_output)

        print("\nONNX output:")
        print(onnx_output)

        absolute_difference = np.abs(
            keras_output - onnx_output
        )

        print("\nMaximum absolute difference:")
        print(np.max(absolute_difference))

        print("\nMean absolute difference:")
        print(np.mean(absolute_difference))

        # --------------------------------------------------------
        # Numerical verification
        # --------------------------------------------------------

        try:

            np.testing.assert_allclose(
                keras_output,
                onnx_output,
                rtol=1e-5,
                atol=1e-5
            )

            print(
                "\nSUCCESS: Keras and ONNX outputs "
                "match within tolerance."
            )

        except AssertionError as e:

            print(
                "\nWARNING: Keras and ONNX outputs "
                "do not match within tolerance."
            )

            print(e)

        # --------------------------------------------------------
        # Compare predicted class
        # --------------------------------------------------------

        keras_class = np.argmax(
            keras_output,
            axis=1
        )

        onnx_class = np.argmax(
            onnx_output,
            axis=1
        )

        print(
            "\nKeras predicted class:",
            keras_class[0]
        )

        print(
            "ONNX predicted class :",
            onnx_class[0]
        )

        if keras_class[0] == onnx_class[0]:

            print(
                "\nPredicted classes MATCH"
            )

        else:

            print(
                "\nPredicted classes DO NOT MATCH"
            )

    # ============================================================
    # Compare TFLite vs ONNX
    # ============================================================

    if model_type == "tflite":

        print("\n" + "=" * 70)
        print("Comparing TFLite vs ONNX")
        print("=" * 70)

        # --------------------------------------------------------
        # Same input for both models
        # --------------------------------------------------------

        common_input = sample_input.astype(
            tflite_input_details[0]["dtype"]
        )

        # --------------------------------------------------------
        # Run TFLite
        # --------------------------------------------------------

        tflite_interpreter.set_tensor(
            tflite_input_details[0]["index"],
            common_input
        )

        tflite_interpreter.invoke()

        tflite_raw_output = (
            tflite_interpreter.get_tensor(
                tflite_output_details[0]["index"]
            )
        )

        # --------------------------------------------------------
        # Run ONNX
        # --------------------------------------------------------

        onnx_raw_output = session.run(
            None,
            {
                input_name: common_input
            }
        )[0]

        print("\nTFLite output:")
        print(tflite_raw_output)

        print("\nONNX output:")
        print(onnx_raw_output)

        # --------------------------------------------------------
        # Compare outputs
        # --------------------------------------------------------

        tflite_output = (
            tflite_raw_output.astype(np.float32)
        )

        onnx_output = (
            onnx_raw_output.astype(np.float32)
        )

        abs_diff = np.abs(
            tflite_output - onnx_output
        )

        print(
            "\nMaximum absolute difference:",
            np.max(abs_diff)
        )

        print(
            "Mean absolute difference   :",
            np.mean(abs_diff)
        )

        # --------------------------------------------------------
        # Compare predicted class
        # --------------------------------------------------------

        tflite_class = np.argmax(
            tflite_output,
            axis=1
        )

        onnx_class = np.argmax(
            onnx_output,
            axis=1
        )

        print(
            "\nTFLite predicted class:",
            tflite_class[0]
        )

        print(
            "ONNX predicted class  :",
            onnx_class[0]
        )

        if np.array_equal(
            tflite_class,
            onnx_class
        ):

            print(
                "\nSUCCESS: TFLite and ONNX "
                "predicted classes MATCH"
            )

        else:

            print(
                "\nWARNING: TFLite and ONNX "
                "predicted classes DO NOT MATCH"
            )

    # ============================================================
    # Show ONNX file size
    # ============================================================

    print("\n" + "=" * 70)
    print("Output information")
    print("=" * 70)

    file_size_bytes = os.path.getsize(
        output_path
    )

    file_size_kb = file_size_bytes / 1024
    file_size_mb = file_size_kb / 1024

    print("\nONNX file :", output_path)
    print(
        "Size      :",
        round(file_size_kb, 2),
        "KB"
    )
    print(
        "Size      :",
        round(file_size_mb, 2),
        "MB"
    )

    print("\n" + "=" * 70)
    print("Conversion completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()