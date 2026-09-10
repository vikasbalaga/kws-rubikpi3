import sys
import os

from onnxruntime.quantization import (
    QuantType,
    quantize,
)

from onnxruntime.quantization.execution_providers.qnn import (
    qnn_preprocess_model,
    get_qnn_qdq_config,
)

import data_reader


def main():

    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "  python quantize_model.py "
            "<input_model.onnx> "
            "<train.npz> "
            "<output_model.onnx>"
        )
        sys.exit(1)

    input_model_path = sys.argv[1]
    calibration_data_path = sys.argv[2]
    output_model_path = sys.argv[3]

    # ---------------------------------------------------------
    # Intermediate preprocessed model
    # ---------------------------------------------------------

    base_name = os.path.splitext(output_model_path)[0]

    preprocessed_model_path = (
        base_name + "_preprocessed.onnx"
    )

    print("===================================================")
    print("QNN HTP Quantization")
    print("===================================================")

    print("\nInput model:")
    print(" ", input_model_path)

    print("\nCalibration data:")
    print(" ", calibration_data_path)

    print("\nOutput model:")
    print(" ", output_model_path)

    print("\nPreprocessed model:")
    print(" ", preprocessed_model_path)

    # ---------------------------------------------------------
    # Step 1:
    # Create calibration data reader
    # ---------------------------------------------------------

    print("\n---------------------------------------------------")
    print("Step 1: Loading calibration data")
    print("---------------------------------------------------")

    calibration_reader = data_reader.DataReader(
        input_model_path,
        calibration_data_path
    )

    # ---------------------------------------------------------
    # Step 2:
    # QNN preprocessing
    # ---------------------------------------------------------

    print("\n---------------------------------------------------")
    print("Step 2: QNN preprocessing")
    print("---------------------------------------------------")

    model_changed = qnn_preprocess_model(
        input_model_path,
        preprocessed_model_path
    )
    
    print("QNN preprocessing completed.")
    print("Model changed:", model_changed)

    # ---------------------------------------------------------
    # Step 3:
    # Create QNN-specific QDQ configuration
    #
    # QNN HTP recommended configuration:
    #
    #   Activations : UINT16
    #   Weights     : UINT8
    #
    # ---------------------------------------------------------

    print("\n---------------------------------------------------")
    print("Step 3: Creating QNN QDQ configuration")
    print("---------------------------------------------------")

    if model_changed:
        model_to_quantize = preprocessed_model_path
    else:
        model_to_quantize = input_model_path
    
    print("Model used for quantization:")
    print(" ", model_to_quantize)
    
    qnn_config = get_qnn_qdq_config(
        model_to_quantize,
        calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8
    )
    print("QNN QDQ configuration created.")

    # ---------------------------------------------------------
    # Step 4:
    # Quantize using QNN configuration
    # ---------------------------------------------------------

    print("\n---------------------------------------------------")
    print("Step 4: Quantizing model")
    print("---------------------------------------------------")

    quantize(
        model_to_quantize,
        output_model_path,
        qnn_config
    )

    print("\n===================================================")
    print("Quantization completed successfully")
    print("===================================================")

    print("\nGenerated model:")
    print(" ", output_model_path)


if __name__ == "__main__":
    main()