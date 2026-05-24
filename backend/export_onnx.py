import os
import sys
import io

# Reconfigure stdout and stderr streams to use UTF-8 to prevent Windows terminal CP1252 crashes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
from backend.models.cnn import EmotionCNN

def export_to_onnx(pth_path="backend/global_model.pth", onnx_path="frontend/global_model.onnx"):
    print("============================================================")
    print("             EXPORTING PYTORCH MODEL TO ONNX")
    print("============================================================")
    
    # 1. Initialize model
    model = EmotionCNN()
    
    # 2. Try to load trained weights, fall back to initial state if file not found
    if os.path.exists(pth_path):
        print(f"Loading trained weights from: {pth_path}...")
        model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    else:
        print(f"[WARNING] Checkpoint not found at {pth_path}. Exporting base initialized model architecture...")
        
    # 3. Set model to evaluation mode
    model.eval()
    
    # 4. Define dummy input matching model specifications (BatchSize, Channel, Height, Width)
    # FER-2013 uses 48x48 grayscale (1 channel) images
    dummy_input = torch.randn(1, 1, 48, 48, requires_grad=False)
    
    # 5. Create output directory if not exists
    os.makedirs(onnx_path, exist_ok=True) if os.path.isdir(onnx_path) else os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # 6. Export the model
    print(f"Exporting model to ONNX format at: {onnx_path}...")
    torch.onnx.export(
        model,                      # Model being run
        dummy_input,                # Model input (or a tuple for multiple inputs)
        onnx_path,                  # Where to save the model
        export_params=True,         # Store the trained parameter weights inside the model file
        opset_version=12,           # The ONNX version to export the model to (11 or 12 are widely compatible)
        do_constant_folding=True,   # Whether to execute constant folding for optimization
        input_names=['input'],      # The model's input names
        output_names=['output'],    # The model's output names
        dynamic_axes={              # Variable length axes (enables flexible batch sizing)
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False                # Disable dynamo exporter to bundle weights inside the .onnx file directly
    )
    
    print("ONNX model export complete!")
    
    # 7. Rapid validation check
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[SUCCESS] ONNX model structural check passed! The graph is valid and browser-ready.")
    except ImportError:
        print("ONNX python library not installed. Skipping structural validation check.")
    except Exception as e:
        print(f"[ERROR] ONNX model validation failed: {e}")

if __name__ == "__main__":
    export_to_onnx()
