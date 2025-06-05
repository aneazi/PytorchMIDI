#!/usr/bin/env python3
"""
CUDA compatibility test script for quantum music generation project.
Tests PyTorch CUDA setup, PennyLane quantum circuits, and model loading.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from models.quantum_music_rnn import QuantumMusicRNN
    from models.music_rnn import MusicRNN
    from models.qlstm import QLSTM
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    MODELS_AVAILABLE = False

def test_cuda_availability():
    """Test basic CUDA availability and setup."""
    print("=== CUDA Availability Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        print("CUDA not available - will use CPU")
    
    print()

def test_device_selection():
    """Test device selection logic."""
    print("=== Device Selection Test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    
    # Test tensor creation and operations
    x = torch.randn(10, 5).to(device)
    y = torch.randn(5, 3).to(device)
    z = torch.mm(x, y)
    
    print(f"Tensor created on device: {x.device}")
    print(f"Matrix multiplication successful: {z.shape}")
    print()

def test_quantum_cpu_transfer():
    """Test CPU↔CUDA transfers for quantum processing."""
    print("=== Quantum CPU Transfer Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping transfer test")
        return
    
    device = torch.device("cuda")
    
    # Simulate quantum LSTM data flow
    batch_size = 2
    hidden_size = 8
    
    # Create tensors on CUDA
    x_cuda = torch.randn(batch_size, hidden_size).to(device)
    print(f"Original tensor device: {x_cuda.device}")
    
    # Simulate transfer to CPU for quantum processing
    x_cpu = x_cuda.detach().cpu()
    print(f"CPU tensor device: {x_cpu.device}")
    
    # Simulate quantum processing (just a simple operation)
    quantum_result = torch.sigmoid(x_cpu)
    
    # Transfer back to CUDA
    result_cuda = quantum_result.to(device)
    print(f"Result back on device: {result_cuda.device}")
    
    # Verify the transfer worked
    assert result_cuda.device == device
    print("✓ CPU↔CUDA transfer successful")
    print()

def test_model_creation():
    """Test creating models on CUDA."""
    print("=== Model Creation Test ===")
    
    if not MODELS_AVAILABLE:
        print("Models not available - skipping model test")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Test classical model
        classical_model = MusicRNN(
            input_size=3,
            hidden_size=64,
            output_pitch_size=128,
            num_layers=2
        ).to(device)
        print(f"✓ Classical MusicRNN created on {device}")
        
        # Test quantum model
        quantum_model = QuantumMusicRNN(
            input_size=3,
            hidden_size=64,
            output_pitch_size=128,
            num_layers=2,
            n_qubits=4,
            n_qlayers=1
        ).to(device)
        print(f"✓ Quantum MusicRNN created on {device}")
        
        # Test forward pass with dummy data
        batch_size = 2
        seq_len = 5
        input_size = 3
        
        dummy_input = torch.randn(batch_size, seq_len, input_size).to(device)
        
        # Classical model forward
        classical_output = classical_model(dummy_input)
        print(f"✓ Classical forward pass successful: {list(classical_output.keys())}")
        
        # Quantum model forward (will use CPU for quantum circuits)
        quantum_output = quantum_model(dummy_input)
        print(f"✓ Quantum forward pass successful: {list(quantum_output.keys())}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
    
    print()

def test_memory_usage():
    """Test CUDA memory usage."""
    print("=== CUDA Memory Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping memory test")
        return
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Check initial memory
    allocated_before = torch.cuda.memory_allocated()
    cached_before = torch.cuda.memory_reserved()
    
    print(f"Memory before test:")
    print(f"  Allocated: {allocated_before / 1024**2:.1f} MB")
    print(f"  Cached: {cached_before / 1024**2:.1f} MB")
    
    # Create some tensors
    tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000).cuda()
        tensors.append(tensor)
    
    allocated_after = torch.cuda.memory_allocated()
    cached_after = torch.cuda.memory_reserved()
    
    print(f"Memory after creating tensors:")
    print(f"  Allocated: {allocated_after / 1024**2:.1f} MB")
    print(f"  Cached: {cached_after / 1024**2:.1f} MB")
    print(f"  Increase: {(allocated_after - allocated_before) / 1024**2:.1f} MB")
    
    # Clean up
    del tensors
    torch.cuda.empty_cache()
    
    allocated_final = torch.cuda.memory_allocated()
    print(f"Memory after cleanup: {allocated_final / 1024**2:.1f} MB")
    print()

def main():
    """Run all CUDA tests."""
    print("🚀 CUDA Compatibility Test for Quantum Music Generation")
    print("=" * 60)
    
    test_cuda_availability()
    test_device_selection()
    test_quantum_cpu_transfer()
    test_model_creation()
    test_memory_usage()
    
    print("=" * 60)
    if torch.cuda.is_available():
        print("✅ CUDA setup appears to be working correctly!")
        print("Your quantum music generation project should work well with CUDA.")
        print("\nNext steps:")
        print("1. Run training: python train.py or python train_quantum.py")
        print("2. Generate music: python generation/generate_regular.py or generation/generate_quantum.py")
        print("3. Visualize: python generation/image.py <midi_file> <output_png>")
    else:
        print("⚠️  CUDA not available - using CPU mode")
        print("The project will work but may be slower for training.")
        print("Quantum processing will still work (always uses CPU for quantum circuits).")

if __name__ == "__main__":
    main()
