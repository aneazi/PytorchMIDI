import torch
import pennylane as qml
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.qlstm import QLSTM
from models.quantum_music_rnn import QuantumMusicRNN
import numpy as np

def visualize_quantum_circuits():
    """
    Visualize the quantum circuits used in the QLSTM model.
    """
    
    # Create a QLSTM instance with default parameters
    qlstm = QLSTM(
        input_size=3,
        hidden_size=128,
        n_qubits=4,  # Updated to match your current config
        n_qlayers=1,
    )
    
    # Create dummy inputs for visualization
    batch_size = 2
    dummy_inputs = torch.randn(batch_size, qlstm.n_qubits)
    
    # Get the QNode for hidden state processing
    qnode = qlstm.qlayer
    
    # Draw the circuit (no weights needed for fixed circuit)
    circuit_drawer = qml.draw(qnode)
    circuit_diagram = circuit_drawer(dummy_inputs[0])
    print(circuit_diagram)



if __name__ == "__main__":
    try:
        # Visualize quantum circuits
        visualize_quantum_circuits()
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Make sure you have pennylane and torch installed.")