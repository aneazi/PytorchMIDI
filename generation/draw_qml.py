import torch
import pennylane as qml
import matplotlib.pyplot as plt
from models.qlstm import QLSTM
from models.quantum_music_rnn import QuantumMusicRNN
import numpy as np

def visualize_quantum_circuits():
    """
    Visualize the quantum circuits used in the QLSTM model.
    """
    print("Creating QLSTM model for visualization...")
    
    # Create a QLSTM instance with default parameters
    qlstm = QLSTM(
        input_size=3,
        hidden_size=128,
        n_qubits=4,
        n_qlayers=1,
        backend="default.qubit"
    )
    
    print("\nQuantum Circuit Architecture:")
    print("=" * 50)
    print(f"Number of qubits per circuit: {qlstm.n_qubits}")
    print(f"Number of quantum layers: {qlstm.n_qlayers}")
    print(f"Input size (concatenated): {qlstm.concat_size}")
    print(f"Hidden size: {qlstm.hidden_size}")
    
    # Create dummy inputs for visualization
    batch_size = 2
    dummy_inputs = torch.randn(batch_size, qlstm.n_qubits)
    dummy_weights = torch.randn(qlstm.n_qlayers, qlstm.n_qubits)
    
    print("\nDrawing Quantum Circuits:")
    print("-" * 30)
    
    # Draw each quantum circuit
    gates = ['forget', 'input', 'update', 'output']
    
    for gate_name in gates:
        print(f"\n{gate_name.upper()} Gate Circuit:")
        print("=" * 25)
        
        # Get the QNode for this gate
        qnode = getattr(qlstm, f'qlayer_{gate_name}')
        
        # Draw the circuit
        circuit_drawer = qml.draw(qnode)
        circuit_diagram = circuit_drawer(dummy_inputs[0], dummy_weights)
        print(circuit_diagram)
        
        # Also create a more detailed drawing
        detailed_drawer = qml.draw(qnode, show_all_wires=True)
        print("\nDetailed view:")
        print(detailed_drawer(dummy_inputs[0], dummy_weights))

def visualize_full_model_architecture():
    """
    Visualize the complete QuantumMusicRNN architecture.
    """
    print("\n" + "="*60)
    print("COMPLETE QUANTUM MUSIC RNN ARCHITECTURE")
    print("="*60)
    
    # Create the full model
    model = QuantumMusicRNN(
        input_size=3, 
        hidden_size=128,
        n_qubits=4,
        n_qlayers=1
    )
    
    print("\nModel Components:")
    print("-" * 20)
    print("1. Input: (batch_size, seq_len, 3) [pitch, step, duration]")
    print("2. Quantum LSTM (QLSTM):")
    print(f"   - Input size: {model.qlstm.n_inputs}")
    print(f"   - Hidden size: {model.qlstm.hidden_size}")
    print(f"   - Concatenated size: {model.qlstm.concat_size}")
    print(f"   - Qubits per circuit: {model.qlstm.n_qubits}")
    print(f"   - Quantum layers: {model.qlstm.n_qlayers}")
    print("   - 4 Quantum circuits (forget, input, update, output gates)")
    print("3. Output heads:")
    print("   - fc_pitch: Linear(128, 128) -> pitch probabilities")
    print("   - fc_step: Linear(128, 1) -> time step prediction")
    print("   - fc_duration: Linear(128, 1) -> duration prediction")
    
    print("\nData Flow:")
    print("-" * 15)
    print("Input (B, T, 3) -> For each timestep:")
    print("  1. Concatenate [hidden_t-1, input_t] -> (B, 131)")
    print("  2. clayer_in: Linear(131, 4) -> (B, 4)")
    print("  3. Move to CPU for quantum processing")
    print("  4. 4 Quantum circuits process (B, 4) -> (B, 4) each")
    print("  5. clayer_out: Linear(4, 128) -> (B, 128) for each gate")
    print("  6. Apply LSTM gate operations (sigmoid/tanh)")
    print("  7. Update cell state and hidden state")
    print("  8. Move back to original device")
    print("-> Final hidden state (B, 128) -> Output heads")

def create_circuit_comparison():
    """
    Create a side-by-side comparison of quantum vs classical processing.
    """
    print("\n" + "="*60)
    print("QUANTUM vs CLASSICAL COMPARISON")
    print("="*60)
    
    print("\nClassical LSTM Gate Computation:")
    print("-" * 35)
    print("Input: [h_t-1, x_t] -> Linear -> Activation")
    print("f_t = sigmoid(W_f * [h_t-1, x_t] + b_f)")
    print("i_t = sigmoid(W_i * [h_t-1, x_t] + b_i)")
    print("g_t = tanh(W_g * [h_t-1, x_t] + b_g)")
    print("o_t = sigmoid(W_o * [h_t-1, x_t] + b_o)")
    
    print("\nQuantum LSTM Gate Computation:")
    print("-" * 32)
    print("Input: [h_t-1, x_t] -> Linear(131, 4) -> Quantum Circuit -> Linear(4, 128) -> Activation")
    print("f_t = sigmoid(Linear(QuantumCircuit_forget([h_t-1, x_t])))")
    print("i_t = sigmoid(Linear(QuantumCircuit_input([h_t-1, x_t])))")
    print("g_t = tanh(Linear(QuantumCircuit_update([h_t-1, x_t])))")
    print("o_t = sigmoid(Linear(QuantumCircuit_output([h_t-1, x_t])))")
    
    print("\nQuantum Circuit Details:")
    print("-" * 25)
    print("1. AngleEmbedding: Encodes classical data into quantum states")
    print("2. BasicEntanglerLayers: Creates quantum entanglement")
    print("3. PauliZ measurements: Extracts classical information")

if __name__ == "__main__":
    try:
        # Visualize quantum circuits
        visualize_quantum_circuits()
        
        # Visualize full model architecture
        visualize_full_model_architecture()
        
        # Create comparison
        create_circuit_comparison()
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Make sure you have pennylane and torch installed.")