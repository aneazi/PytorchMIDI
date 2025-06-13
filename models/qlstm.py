import torch
import torch.nn as nn
import pennylane as qml
import time
import math
class QLSTM(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size, 
                n_qubits=4,
                n_qlayers=1,
                batch_first=True,
                return_sequences=False, 
                return_state=False,
                backend="default.qubit"):
        super(QLSTM, self).__init__()
        print(f"Initializing QLSTM: input_size={input_size}, hidden_size={hidden_size}, n_qubits={n_qubits}, backend={backend}")
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        # New architecture: quantum processing of hidden state, then concatenate with input
        self.quantum_concat_size = self.hidden_size + self.n_inputs  # quantum processed hidden + input
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Create quantum device and circuit for hidden state processing
        self.wires = [f"wire_{i}" for i in range(self.n_qubits)]
        self.dev = qml.device(self.backend, wires=self.wires)

        # Create QNode for quantum processing of hidden state (no trainable parameters)
        self.qlayer = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

        print("Creating classical layers and fixed quantum processing components...")
        # Compress hidden state to qubits
        self.hidden_to_qubits = torch.nn.Linear(self.hidden_size, n_qubits)
        # Fixed quantum processing layer (no trainable parameters)
        # Expand quantum output back to hidden size
        self.qubits_to_hidden = torch.nn.Linear(self.n_qubits, self.hidden_size)
        
        # Classical LSTM gates (process concatenated quantum-processed hidden + input)
        self.W_f = torch.nn.Linear(self.quantum_concat_size, self.hidden_size)  # forget gate
        self.W_i = torch.nn.Linear(self.quantum_concat_size, self.hidden_size)  # input gate
        self.W_g = torch.nn.Linear(self.quantum_concat_size, self.hidden_size)  # candidate gate
        self.W_o = torch.nn.Linear(self.quantum_concat_size, self.hidden_size)  # output gate

    # Define quantum circuit for hidden state processing (fixed transformation)
    def _quantum_circuit(self, inputs):
        # Apply angle embedding to encode classical data
        qml.AngleEmbedding(inputs, wires=self.wires, rotation="Y")

        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
        qml.CNOT(wires=[self.wires[-1], self.wires[0]])  # Connect last to first
        
        for i in range(self.n_qubits - 2):
            qml.CNOT(wires=[self.wires[i], self.wires[i + 2]])
        qml.CNOT(wires=[self.wires[-2], self.wires[0]])  # Connect second last to first
        qml.CNOT(wires=[self.wires[-1], self.wires[1]])  # Connect last to second
        
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires]

    def forward(self, x, init_states=None):
        """
        x.shape is (batch_size, seq_length, feature_size)
        """
        if self.batch_first:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]
        for t in range(seq_length):
            # Get features from the t-th element in seq
            x_t = x[:, t, :]
            # Step 1: Compress hidden state to qubit representation
            h_compressed = self.hidden_to_qubits(h_t)  # (batch_size, n_qubits)
            
            # Step 2: Process through quantum circuit
            h_quantum_raw = self.qlayer(h_compressed)  # (batch_size, n_qubits)
            
            # Step 3: Convert to tensor if it's a list/tuple
            if isinstance(h_quantum_raw, (list, tuple)):
                h_quantum = torch.stack(h_quantum_raw, dim=-1)  # Stack along last dimension
            else:
                h_quantum = h_quantum_raw
            
            h_quantum=h_quantum.float()  # Ensure it's a float tensor
            
            # Step 4: Expand quantum output back to hidden size
            h_quantum_expanded = self.qubits_to_hidden(h_quantum)  # (batch_size, hidden_size)
            
            # Step 5: Concatenate quantum-processed hidden with current input
            combined_input = torch.cat((h_quantum_expanded, x_t), dim=1)  # (batch_size, hidden_size + input_size)
            # Step 6: Classical LSTM gates
            f_t = torch.sigmoid(self.W_f(combined_input))  # forget gate
            i_t = torch.sigmoid(self.W_i(combined_input))  # input gate  
            g_t = torch.tanh(self.W_g(combined_input))     # candidate gate
            o_t = torch.sigmoid(self.W_o(combined_input))  # output gate
            
            # Step 7: Update cell and hidden states (standard LSTM)
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        return hidden_seq, (h_t, c_t)
