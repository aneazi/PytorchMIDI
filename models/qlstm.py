import torch
import torch.nn as nn
import pennylane as qml
import time

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
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Create separate devices for each gate
        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        # Create QNodes with pre-defined circuits (defined once)
        self.qlayer_forget = qml.QNode(self._circuit_forget, self.dev_forget, interface="torch")
        self.qlayer_input = qml.QNode(self._circuit_input, self.dev_input, interface="torch")
        self.qlayer_update = qml.QNode(self._circuit_update, self.dev_update, interface="torch")
        self.qlayer_output = qml.QNode(self._circuit_output, self.dev_output, interface="torch")

        # Weight shapes for quantum layers
        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"QLSTM weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        # Classical layers
        print("Creating classical layers and VQC components...")
        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = nn.ModuleDict({
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'candidate': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        })
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    # Define circuits as class methods (created once, not per forward pass)
    def _circuit_forget(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
    
    def _circuit_input(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]
    
    def _circuit_update(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
    
    def _circuit_output(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

    def forward(self, x, init_states=None):
        """
        x.shape is (batch_size, seq_length, feature_size)
        """
        time_total_start = time.time()
        
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
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # Match qubit dimension
            y_t = self.clayer_in(v_t)
            
            # Time quantum circuits
            time_quantum_start = time.time()
            f_out = self.VQC['forget'](y_t)
            i_out = self.VQC['input'](y_t)
            g_out = self.VQC['candidate'](y_t)
            o_out = self.VQC['output'](y_t)
            time_quantum_total += time.time() - time_quantum_start
            
            # Time classical operations
            time_classical_start = time.time()
            f_t = torch.sigmoid(self.clayer_out(f_out))  # forget gate
            i_t = torch.sigmoid(self.clayer_out(i_out))   # input gate
            g_t = torch.tanh(self.clayer_out(g_out))     # update gate
            o_t = torch.sigmoid(self.clayer_out(o_out))  # output gate
            
            # Update cell and hidden states
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
            time_classical_total += time.time() - time_classical_start
        
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        time_total_end = time.time()
        # Uncomment for performance debugging
        # print(f"Total forward pass: {time_total_end - time_total_start:.4f}s")
        # print(f"  - Quantum circuits: {time_quantum_total:.4f}s ({time_quantum_total/(time_total_end - time_total_start)*100:.1f}%)")
        # print(f"  - Classical ops: {time_classical_total:.4f}s ({time_classical_total/(time_total_end - time_total_start)*100:.1f}%)")
        
        return hidden_seq, (h_t, c_t)