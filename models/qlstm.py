import torch
import torch.nn as nn
import pennylane as qml


class QLSTM(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size, 
                n_qubits=4,
                n_qlayers=1,
                batch_first=True,
                return_sequences=False, 
                return_state=False,
                backend="lightning.gpu"):
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

        # Define quantum circuits for each LSTM gate
        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
        
        def _circuit_input(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]
        
        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
        
        def _circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

        # Create QNodes
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")
        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")
        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")

        # Weight shapes for quantum layers
        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"QLSTM weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        # Classical layers
        print("Creating classical layers and VQC components...")
        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

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
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # Match qubit dimension
            y_t = self.clayer_in(v_t)

            # LSTM gates using quantum circuits
            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))  # forget gate
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))   # input gate
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t)))     # update gate
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t)))  # output gate

            # Update cell and hidden states
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        return hidden_seq, (h_t, c_t)
