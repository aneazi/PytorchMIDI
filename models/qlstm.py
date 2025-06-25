import torch
import torch.nn as nn
import pennylane as qml
import time
class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, n_qlayers):
        super(QLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        

        self.memory_to_qubits = nn.Linear(hidden_size, n_qubits)
        
        dev = qml.device("default.qubit", wires=self.n_qubits)
        reps = 10  # number of data‐upload repeats

        @qml.qnode(dev, interface="torch")
        def circuit(inputs):
            for _ in range(reps):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))

                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
                for i in range(n_qubits - 2):
                    qml.CNOT(wires=[i, i + 2])
                qml.CNOT(wires=[n_qubits - 2, 0])
                qml.CNOT(wires=[n_qubits - 1, 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.qlayer = circuit
        self.qubits_to_memory = nn.Linear(n_qubits, hidden_size)
        

        self.forget_gate_input = nn.Linear(input_size, hidden_size)
        self.forget_gate_hidden = nn.Linear(hidden_size, hidden_size)
        
        self.input_gate_input = nn.Linear(input_size, hidden_size)
        self.input_gate_hidden = nn.Linear(hidden_size, hidden_size)
        
        self.candidate_input = nn.Linear(input_size, hidden_size)
        self.candidate_hidden = nn.Linear(hidden_size, hidden_size)
        
        self.output_gate_input = nn.Linear(input_size, hidden_size)
        self.output_gate_hidden = nn.Linear(hidden_size, hidden_size)
        
    
    def forward(self, x_t, hidden_states):
        h_t, c_t = hidden_states        
        c_compressed = self.memory_to_qubits(c_t)
        c_quantum_raw = self.qlayer(c_compressed)
        c_quantum_raw = torch.stack(c_quantum_raw, dim=-1).type(torch.float32)
        c_quantum_expanded = self.qubits_to_memory(c_quantum_raw)
        
        f_t = torch.sigmoid(
            self.forget_gate_input(x_t) + 
            self.forget_gate_hidden(h_t)
        )
        
        i_t = torch.sigmoid(
            self.input_gate_input(x_t) + 
            self.input_gate_hidden(h_t)
        )
        
        c_candidate = torch.tanh(
            self.candidate_input(x_t) + 
            self.candidate_hidden(h_t)
        )
        
        o_t = torch.sigmoid(
            self.output_gate_input(x_t) + 
            self.output_gate_hidden(h_t)
        )
        
        # Update cell state
        c_t = (f_t * c_quantum_expanded) + (i_t * c_candidate)
        
        # Update hidden state
        h_t = o_t * torch.tanh(c_t)
        return h_t, (h_t, c_t)
