import torch
import torch.nn as nn



def add_diff(X):
    X_diff = torch.diff(X, axis=1, prepend=torch.zeros(X.size(0), 1, X.size(2)).to(X.device))
    X = torch.cat([X, X_diff], axis=2)
    return X


def augmentation(sample, rate):
    raise NotImplementedError
    # sample: [batch, time, features]

    shifted_right = torch.roll(sample, shifts=1, dims=1)
    shifted_left = torch.roll(sample, shifts=-1, dims=1)
    local_max_mask = (sample > shifted_right) & (sample > shifted_left)

    local_min_mask = (sample < shifted_right) & (sample < shifted_left)

    extrema_mask = local_max_mask | local_min_mask

    non_extrema_mask = ~extrema_mask

    indices = torch.randperm(sample.size(1))

    num_augmented = int(sample.size(1) * rate)

    non_extrema_mask = non_extrema_mask[:, indices[:num_augmented], :]


class Loss:
    def __init__(self, y_pred, y, energies, mask, device):
        self.mse_loss_init = (torch.mean((y_pred - y)**2 * mask)).to(device).detach()
        self.phys_loss_init = (self.drucker_loss(y_pred, energies, mask)).to(device).detach()
        print('MSE Loss / Phys Loss will be normalized by {}/{}'.format(self.mse_loss_init, self.phys_loss_init))
        pass

    def combined_loss(self, y_pred, y, energies, mask, alpha=0.2):
        self.mse_loss = torch.mean((y_pred - y)**2 * mask) / self.mse_loss_init
        self.phys_loss = self.drucker_loss(y_pred, energies, mask) / self.phys_loss_init  # Assuming y_pred and y are 2D tensors [batch_size x seq_length]
        print('MSE Loss / Phys Loss: {:.8f}/{:.8f}'.format(self.mse_loss, self.phys_loss), end='\r')
        return (1-alpha) * self.mse_loss + alpha * self.phys_loss
    
    def drucker_loss(self, f, e, mask):
    
        num_samples, time_length = f.size()
        num_chops = 201
        chop_vector = torch.linspace(-1, 1, num_chops).to(f.device)

        # Expand the mask and tensors along the second axis
        mask_expanded = mask.unsqueeze(1).expand(-1, chop_vector.size(0), -1)
        f_expanded = f.unsqueeze(1).expand(-1, chop_vector.size(0), -1)
        e_expanded = e.unsqueeze(1).expand(-1, chop_vector.size(0), -1)

        chop_vector_expanded = chop_vector.unsqueeze(0).unsqueeze(-1).expand(num_samples, -1, time_length)
        deducted_f_expanded = f_expanded - chop_vector_expanded

        # Create deducted_f_sign tensor with size [num_samples, num_chops, time_length]
        deducted_f_sign = (deducted_f_expanded > 0).int()

        diff_sign = torch.diff(deducted_f_sign, dim=2, prepend=deducted_f_sign[:, :, 0].unsqueeze(2))
        change_bool = ((diff_sign != 0) * mask_expanded).bool()
        change_idx = torch.nonzero(change_bool)
        selected_e = e_expanded[change_idx[:, 0], change_idx[:, 1], change_idx[:, 2]]
        diff_e = selected_e[1:] - selected_e[:-1]
        diff_idx = change_idx[1:] - change_idx[:-1]
        diff_e = diff_e * (diff_idx[:, 0] == 0) * (diff_idx[:, 1] == 0)
        diff_e_neg = diff_e[diff_e < 0]
        return -torch.sum(diff_e_neg) / (num_samples * time_length)


class CustomLSTMCell2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell2, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear transformation for energy
        self.energy_transform = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)  ##### 이게 맞는지 모르겠다
        
        # Input gate
        self.fc_i = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)  # *2 for energy
        # Forget gate
        self.fc_f = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)
        # Cell state
        self.fc_c = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)
        # Output gate
        self.fc_o = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)

        self.ln_h = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, h_energy, states):
        h, c = states
        h_normalized = self.ln_h(h)
        # Transform the energy
        transformed_energy = torch.tanh(self.energy_transform(h_energy))
        
        h_combined = torch.cat([x, transformed_energy, h_normalized], 1)  # concatenate along the feature dimension

        i = torch.sigmoid((self.fc_i(h_combined)))
        f = torch.sigmoid((self.fc_f(h_combined)))
        g = torch.tanh((self.fc_c(h_combined)))
        o = torch.sigmoid((self.fc_o(h_combined)))
        
        c_next = f * c + i * g
        h_next = self.ln_h(o * torch.tanh(c_next) + h)    # Adding residual connection

        return h_next, c_next


class CustomLSTM2(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, output_dim):
        super(CustomLSTM2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        
        self.cell = CustomLSTMCell(input_dim, hidden_dim)
        self.cell2 = CustomLSTMCell(hidden_dim, hidden_dim2)
        self.fc = nn.Linear(hidden_dim2 + 1, output_dim, bias=False)
        

    def forward(self, x, states=None):
        batch_size, seq_length, _ = x.size()

        if states is None:
            h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            h2 = torch.zeros(batch_size, self.hidden_dim2).to(x.device)
            c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c2 = torch.zeros(batch_size, self.hidden_dim2).to(x.device)
            prev_output = torch.zeros(batch_size, 1).to(x.device)
            h_energy = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            h_energy2 = torch.zeros(batch_size, self.hidden_dim2).to(x.device)
            energy = torch.zeros(batch_size, 1).to(x.device)
            previous_x = torch.zeros(batch_size, 1).to(x.device)  # Initialize previous_x with zeros
        else:
            h, h2, c, c2, prev_output, h_energy, h_energy2, energy, previous_x = states

        outputs = []
        energies = []
        for t in range(seq_length):
            prev_h = h
            prev_h2 = h2
            h, c = self.cell(x[:, t, :], h_energy, (h, c))
            h2, c2 = self.cell2(h, h_energy2, (h2, c2))
            output = self.fc(torch.cat([h2, x[:, t, 0].unsqueeze(1)], dim=1)) # Need to check if this is correct
            
            # Calculate and accumulate energy using the trapezoid rule
            current_x = x[:, t, 0].unsqueeze(1)
            delta_disp = current_x - previous_x
            h_energy = h_energy + (h + prev_h) / 2 * delta_disp
            h_energy2 = h_energy2 + (h2 + prev_h2) / 2 * delta_disp
            energy = energy + (output + prev_output) / 2 * delta_disp
            energies.append(energy)
            prev_output = output
            previous_x = current_x
            outputs.append(output)

        return torch.stack(outputs, dim=1), torch.stack(energies, dim=1), (h, h2, c, c2, prev_output, h_energy, h_energy2, energy, previous_x)


class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear transformation for energy
        self.energy_transform = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # Input gate
        self.fc_i = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)  # *2 for energy
        # Forget gate
        self.fc_f = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)
        # Cell state
        self.fc_c = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)
        # Output gate
        self.fc_o = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)

        self.ln_h = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, h_energy, states):
        h, c = states
        h_normalized = self.ln_h(h)
        # Transform the energy
        transformed_energy = torch.tanh(self.energy_transform(h_energy))
        
        h_combined = torch.cat([x, transformed_energy, h_normalized], 1)  # concatenate along the feature dimension

        i = torch.sigmoid((self.fc_i(h_combined)))
        f = torch.sigmoid((self.fc_f(h_combined)))
        g = torch.tanh((self.fc_c(h_combined)))
        o = torch.sigmoid((self.fc_o(h_combined)))
        
        c_next = f * c + i * g
        h_next = self.ln_h(o * torch.tanh(c_next) + h)    # Adding residual connection

        return h_next, c_next


class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.cell = CustomLSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + 1, output_dim, bias=False)
        

    def forward(self, x, states=None):
        batch_size, seq_length, _ = x.size()

        if states is None:
            h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            prev_output = torch.zeros(batch_size, 1).to(x.device)
            h_energy = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            energy = torch.zeros(batch_size, 1).to(x.device)
            previous_x = torch.zeros(batch_size, 1).to(x.device)  # Initialize previous_x with zeros
        else:
            h, c, prev_output, h_energy, energy, previous_x = states

        outputs = []
        energies = []
        for t in range(seq_length):
            prev_h = h
            h, c = self.cell(x[:, t, :], h_energy, (h, c))
            output = self.fc(torch.cat([h, x[:, t, 0].unsqueeze(1)], dim=1)) # Need to check if this is correct
            
            # Calculate and accumulate energy using the trapezoid rule
            current_x = x[:, t, 0].unsqueeze(1)
            delta_disp = current_x - previous_x
            h_energy = h_energy + (h + prev_h) / 2 * delta_disp
            energy = energy + (output + prev_output) / 2 * delta_disp
            energies.append(energy)
            prev_output = output
            previous_x = current_x
            outputs.append(output)

        return torch.stack(outputs, dim=1), torch.stack(energies, dim=1), (h, c, prev_output, h_energy, energy, previous_x)