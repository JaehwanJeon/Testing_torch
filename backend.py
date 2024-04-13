import torch




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

