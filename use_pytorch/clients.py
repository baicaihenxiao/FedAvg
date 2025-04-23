# --- clients.py ---
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
# Assuming getData.py is available in the parent directory or PYTHONPATH
from getData import GetDataSet
import copy # Needed for deepcopy if we store initial params safely

# Helper function for top-k sparsification (same as before)
def sparsify_top_k(tensor, compress_ratio):
    """
    Performs top-k sparsification on a tensor.

    Args:
        tensor: The input tensor (parameter delta in this case).
        compress_ratio: The ratio of elements to keep (0.0 to 1.0).

    Returns:
        A tuple containing:
        - original_tensor_shape: The original shape of the tensor.
        - topk_indices: The indices of the top-k elements in the flattened tensor.
        - topk_values: The values of the top-k elements.
    """
    original_tensor_shape = tensor.shape
    tensor_flat = tensor.flatten()
    numel = tensor_flat.numel()
    k = max(1, int(numel * compress_ratio)) # Ensure k is at least 1

    if k >= numel: # No compression needed if k covers all elements
        all_indices = torch.arange(numel, device=tensor.device)
        return original_tensor_shape, all_indices, tensor_flat

    abs_values = tensor_flat.abs()
    _, topk_indices = torch.topk(abs_values, k)
    topk_values = tensor_flat[topk_indices]

    return original_tensor_shape, topk_indices, topk_values


class client(object):
    def __init__(self, trainDataSet, dev, compress_ratio=0.01): # Add compress_ratio
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        # --- Accumulator for Parameter Delta Residuals ---
        self.delta_accumulator = {} # Store residuals locally
        self.compress_ratio = compress_ratio

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti_params, global_parameters):
        """
        Performs local training, calculates parameter delta, compresses delta,
        and returns sparse delta updates.

        Args:
            localEpoch: Number of local epochs.
            localBatchSize: Batch size for local training.
            Net: The model instance (will be modified locally).
            lossFun: The loss function.
            opti_params: Dictionary containing optimizer parameters like learning rate.
            global_parameters: The current global model parameters (used as starting point).

        Returns:
            dict: A dictionary containing sparse delta updates for each parameter.
                  Format: { param_name: (original_shape, indices, values) }
        """
        # Store initial parameters carefully to calculate delta later
        # Use deepcopy to avoid modifying the dictionary passed from the server
        initial_params_state = copy.deepcopy(global_parameters)

        Net.load_state_dict(initial_params_state, strict=True) # Load global model state
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True, drop_last=True)

        # Use a standard optimizer for local steps
        optimizer = torch.optim.SGD(Net.parameters(), lr=opti_params['lr'])

        # Set model to training mode
        Net.train()

        # --- Training Loop (Now includes optimizer.step()) ---
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                if data.shape[0] == 0: continue
                data, label = data.to(self.dev), label.to(self.dev)

                optimizer.zero_grad()
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                optimizer.step() # Update local model parameters

        # --- Calculate Parameter Delta & Accumulate Residuals ---
        parameter_delta = {}
        current_params_state = Net.state_dict()
        with torch.no_grad():
            for name, initial_param in initial_params_state.items():
                # Calculate the change (delta) for this round
                delta = current_params_state[name] - initial_param

                # Add this round's delta to the local accumulator
                if name not in self.delta_accumulator:
                    self.delta_accumulator[name] = torch.zeros_like(delta)
                self.delta_accumulator[name] += delta

        # --- Sparsification & Prepare Communication ---
        sparse_updates_comm = {}
        with torch.no_grad():
            for name, accumulated_delta in self.delta_accumulator.items():
                # Sparsify the accumulated delta
                original_shape, indices, values = sparsify_top_k(accumulated_delta, self.compress_ratio)

                # Store sparse info for communication (move to CPU)
                sparse_updates_comm[name] = (original_shape, indices.cpu(), values.cpu())

                # Update accumulator with the residual delta (what wasn't sent)
                residual_delta = accumulated_delta.clone()
                # Create a zero mask and fill in the values that were sent
                sent_mask_flat = torch.zeros_like(residual_delta.flatten())
                sent_mask_flat[indices] = 1
                # Subtract the sent values (effectively zeroing them out in the residual)
                residual_delta = residual_delta - (residual_delta * sent_mask_flat.view_as(residual_delta))

                # Store the residual delta for the next round
                self.delta_accumulator[name] = residual_delta

        # Return sparse delta updates (shape, indices, and values)
        return sparse_updates_comm

    # local_val remains the same
    def local_val(self):
        pass


class ClientsGroup(object):
    # Init and dataSetBalanceAllocation remain the same as the previous Top-K Gradient version
    # Ensure compress_ratio is passed to client constructor
    def __init__(self, dataSetName, isIID, numOfClients, dev, compress_ratio=0.01):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.compress_ratio = compress_ratio
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        # ... (Exactly same data loading/splitting as previous Top-K Gradient version)...

        # --- Client Creation ---
        client_count = 0
        # Use the same shard allocation logic as before
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label
        shard_size = max(1, mnistDataSet.train_data_size // self.num_of_clients // 2)
        num_shards = mnistDataSet.train_data_size // shard_size
        if num_shards < self.num_of_clients * 2:
             print(f"Warning: Not enough data shards ({num_shards}). Adjusting client count.")
             self.num_of_clients = num_shards // 2
             print(f"Adjusted number of clients to {self.num_of_clients}")
        if self.num_of_clients == 0:
             raise ValueError("Cannot create clients with current data/shard settings.")

        shards_id = np.random.permutation(num_shards)

        for i in range(0, (self.num_of_clients * 2) -1 , 2): # Ensure we don't exceed adjusted client count needs
            if i + 1 >= len(shards_id): break # Avoid index error if odd num_shards
            shards_id1_idx = i
            shards_id2_idx = i + 1
            shards_id1 = shards_id[shards_id1_idx]
            shards_id2 = shards_id[shards_id2_idx]

            data_shards1 = train_data[shards_id1 * shard_size: (shards_id1 + 1) * shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: (shards_id2 + 1) * shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: (shards_id1 + 1) * shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: (shards_id2 + 1) * shard_size]

            local_data = np.vstack((data_shards1, data_shards2))
            local_label = np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)

            someone = client(TensorDataset(torch.tensor(local_data, dtype=torch.float32), torch.tensor(local_label, dtype=torch.long)), self.dev, self.compress_ratio)
            self.clients_set['client{}'.format(client_count)] = someone
            client_count += 1

        if not self.clients_set:
             raise ValueError("No clients were created. Check data size and number of clients.")
        print(f"Successfully created {len(self.clients_set)} clients.")