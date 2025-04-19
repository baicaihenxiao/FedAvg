# --- clients.py ---
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
# Assuming getData.py is available in the parent directory or PYTHONPATH
from getData import GetDataSet

# Helper function for top-k sparsification
def sparsify_top_k(tensor, compress_ratio):
    """
    Performs top-k sparsification on a tensor.

    Args:
        tensor: The input tensor.
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
        # Still return in sparse format for consistency
        all_indices = torch.arange(numel, device=tensor.device)
        return original_tensor_shape, all_indices, tensor_flat

    # Find the top-k largest absolute values
    abs_values = tensor_flat.abs()
    _, topk_indices = torch.topk(abs_values, k)
    topk_values = tensor_flat[topk_indices]

    return original_tensor_shape, topk_indices, topk_values


class client(object):
    def __init__(self, trainDataSet, dev, compress_ratio=0.01): # Add compress_ratio
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        # --- DGC Style Accumulator ---
        self.gradient_accumulator = {} # Store residuals locally
        self.compress_ratio = compress_ratio

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti_params, global_parameters):
        """
        Performs local training, compresses gradients, and returns sparse updates.

        Args:
            localEpoch: Number of local epochs.
            localBatchSize: Batch size for local training.
            Net: The model instance (will be modified locally).
            lossFun: The loss function.
            opti_params: Dictionary containing optimizer parameters like learning rate.
            global_parameters: The current global model parameters.

        Returns:
            dict: A dictionary containing sparse updates for each parameter.
                  Format: { param_name: (original_shape, indices, values) }
        """
        Net.load_state_dict(global_parameters, strict=True) # Load global model state
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True, drop_last=True)

        # Use a standard optimizer for local steps *within* an epoch
        # Note: DGC logic applies *after* accumulating gradients over epochs
        optimizer = torch.optim.SGD(Net.parameters(), lr=opti_params['lr'])

        # Set model to training mode
        Net.train()
        epoch_gradients = {name: [] for name, param in Net.named_parameters() if param.requires_grad}

        # --- Training Loop ---
        for epoch in range(localEpoch):
            batch_grads = {name: [] for name in epoch_gradients.keys()}
            for data, label in self.train_dl:
                if data.shape[0] == 0: continue
                data, label = data.to(self.dev), label.to(self.dev)

                optimizer.zero_grad()
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()

                # Store gradients for this batch before optimizer step
                with torch.no_grad():
                    for name, param in Net.named_parameters():
                         if param.grad is not None and param.requires_grad:
                             batch_grads[name].append(param.grad.clone())

                # Optional: Step optimizer to update local model if needed by compression logic
                # If compressing the *difference* between initial and final weights,
                # you would call optimizer.step() here.
                # If compressing *gradients*, you might not step here but accumulate grads.
                # Let's stick to compressing accumulated gradients for simplicity.
                # optimizer.step() # Uncomment if compressing parameter delta

        # --- Gradient Accumulation (DGC Style) ---
        # Average gradients across batches for the epoch(s) and add to accumulator
        with torch.no_grad():
            for name, grads_list in batch_grads.items():
                if grads_list: # Check if list is not empty
                    # Calculate the average gradient across batches for this epoch/client
                    avg_gradient = torch.stack(grads_list).mean(dim=0)

                    # Add the average gradient for this round to the local accumulator
                    if name not in self.gradient_accumulator:
                        self.gradient_accumulator[name] = torch.zeros_like(avg_gradient)
                    self.gradient_accumulator[name] += avg_gradient
                # else:
                    # Handle case where a parameter might not receive any gradients if layers are frozen etc.
                    # print(f"Warning: No gradients computed for parameter {name} in local update.")


        # --- Sparsification & Prepare Communication ---
        sparse_updates_comm = {}
        with torch.no_grad():
            for name, accumulated_grad in self.gradient_accumulator.items():
                # Sparsify the accumulated gradient
                original_shape, indices, values = sparsify_top_k(accumulated_grad, self.compress_ratio)

                # Store sparse info for communication (move to CPU)
                sparse_updates_comm[name] = (original_shape, indices.cpu(), values.cpu())

                # Update accumulator with the residual (what wasn't sent)
                residual = accumulated_grad.clone()
                # Create a zero mask and fill in the values that were sent
                sent_mask_flat = torch.zeros_like(residual.flatten())
                sent_mask_flat[indices] = 1
                # Subtract the sent values (effectively zeroing them out in the residual)
                # Note: Ensure proper indexing if indices are not unique/sorted (topk gives unique indices)
                residual = residual - (residual * sent_mask_flat.view_as(residual))

                # Store the residual for the next round
                self.gradient_accumulator[name] = residual


        # Return sparse updates (shape, indices, and values)
        return sparse_updates_comm

    # local_val remains the same
    def local_val(self):
        pass


class ClientsGroup(object):
    # Add compress_ratio to __init__
    def __init__(self, dataSetName, isIID, numOfClients, dev, compress_ratio=0.01):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.compress_ratio = compress_ratio # Store ratio
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        if shard_size < 1: shard_size = 1 # Avoid shard_size 0

        num_shards = mnistDataSet.train_data_size // shard_size
        if num_shards < self.num_of_clients * 2:
             print(f"Warning: Not enough data shards ({num_shards}) for {self.num_of_clients} clients each needing 2 shards. Adjusting client count or data split.")
             # Adjust num_clients or handle differently if needed
             self.num_of_clients = num_shards // 2
             print(f"Adjusted number of clients to {self.num_of_clients}")


        shards_id = np.random.permutation(num_shards)


        client_count = 0
        for i in range(0, num_shards - 1, 2): # Ensure we have pairs of shards
            if client_count >= self.num_of_clients: break

            shards_id1_idx = i
            shards_id2_idx = i + 1

            shards_id1 = shards_id[shards_id1_idx]
            shards_id2 = shards_id[shards_id2_idx]

            data_shards1 = train_data[shards_id1 * shard_size: (shards_id1 + 1) * shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: (shards_id2 + 1) * shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: (shards_id1 + 1) * shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: (shards_id2 + 1) * shard_size]

            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1) # Convert one-hot to class index

            # Create client with compress_ratio
            someone = client(TensorDataset(torch.tensor(local_data, dtype=torch.float32), torch.tensor(local_label, dtype=torch.long)), self.dev, self.compress_ratio)
            self.clients_set['client{}'.format(client_count)] = someone
            client_count += 1

        if not self.clients_set:
             raise ValueError("No clients were created. Check data size and number of clients.")
        print(f"Successfully created {len(self.clients_set)} clients.")


# Example usage (if needed for testing clients.py directly)
# if __name__=="__main__":
#     MyClients = ClientsGroup('mnist', True, 100, 'cpu', compress_ratio=0.1)
#     if 'client10' in MyClients.clients_set:
#         print(MyClients.clients_set['client10'].train_ds[0:10])
#     else:
#         print("Client 10 not found, check client creation.")