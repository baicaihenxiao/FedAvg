# --- server.py ---
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import copy # Import copy
# Assuming Models.py and modified clients.py are available
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup # Use modified ClientsGroup

# --- Parser definition remains the same as previous Top-K Gradient version ---
# (Ensure --compress_ratio argument is present)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg-TopK-Delta")
# ... (all args including -cr) ...
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="LOCAL learning rate for clients") # Note: Server doesn't use LR directly for update
parser.add_argument('-cr', '--compress_ratio', type=float, default=0.01, help='Top-k compression ratio (0 to 1)')
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')

# --- test_mkdir function remains the same ---
def test_mkdir(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
            print(f"Created directory: {path}")
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
            if not os.path.isdir(path): exit()


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__
    print("Arguments:", args)

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {dev}")

    # --- Model loading remains the same ---
    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    else:
        raise ValueError(f"Unknown model name: {args['model_name']}")
    net = net.to(dev)
    print(f"Model loaded: {args['model_name']}")

    loss_func = F.cross_entropy
    # Optimizer parameters only needed for client's local training
    opti_params = {'lr': args['learning_rate']}

    # Initialize ClientsGroup with compress_ratio
    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev, args['compress_ratio'])
    testDataLoader = myClients.test_data_loader
    print(f"Number of clients: {len(myClients.clients_set)}")
    if not myClients.clients_set:
         print("Error: No clients were initialized. Exiting.")
         exit()

    # Adjust num_in_comm if client creation adjusted the number of clients
    actual_num_clients = len(myClients.clients_set)
    num_in_comm = int(max(actual_num_clients * args['cfraction'], 1))
    print(f"Clients per round: {num_in_comm}")

    # Get global model parameters
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # --- Modified Communication Loop ---
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        # --- Client Selection (remains same) ---
        possible_client_indices = list(range(actual_num_clients))
        order = np.random.permutation(possible_client_indices)
        clients_in_comm_indices = order[0:num_in_comm]
        clients_in_comm = ['client{}'.format(idx) for idx in clients_in_comm_indices]
        print(f"Selected clients: {clients_in_comm}")

        # --- Aggregation for Sparse Parameter Deltas ---
        # Initialize a structure to hold summed sparse deltas ON THE CORRECT DEVICE
        # Use net.state_dict() keys/shapes as reference
        summed_sparse_deltas = {name: torch.zeros(param.shape, device=dev)
                                for name, param in net.named_parameters()} # Include all params initially
        client_updates_received = 0

        # Keep a copy of parameters *before* this round for client function call
        # (although client now makes its own deepcopy)
        current_global_parameters = copy.deepcopy(global_parameters)

        for client_name in tqdm(clients_in_comm):
            if client_name not in myClients.clients_set:
                 print(f"Warning: Client {client_name} not found. Skipping.")
                 continue

            # Client performs local training and returns sparse DELTA updates
            sparse_delta_updates = myClients.clients_set[client_name].localUpdate(
                 args['epoch'], args['batchsize'], net, loss_func, opti_params, current_global_parameters
            )
            client_updates_received += 1

            # Accumulate sparse delta updates on the server
            with torch.no_grad():
                for name, update_info in sparse_delta_updates.items():
                    if name not in summed_sparse_deltas:
                        # This might happen if client has parameters server doesn't (unlikely here)
                        print(f"Warning: Received delta for unexpected parameter {name} from {client_name}. Skipping.")
                        continue

                    original_shape, indices, values = update_info
                    indices = indices.to(dev)
                    values = values.to(dev)

                    num_elements_in_param = summed_sparse_deltas[name].numel()
                    if indices.max() >= num_elements_in_param:
                           print(f"Warning: Index out of bounds for parameter delta {name} from {client_name}. Skipping update.")
                           continue

                    # Add the sparse delta values to the correct positions
                    summed_sparse_deltas[name].flatten().index_add_(0, indices, values)


        if client_updates_received == 0:
            print("Warning: No client updates received in this round. Skipping global model update.")
            continue

        # --- Update Global Model using Averaged Delta ---
        with torch.no_grad():
            for name, param in net.named_parameters():
                 if name in summed_sparse_deltas:
                     # Average the summed sparse deltas
                     averaged_sparse_delta = summed_sparse_deltas[name] / client_updates_received
                     # Apply the averaged delta directly to the global model parameters
                     # param = param + avg_delta
                     param.data.add_(averaged_sparse_delta) # alpha=1 (default)

            # Update the global_parameters dictionary to reflect the changes in net
            for key, var in net.state_dict().items():
                 global_parameters[key] = var.clone()


        # --- Evaluation (remains the same as previous Top-K version) ---
        if (i + 1) % args['val_freq'] == 0:
             net.eval() # Set to eval mode
             with torch.no_grad():
                 sum_accu = 0
                 num = 0
                 for data, label in testDataLoader:
                     data, label = data.to(dev), label.to(dev)
                     preds = net(data)
                     preds = torch.argmax(preds, dim=1)
                     sum_accu += (preds == label).float().mean()
                     num += 1
                 if num > 0:
                     print('accuracy: {:.4f}'.format(sum_accu / num))
                 else:
                      print("No data in test loader.")
             # net.train() # Optional: Set back to train mode

        # --- Saving Checkpoints (remains the same, filename reflects compression) ---
        if (i + 1) % args['save_freq'] == 0:
             save_filename = '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}_cr{}_delta'.format(
                                                 args['model_name'], i + 1, args['epoch'], args['batchsize'],
                                                 args['learning_rate'], actual_num_clients, args['cfraction'],
                                                 args['compress_ratio'] # Add compression ratio
                                                 ) # Indicate delta compression
             torch.save(net, os.path.join(args['save_path'], save_filename + '.pth'))
             print(f"Checkpoint saved: {save_filename}.pth")

    print("Federated training (Top-K Delta) finished.")