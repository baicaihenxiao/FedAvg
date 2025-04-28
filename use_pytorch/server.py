# --- server.py ---
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import json
import time
import matplotlib.pyplot as plt




# Assuming Models.py and modified clients.py are available
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup # Use modified ClientsGroup

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg-TopK")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train') # Default to CNN
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate for server update")
parser.add_argument('-cr', '--compress_ratio', type=float, default=0.01, help='Top-k compression ratio (0 to 1)') # Added
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
            print(f"Created directory: {path}")
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
            # Decide how to handle error, e.g., exit or continue if exists
            if not os.path.isdir(path): # Check again in case of race condition
                 exit()


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__
    print("Arguments:", args)

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {dev}")

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    else:
        raise ValueError(f"Unknown model name: {args['model_name']}")

    # No need for DataParallel in this simulation structure
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = torch.nn.DataParallel(net)
    net = net.to(dev)
    print(f"Model loaded: {args['model_name']}")

    loss_func = F.cross_entropy
    # Optimizer parameters dictionary to pass to clients if needed
    opti_params = {'lr': args['learning_rate']} # Client needs LR for local SGD

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


    # --- Initialize Log List ---
    accuracy_log = []
    # --- Variable to store the latest accuracy ---
    last_recorded_accuracy = 0.0


    # --- Modified Communication Loop ---
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        # Ensure client indices are valid for the actual number of clients
        possible_client_indices = list(range(actual_num_clients))
        order = np.random.permutation(possible_client_indices)
        clients_in_comm_indices = order[0:num_in_comm]
        clients_in_comm = ['client{}'.format(idx) for idx in clients_in_comm_indices]
        print(f"Selected clients: {clients_in_comm}")


        # --- Aggregation for Sparse Updates ---
        # Initialize a structure to hold summed sparse gradients ON THE CORRECT DEVICE
        summed_sparse_gradients = {name: torch.zeros(param.shape, device=dev)
                                  for name, param in net.named_parameters() if param.requires_grad}
        client_updates_received = 0

        for client_name in tqdm(clients_in_comm):
             if client_name not in myClients.clients_set:
                 print(f"Warning: Client {client_name} not found in clients_set. Skipping.")
                 continue

             # Client performs local training and returns sparse updates (shape, indices, values)
             sparse_updates = myClients.clients_set[client_name].localUpdate(
                 args['epoch'], args['batchsize'], net, loss_func, opti_params, global_parameters
             )
             client_updates_received += 1

             # Accumulate sparse updates on the server
             with torch.no_grad():
                 for name, update_info in sparse_updates.items():
                     if name not in summed_sparse_gradients:
                         # This case shouldn't happen if client computes grads for all required params
                         print(f"Warning: Received update for unexpected parameter {name} from {client_name}. Skipping.")
                         continue

                     original_shape, indices, values = update_info
                     indices = indices.to(dev) # Move indices to server device
                     values = values.to(dev)   # Move values to server device

                     # Ensure indices are within bounds
                     num_elements_in_param = summed_sparse_gradients[name].numel()
                     if indices.max() >= num_elements_in_param:
                           print(f"Warning: Index out of bounds for parameter {name} from {client_name}. Max index: {indices.max()}, Num elements: {num_elements_in_param}. Skipping update for this param.")
                           continue

                     # Add the sparse values to the correct positions in the summed gradient tensor
                     # Use index_add_ for sparse addition (safer for potentially repeated indices, though topk shouldn't repeat)
                     summed_sparse_gradients[name].flatten().index_add_(0, indices, values)


        if client_updates_received == 0:
            print("Warning: No client updates received in this round. Skipping global model update.")
            continue

        # --- Update Global Model ---
        with torch.no_grad():
            server_lr = args['learning_rate'] # Get LR for this round
            for name, param in net.named_parameters():
                 if param.requires_grad and name in summed_sparse_gradients: # Check if grad is required and update exists
                     # Apply the averaged sparse gradient update
                     averaged_sparse_gradient = summed_sparse_gradients[name] / client_updates_received # Use actual number received
                     param.data.add_(averaged_sparse_gradient, alpha=-server_lr) # Apply update: param = param - lr * avg_grad

            # Update the global_parameters dictionary to reflect the changes in net
            for key, var in net.state_dict().items():
                 global_parameters[key] = var.clone()


        # --- Evaluation ---
        if (i + 1) % args['val_freq'] == 0:
             # Set model to evaluation mode
             net.eval()
             with torch.no_grad():
                 # net.load_state_dict(global_parameters, strict=True) # Net already has latest params
                 sum_accu = 0
                 num = 0
                 for data, label in testDataLoader:
                     data, label = data.to(dev), label.to(dev)
                     preds = net(data)
                     # Convert preds to class predictions
                     preds = torch.argmax(preds, dim=1)
                     correct_preds = (preds == label).float()
                     sum_accu += correct_preds.mean() # Calculate mean accuracy for the batch
                     num += 1 # Count number of batches

                 if num > 0:
                     current_accuracy = (sum_accu / num).item()
                     # --- Update the last recorded accuracy ---
                     last_recorded_accuracy = current_accuracy
                     print('Round {} | accuracy: {:.4f}'.format(i + 1, current_accuracy))
                     accuracy_log.append({'round': i + 1, 'accuracy': current_accuracy})

                 else:
                      print("No data in test loader for evaluation.")
             # Set back to train mode potentially? Or do it at start of client update.
             # net.train() # Good practice to set back if training continues immediately


        # --- Saving Checkpoints ---
        # if (i + 1) % args['save_freq'] == 0:
        #      save_filename = '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}_cr{}'.format(
        #                                          args['model_name'], i + 1, args['epoch'], args['batchsize'],
        #                                          args['learning_rate'], actual_num_clients, args['cfraction'],
        #                                          args['compress_ratio'] # Add compression ratio to filename
        #                                          )
        #      torch.save(net, os.path.join(args['save_path'], save_filename + '.pth'))
        #      print(f"Checkpoint saved: {save_filename}.pth")

    finish_time = time.strftime('%Y-%m-%d-%H%M%S')

    # --- Construct final JSON filename using the LAST recorded accuracy ---
    final_acc_str = "{:.4f}".format(last_recorded_accuracy).replace('.', '_') # Format accuracy for filename
    save_filename = '{}_acc{}_{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}_cr{}'.format(
        finish_time, final_acc_str, args['model_name'], args['num_comm'], args['epoch'], args['batchsize'],
        args['learning_rate'], actual_num_clients, args['cfraction'],
        args['compress_ratio']  # Add compression ratio to filename
    )

    # --- Save Log to JSON File ---
    log_filepath_final = os.path.join(args['save_path'], save_filename + '.json')
    print(f"Saving final accuracy log to: {log_filepath_final}")
    try:
        with open(log_filepath_final, 'w') as f:
            json.dump(accuracy_log, f, indent=4)
        print(f"Accuracy log saved successfully.")
    except Exception as e:
        print(f"Error saving accuracy log to {log_filepath_final}: {e}")

    # save final model
    torch.save(net, os.path.join(args['save_path'], save_filename + '.pth'))
    print(f"Checkpoint saved: {save_filename}.pth")

    # plot rounds and accuracies
    rounds = [d['round'] for d in accuracy_log]
    accuracies = [d['accuracy'] for d in accuracy_log]

    # Plot
    plt.plot(rounds, accuracies, marker='o')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Rounds')
    plt.grid(True)
    plt.savefig(os.path.join(args['save_path'], save_filename + '.png'))
    plt.show()




    print("Federated training finished.")