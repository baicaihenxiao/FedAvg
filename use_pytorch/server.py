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
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=100, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    # 2nn 是 MLP，Treats input data as a flat vector，his process loses spatial information – the network doesn't inherently know which input features were originally neighbors in the image,
    # CNN Explicitly designed to process grid-like data (like images) by preserving spatial relationships.
    # todo: ？但是 use_pytorch.getData.GetDataSet.mnistDataSetConstruct 里没有根据哪种模型来判断啊，全都转成 flat input 了？
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # --- Initialize Log List ---
    accuracy_log = []
    # --- Variable to store the latest accuracy ---
    last_recorded_accuracy = 0.0

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1

                if num > 0:
                    current_accuracy = (sum_accu / num).item()
                    # --- Update the last recorded accuracy ---
                    last_recorded_accuracy = current_accuracy
                    print('Round {} | accuracy: {:.4f}'.format(i + 1, current_accuracy))
                    accuracy_log.append({'round': i + 1, 'accuracy': current_accuracy})

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))


    finish_time = time.strftime('%Y-%m-%d-%H%M%S')

    # --- Construct final JSON filename using the LAST recorded accuracy ---
    final_acc_str = "{:.4f}".format(last_recorded_accuracy).replace('.', '_') # Format accuracy for filename
    save_filename = 'master_{}_acc{}_{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(
        finish_time, final_acc_str, args['model_name'], args['num_comm'], args['epoch'], args['batchsize'],
        args['learning_rate'], args['num_of_clients'], args['cfraction'],
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

