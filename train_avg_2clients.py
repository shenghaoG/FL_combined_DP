import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy  # <-- NEW: import the Pysyft library
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
# hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
# bob = sy.VirtualWorker(hook, id="bob")  # <-- NEW: define remote worker bob
# alice = sy.VirtualWorker(hook, id="alice")  # <-- NEW: and alice


class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False
        self.clients = 2

args = Arguments()

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda:0" if use_cuda else "cpu")
# device = torch.device("cpu")
kwargs = {}

# federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))
#     .federate((bob, alice)), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
#     batch_size=args.batch_size, shuffle=True, **kwargs)


data_train = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
length = []
subset_size = int(len(data_train) / args.clients)
for i in range(args.clients):
    length.append(subset_size)
length[-1] = int(len(data_train) - (args.clients - 1) * subset_size)
data1, data2 = torch.utils.data.random_split(data_train, [int(len(data_train) / 2), len(data_train) - int(len(data_train) / 2)])
data_bob = torch.utils.data.DataLoader(data1, batch_size=args.batch_size, shuffle=True, **kwargs)
data_alice = torch.utils.data.DataLoader(data2, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# def add_DP_noise(torch_tensor):


def train(args, model_bob, model_alice, device, federated_train_loader, epoch):
    # data_bob = []
    # data_alice = []
    # for batch_idx, (data, target) in enumerate(federated_train_loader):
    #     model_pre = model.send(data.location)  # <-- NEW: send the model to the right location
    #     data = data.to(device)
    #
    #     output = model_pre(data)
    #     # data_target = [data, target]
    #     if data.location is bob:
    #         data_bob.append(data.copy())
    #     else:
    #         data_alice.append(data.copy())
    # # model.train()
    # # model_alice = model.send(alice)
    optimizer_bob = optim.SGD(model_bob.parameters(), lr=args.lr)
    optimizer_alice = optim.SGD(model_alice.parameters(), lr=args.lr)
    model_bob.train()
    model_alice.train()
    for batch_idx, ((input_bob, target_bob), (input_alice, target_alice)) in enumerate(zip(data_bob, data_alice)):
        # model_bob = model.send(bob)
        # optimizer_bob = optim.SGD(model_bob.parameters(), lr=args.lr)
        # model_bob.train()
        # data, target = i[0].to(device), i[1].to(device)

        # model_pre = model.send(bob)  # <-- NEW: send the model to the right location
        # data = data.to(device)
        input_bob, target_bob, input_alice, target_alice = input_bob.to(device), target_bob.to(device),\
                                                           input_alice.to(device), target_alice.to(device)
        optimizer_bob.zero_grad()
        optimizer_alice.zero_grad()
        # print(input_bob.size())
        output_bob = model_bob(input_bob)
        output_alice = model_alice(input_alice)
        loss_bob = F.nll_loss(output_bob, target_bob)
        loss_alice = F.nll_loss(output_alice, target_alice)
        loss_bob.backward()
        loss_alice.backward()
        for pram_bob, pram_alice in zip(model_bob.parameters(), model_alice.parameters()):
            # print(pram_bob.grad.size())
            grad_avg = (pram_bob.grad + pram_alice.grad) / 2.0
            grad_avg = grad_avg + torch.rand_like(grad_avg) * 0.0173
            pram_bob.grad = grad_avg
            pram_alice.grad = grad_avg
        # exit()
        # grad_alice = []
        # for pram in model_alice.parameters():
        #     grad_alice.append(pram.grad.clone())
        # grad_avg = (grad_bob + grad_alice) / 2.0
        # for pram, grad in zip(model_bob.parameters(), grad_avg):
        #     pram.grad = grad
        # for pram, grad in zip(model_alice.parameters(), grad_avg):
        #     pram.grad = grad
        optimizer_bob.step()
        optimizer_alice.step()
        # optimizer_bob.zero_grad()
        # output = model_bob(data)
        # loss = F.nll_loss(output, target)
        # loss.backward()
        # optimizer_bob.step()
        # model.get() # <-- NEW: get the model back
        # loss = loss.get() # <-- NEW: get the loss back
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_bob: {:.6f}\tLoss_alice: {:.6f}'.format(
            epoch, batch_idx * args.batch_size, len(data_bob) * args.batch_size,
            100. * batch_idx / len(data_bob), loss_bob.item(), loss_alice.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


model_bob = Net().to(device)
model_alice = Net()
model_alice.load_state_dict(model_bob.state_dict())
model_alice.to(device)
for epoch in range(1, args.epochs + 1):
    train(args, model_bob, model_alice, device, test_loader, epoch)
    print("Accuracy of model_bob:\n")
    test(args, model_bob, device, test_loader)
    print("Accuracy of model_alice:\n")
    test(args, model_alice, device, test_loader)
