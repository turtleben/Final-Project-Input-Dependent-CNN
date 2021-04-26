import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNetCIFAR
from train import *
from dynamicConv import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()

def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    # TODO: Implement the PGD attack
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool
    original_dat = dat.clone().detach()
    if rand_start:
        dat = dat + torch.rand(dat.shape).to(device) * 2 * eps - eps
    for i in range(iters):
        dat = dat + alpha * torch.sign(gradient_wrt_data(model, device, dat, lbl))
        dat = torch.max(torch.min(dat, original_dat + eps), original_dat - eps)
    # dat = torch.clamp(dat, min=0, max=1)
    return dat

def FGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float

    return PGD_attack(model=model, device=device, dat=dat, lbl=lbl, eps=eps, alpha=1, iters=1, rand_start=False)

def MomentumIterative_attack(model, device, dat, lbl, eps, alpha, iters, mu):
    # TODO: Implement the Momentum Iterative Method
    # - dat and lbl are tensors
    # - eps, alpha and mu are floats
    # - iters is an integer
    
    original_dat = dat.clone().detach()
    momentum = torch.zeros_like(dat).detach().to(device)

    for i in range(iters):
        grad = gradient_wrt_data(model, device, dat, lbl)
        # print('grad:', grad)
        grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
        # print('grad_norm:', grad_norm)
        # print('grad_norm1:', grad_norm.view([-1]+[1]*(len(grad.shape)-1)))
        grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
        momentum = momentum * mu + grad
        assert(momentum.shape == grad.shape)
        dat = dat + alpha * torch.sign(momentum)
        dat = torch.max(torch.min(dat, original_dat + eps), original_dat - eps)
    # dat = torch.clamp(dat, min=0, max=1)
    return dat

def attack(whitebox_model, blackbox_model, adv_type):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    # To modify the model name to do white box or black box attack

    whitebox = ResNetCIFAR(num_layers=20, K=whitebox_model)
    blackbox = ResNetCIFAR(num_layers=20, K=blackbox_model)

    whitebox.load_state_dict(torch.load("pretrained_model_K{}.pt".format(whitebox_model)))
    blackbox.load_state_dict(torch.load("pretrained_model_K{}.pt".format(blackbox_model)))

    whitebox = whitebox.to(device)
    blackbox = blackbox.to(device) 
    whitebox.eval()
    blackbox.eval()

    # _, test_acc = test(whitebox)
    # print("Initial Accuracy of Whitebox Model: ", test_acc)
    # _, test_acc = test(blackbox)
    # print("Initial Accuracy of Blackbox Model: ",test_acc)

    ## Test the models against an adversarial attack

    # TODO: Set attack parameters here
    ATK_EPS = np.linspace(0,0.1,11)
    ATK_ITERS = 10
    # ATK_ALPHA = 1.85*(EPS/ITS)

    whitebox_acc_list = []
    blackbox_acc_list = []

    for i in range(ATK_EPS.size):
        eps = ATK_EPS[i]
        alpha = 1.85 * (eps / ATK_ITERS)
        print('current eps:', eps, ' alpha:', alpha)
        whitebox_correct = 0.
        blackbox_correct = 0.
        running_total = 0.
        for batch_idx, (data, labels) in enumerate(testloader):
            data = data.to(device) 
            labels = labels.to(device)

            # TODO: Perform adversarial attack here
            if adv_type == 'FGSM':
                adv_data = FGSM_attack(whitebox, device, data.clone().detach(), labels, eps=eps)
            elif adv_type == 'PGD':
                adv_data = PGD_attack(whitebox, device, data.clone().detach(), labels, eps=eps, alpha=alpha, iters=ATK_ITERS, rand_start=True)
            else:
                adv_data = MomentumIterative_attack(whitebox, device, data.clone().detach(), labels, eps=eps, alpha=alpha, iters=ATK_ITERS, mu=1)
            

            # Sanity checking if adversarial example is "legal"
            assert(torch.max(torch.abs(adv_data-data)) <= (eps + 1e-5) )
            # assert(adv_data.max() == 1.)
            # assert(adv_data.min() == 0.)
            
            # Compute accuracy on perturbed data
            with torch.no_grad():
                # Stat keeping - whitebox
                whitebox_outputs = whitebox(adv_data)
                _,whitebox_preds = whitebox_outputs.max(1)
                whitebox_correct += whitebox_preds.eq(labels).sum().item()
                # Stat keeping - blackbox
                blackbox_outputs = blackbox(adv_data)
                _,blackbox_preds = blackbox_outputs.max(1)
                blackbox_correct += blackbox_preds.eq(labels).sum().item()
                running_total += labels.size(0)
            
            # Plot some samples
            if batch_idx == 1:
                plt.figure(figsize=(15,5))
                for jj in range(12):
                    plt.subplot(2,6,jj+1);plt.imshow(adv_data[jj,0].cpu().numpy(),cmap='gray');plt.axis("off")
                plt.tight_layout()
                plt.show()
                plt.close()

        # Print final 
        whitebox_acc = whitebox_correct/running_total
        blackbox_acc = blackbox_correct/running_total

        whitebox_acc_list.append(whitebox_acc)
        blackbox_acc_list.append(blackbox_acc)
        print("Attack Epsilon: {}; Whitebox Accuracy: {}; Blackbox Accuracy: {}".format(eps, whitebox_acc, blackbox_acc))

    print('whitebox_acc:', whitebox_acc_list)
    print('blackbox_acc:', blackbox_acc_list)

    plt.plot(ATK_EPS, whitebox_acc_list)
    plt.title('whitebox attack with {} on model K_{}'.format(adv_type, whitebox_model))
    plt.savefig('whitebox attack with {} on model K_{}'.format(adv_type, whitebox_model))
    plt.close()

    plt.plot(ATK_EPS, blackbox_acc_list)
    plt.title('blackbox attack with {} on model K_{} from K_{}'.format(adv_type, blackbox_model, whitebox_model))
    plt.savefig('blackbox attack with {} on model K_{} from K_{}'.format(adv_type, blackbox_model, whitebox_model))
    plt.close()

    print("Done!")


def attack_log(whitebox_model, blackbox_model):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    class_str = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    criterion = nn.CrossEntropyLoss()

    whitebox = ResNetCIFAR(num_layers=20, K=whitebox_model, log=True)
    blackbox = ResNetCIFAR(num_layers=20, K=blackbox_model, log=True)
    whitebox_adv = ResNetCIFAR(num_layers=20, K=whitebox_model)

    whitebox.load_state_dict(torch.load("pretrained_model_K{}.pt".format(whitebox_model)))
    blackbox.load_state_dict(torch.load("pretrained_model_K{}.pt".format(blackbox_model)))
    whitebox_adv.load_state_dict(torch.load("pretrained_model_K{}.pt".format(whitebox_model)))

    whitebox = whitebox.to(device)
    blackbox = blackbox.to(device) 
    whitebox_adv = whitebox_adv.to(device)
    whitebox.eval()
    blackbox.eval()
    whitebox_adv.eval()

    ATK_EPS = np.linspace(0.02, 0.04, 2)
    ATK_ITERS = 10
    # ATK_ALPHA = 1.85*(EPS/ITS)

    whitebox_acc_list = []
    blackbox_acc_list = []

    for i in range(ATK_EPS.size):
        features = []
        total_target = []
        features_b = []
        total_target_b = []
        eps = ATK_EPS[i]
        alpha = 1.85 * (eps / ATK_ITERS)
        print('current eps:', eps, ' alpha:', alpha)
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_data = PGD_attack(whitebox_adv, device, inputs.clone().detach(), targets, eps=eps, alpha=alpha, iters=ATK_ITERS, rand_start=True)
            outputs = whitebox(adv_data)
            current_outputs = outputs.cpu().detach().numpy()
            # print(current_outputs.shape)
            # print(targets)
            features.append(current_outputs.reshape(-1))
            total_target.append(targets.cpu().detach().numpy().reshape(-1)[0])

            outputs = blackbox(adv_data)
            current_outputs = outputs.cpu().detach().numpy()
            # print(current_outputs.shape)
            # print(targets)
            features_b.append(current_outputs.reshape(-1))
            total_target_b.append(targets.cpu().detach().numpy().reshape(-1)[0])

            if batch_idx == 500:
                break
        features = np.asarray(features)
        tsne = TSNE(n_components=2).fit_transform(features)
        # scale and move the coordinates so they fit [0; 1] range
        def scale_to_01_range(x):
            value_range = (np.max(x) - np.min(x))
            starts_from_zero = x - np.min(x)
            return starts_from_zero / value_range
        tx = tsne[:, 0]
        ty = tsne[:, 1]

        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors_per_class = [i for i in range(10)]
        color = ["orange","pink","blue","brown","red","grey","yellow","green", "black", "purple"]
        for label in colors_per_class:
            indices = [i for i, l in enumerate(total_target) if l == label]
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            ax.scatter(current_tx, current_ty, c=color[label], label=class_str[label])
        ax.legend(loc='best')
        plt.title('TSNE with white box PGD attack on model {} with eps {}'.format(whitebox_model, eps))
        plt.savefig('figure/TSNE with white box PGD attack on model {} with {}_eps'.format(whitebox_model, int(eps*100)))
        plt.close()
        features = np.asarray(features)
        tsne = TSNE(n_components=2).fit_transform(features_b)

        tx = tsne[:, 0]
        ty = tsne[:, 1]

        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors_per_class = [i for i in range(10)]
        color = ["orange", "pink", "blue", "brown", "red", "grey", "yellow", "green", "black", "purple"]
        for label in colors_per_class:
            indices = [i for i, l in enumerate(total_target_b) if l == label]
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            ax.scatter(current_tx, current_ty, c=color[label], label=class_str[label])
        ax.legend(loc='best')
        plt.title('TSNE with black box attack on model {} with eps {} from model {}'.format(blackbox_model, eps, whitebox_model))
        plt.savefig('figure/TSNE with black box attack on model {} with {}_eps from model {}'.format(blackbox_model, int(eps*100), whitebox_model))
        plt.close()