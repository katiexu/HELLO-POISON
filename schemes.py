import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from datasets import MNISTDataLoaders, qml_Dataloaders
from FusionModel import QNet
from FusionModel import translator, single_enta_to_design
from poison import poison

from Arguments import Arguments
import random
import os
from tqdm import tqdm

def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    print(YELLOW + "\nTest Accuracy: {}".format(metrics) + RESET)

    
def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for feed_dict in tqdm(data_loader, desc="Training", disable=True):
    # for feed_dict in data_loader:
        images = feed_dict['image'].to(args.device)
        targets = feed_dict['digit'].to(args.device)    
        optimizer.zero_grad()
        output = model(images, args.n_qubits, args.task)
        loss = criterion(output, targets)        
        loss.backward()
        optimizer.step()

def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    target_all = torch.Tensor().to(args.device)
    output_all = torch.Tensor().to(args.device)
    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)        
            output = model(images, args.n_qubits, args.task)
            instant_loss = criterion(output, targets).item()
            total_loss += instant_loss
            target_all = torch.cat((target_all, targets), dim=0)
            output_all = torch.cat((output_all, output), dim=0) 
    total_loss /= len(data_loader)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    
    return total_loss, accuracy

def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    if args.backend == 'qi':
        tqdm_disable = False
    else:
        tqdm_disable = True
    
    with torch.no_grad():
        for feed_dict in tqdm(data_loader, desc="Evaluating", disable=tqdm_disable):
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)        
            output = model(images, args.n_qubits, args.task)

    _, indices = output.topk(1, dim=1)
    masks = indices.eq(targets.view(-1, 1).expand_as(indices))
    size = targets.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    metrics = accuracy
    # metrics = output.cpu().numpy()
    return metrics

def Scheme_eval(design, task, backend='tq',nums=(1,9),poison_x=0,poison_y=0):
    result = {}  
    args = Arguments(**task)
    args.backend = 'qi'
    path = 'weights/'  
    if task['task'].startswith('QML'):
        dataloader = qml_Dataloaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task['task'],nums=nums,poison_x=poison_x,poison_y=poison_y)
   
    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).to(args.device)
    model.load_state_dict(torch.load(path+f'tmp_{nums[0], nums[1]}'), strict= False)
    result['mae'] = evaluate(model, test_loader, args)
    return result

def Scheme(design, task, weight='base', epochs=None, verbs=None, save=None, poison_x=0, poison_y=0, dataloader=None):
    model_path=f'weights/tmp_{nums[0], nums[1]}_poison_({poison_x:.1f}, {poison_y:.1f})'
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = Arguments(**task)
    if epochs == None:
        epochs = args.epochs

    model = QNet(args, design).to(args.device)
    best_model = model

    train_loader, val_loader, test_loader = dataloader
    poison_loader=poison(train_loader,poison_x,poison_y)

    if os.path.exists(model_path):
        print(f'Weights found at {model_path}, skipping training.')
        best_model.load_state_dict(torch.load(model_path))
    else:
        if weight != 'init':
            if weight != 'base':
                model.load_state_dict(weight, strict=False)
            else:
                model.load_state_dict(torch.load('init_weights/base_fashion'))

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
        train_loss_list, val_acc_list = [], []
        best_val_acc = 0
        start = time.time()
        best_model = model

        for epoch in range(epochs):
            try:
                train(model, poison_loader, optimizer, criterion, args)
            except Exception as e:
                print('No parameter gate exists')
            train_loss, train_acc = test(model, poison_loader, criterion, args)
            train_loss_list.append(train_loss)        
            val_acc = evaluate(model, val_loader, args)
            val_acc_list.append(val_acc)
            test_acc = evaluate(model, test_loader, args)
            val_acc = 0.5 *(val_acc+train_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if verbs: print(epoch, train_loss, train_acc, test_acc, 'saving model')
                best_model = copy.deepcopy(model)           
            else:
                if verbs: print(epoch, train_loss, train_acc, test_acc)
        if save:
            os.makedirs('weights', exist_ok=True)
            torch.save(best_model.state_dict(), model_path)
        end = time.time()
        print("Running time: %s seconds" % (end - start))

    train_acc = evaluate(best_model, poison_loader, args)
    test_acc = evaluate(best_model, test_loader, args)
    display(test_acc)
    return train_acc, test_acc


if __name__ == '__main__':
    # task = {
    # 'task': 'MNIST_10',
    # 'option': 'mix_reg',
    # 'n_qubits': 10,
    # 'n_layers': 4,
    # 'fold': 2
    # }

    task = {
    'task': 'MNIST_4',
    'option': 'mix_reg',
    'n_qubits': 4,
    'n_layers': 4,
    'fold': 1,
    'backend': 'tq'
    }

    # task = {
    # 'task': 'QML_Hidden_80d',
    # 'n_qubits': 20,
    # 'n_layers': 4,
    # 'fold': 5,
    # 'option': 'mix_reg',
    # 'regular': True,
    # 'num_processes': 2
    # }
    
    arch_code = [task['n_qubits'], task['n_layers']]
    args = Arguments(**task)
    n_layers = arch_code[1]
    n_qubits = int(arch_code[0] / args.fold)
    single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]

    # single = [[5, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 1, 1, 0, 0, 0, 1], [2, 0, 1, 1, 1, 1, 1, 0, 1], [3, 0, 1, 1, 1, 1, 1, 1, 1], [4, 0, 1, 1, 1, 0, 1, 1, 1]]
    # enta =  [[1, 2, 2, 3, 2], [2, 1, 3, 3, 5], [3, 2, 2, 1, 4], [4, 1, 1, 2, 2], [5, 1, 2, 4, 4]]

    
    # design = translator(single, enta, 'full', arch_code, args.fold)
    # design = op_list_to_design(op_list, arch_code)
    design = single_enta_to_design(single, enta, arch_code, args.fold)


    res=pd.DataFrame(columns=['nums','x_alpha','y_alpha','train_acc','test_acc'])
    for nums in [(0, 1), (1, 9), (3, 6)]:
        args.digits_of_interest=nums
        dataloader = MNISTDataLoaders(args, task['task'])
        for alpha in np.arange(0, 1, 0.1):
            print('-'*20+f'nums: {nums}, alpha: {alpha}'+'-'*20)
            acc_x_train, acc_x_test = Scheme(design, task, 'init', 10, verbs=False, save=True,poison_x=alpha,dataloader=dataloader)
            res.loc[len(res)] = [str(nums), alpha, 0, acc_x_train, acc_x_test]
            acc_y_train, acc_y_test = Scheme(design, task, 'init', 10, verbs=False, save=True,poison_y=alpha,dataloader=dataloader)
            res.loc[len(res)] = [str(nums), 0, alpha, acc_y_train, acc_y_test]

    # 保存结果到文件
    output_path = 'poison_results.csv'
    res.to_csv(output_path, index=False)
    print(f"\n✓ 实验结果已保存到: {output_path}")
    # result = Scheme_eval(design, task, 'tmp', noise=True)
    # display(result)

    # torch.save(best_model.state_dict(), 'weights/base_fashion')