import torch
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

import math

from utils.metric import Metric
from utils.store import save_state

def train(model, datasets_train, dataset_val, dataset_test, samples_source, device, output_dir=None, metrics=['acc'], metric_choose=None, optimizer=None, scheduler=None, batch_size=16, epochs=40, criterion=None, loss_func=None, loss_param=None):
    if metrics is None:
        metrics = ['acc']
    if metric_choose is None:
        metric_choose = metrics[0]
    source_loaders = []
    for j, dataset_train in enumerate(datasets_train):
        sampler_train = RandomSampler(dataset_train)
        source_loaders.append(DataLoader(dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4, drop_last=True))
    # data sampler for train and test data
    sampler_test = SequentialSampler(dataset_test)
    sampler_val = SequentialSampler(dataset_val)
    # load dataset
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4, drop_last=True
    )
    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4, drop_last=True
    )
    model = model.to(device)
    best_metric = {s: 0. for s in metrics}
    iteration = math.ceil(samples_source/batch_size)
    iterations = epochs * iteration
    log_interval = 10
    target_iter = iter(data_loader_val)
    source_iters = []
    for i in range(len(source_loaders)):
        source_iters.append(iter(source_loaders[i]))
    for epoch in range(epochs):
        _tqdm = tqdm(range(iteration), desc= f"Train Epoch {epoch  + 1}/{epochs}",leave=False)
        #, colour='red' position=2,, desc= f"Train Epoch {epoch + args.resume_epoch + 1}/{args.epochs}",
        for idx in _tqdm:
            model.train()
            # the optimizer for train
            # optimizer = torch.optim.Adam(
            #         model.parameters(), lr=LEARNING_RATE)
            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception as err:
                    source_iters[j] = iter(source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                try:
                    target_data, _ = next(target_iter)
                except Exception as err:
                    target_iter = iter(data_loader_val)
                    target_data, _ = next(target_iter)
                source_data, source_label = source_data.to(device), source_label.to(device)
                target_data = target_data.to(device)

                optimizer.zero_grad()
                # print(source_data.shape, target_data.shape, source_label.shape, len(source_loaders), j)
                cls_loss, mmd_loss, l1_loss = model(source_data, number_of_source=len(source_loaders),
                                                         data_tgt=target_data, label_src=source_label, mark=j)
                gamma = 2 / (1 + math.exp(-10 * (epoch*iteration+idx) / (iterations))) - 1
                beta = gamma/100
                loss = cls_loss + gamma * mmd_loss + beta * l1_loss
                loss.backward()
                optimizer.step()
                _tqdm.set_postfix_str(f"loss: {loss.item():.2f}")
            metric_value = evaluate(model, data_loader_val, device, metrics, nn.NLLLoss(), source_num=len(source_loaders))
            for m in metrics:
            # if metric is the best, save the model state
                if metric_value[m] > best_metric[m]:
                    best_metric[m] = metric_value[m]
                    save_state(output_dir, model, optimizer, epoch + 1, metric=m)
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")['model'])
    metric_value = evaluate(model, data_loader_test, device, metrics, criterion, source_num=len(source_loaders))
    # print best metrics
    for m in metrics:
        print(f"best_val_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")
    return metric_value

@torch.no_grad()
def evaluate(model, data_loader_test, device, metrics, criterion, source_num, loss_func=None, loss_param=None):
    model.eval()
    # create Metric object
    metric = Metric(metrics)
    for idx, (data, target) in tqdm(enumerate(data_loader_test), total=len(data_loader_test),
                                    desc=f"Evaluating : ", leave=False):
        # ,, position=1
        data = data.to(device)
        target = target.to(device)
        preds = model(data, source_num)

        for i in range(len(preds)):
            preds[i] = F.softmax(preds[i], dim=1)

        pred = sum(preds) / len(preds)  # 经过len(preds)个源域后预测的平均值
        test_loss = criterion(F.log_softmax(pred,
                                             dim=1), target.long().squeeze())

        metric.update(torch.argmax(pred, dim=1), target.data.squeeze(), test_loss.item())
    print("\033[34m eval state: " + metric.value())
    return metric.values