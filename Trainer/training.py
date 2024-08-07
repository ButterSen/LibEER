import torch
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from tqdm import tqdm

from utils.metric import Metric
from utils.store import save_state

def train(model, dataset_train, dataset_val, dataset_test, device, output_dir="result/", metrics=None, metric_choose=None, optimizer=None, scheduler=None, batch_size=16, epochs=40, criterion=None, loss_func=None, loss_param=None):
    if metrics is None:
        metrics = ['acc']
    if metric_choose is None:
        metric_choose = metrics[0]
    # data sampler for train and test data
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)
    # load dataset
    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4
    )
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4
    )
    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4
    )
    model = model.to(device)
    best_metric = {s: 0. for s in metrics}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # create Metric object
        metric = Metric(metrics)
        train_bar = tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc=
        f"Train Epoch {epoch}/{epochs}: lr:{optimizer.param_groups[0]['lr']}")
        for idx, (samples, targets) in train_bar:
            # load the samples into the device
            samples = samples.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            # perform emotion recognition
            outputs = model(samples)
            # calculate the loss value
            loss = criterion(outputs, targets) +  (0 if loss_func is None else loss_func(loss_param))
            metric.update(torch.argmax(outputs, dim=1), targets, loss.item())
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        print("\033[32m train state: " + metric.value())
        metric_value = evaluate(model, data_loader_val, device, metrics, criterion, loss_func, loss_param)
        for m in metrics:
            # if metric is the best, save the model state
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch+1, metric=m)
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")['model'])
    metric_value = evaluate(model, data_loader_test, device, metrics, criterion, loss_func, loss_param)
    for m in metrics:
        print(f"best_val_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")
    return metric_value

@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, loss_func, loss_param):
    model.eval()
    # create Metric object
    metric = Metric(metrics)
    for idx, (samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader),
                                        desc=f"Evaluating : "):
        # load the samples into the device
        samples = samples.to(device)
        targets = targets.to(device)

        # perform emotion recognition
        outputs = model(samples)

        # calculate the loss value
        loss = criterion(outputs, targets) + (0 if loss_func is None else loss_func(loss_param))
        # one hot code
        # loss = criterion(outputs, targets)
        metric.update(torch.argmax(outputs, dim=1), targets, loss.item())

    print("\033[34m eval state: " + metric.value())
    return metric.values