import torch
import torch.utils.data
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from tqdm import tqdm
from itertools import cycle

from utils.metric import Metric
from utils.store import save_state

def train(model, dataset_train, dataset_val, dataset_test, device, output_dir="result/", metrics=None, metric_choose=None, optimizer=None, 
          scheduler=None, batch_size=16, epochs=40, f_criterion=None, left_ld_criterion=None, right_ld_criterion=None, global_criterion=None,loss_func=None, loss_param=None):
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
    # create criterion
    best_metric = {s: 0. for s in metrics}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        metric = Metric(metrics)
        train_bar = tqdm(enumerate(zip(data_loader_train, cycle(data_loader_val) )), total=len(data_loader_train), desc=
        f"Train Epoch {epoch+1}/{epochs}: lr:{optimizer.param_groups[0]['lr']}")
        for idx, ((features, labels), (target_features, target_labels)) in train_bar:
             # load the samples into the device
            features = features.to(device)
            labels = labels.to(device)
            labels = torch.argmax(labels, dim=1)
            target_features = target_features.to(device)
            target_labels = target_labels.to(device)
            target_labels = torch.argmax(target_labels, dim=1)#turn one-hot to [batch_size]tensor
            #create domain label，0 is source domain，1 is target domain
            domain_label = torch.tensor([0]*len(features)+[1]*len(target_features)).to(device)
            optimizer.zero_grad()
            # perform emotion recognition
            label_pred, l_domain_pred, r_domain_pred, total_domain_pred = model(features, target_features)
            # calculate the loss
            c_loss = f_criterion(label_pred, labels)
            left_ld_loss = left_ld_criterion(l_domain_pred, domain_label)
            right_ld_losss = right_ld_criterion(r_domain_pred, domain_label)
            global_ld_loss = global_criterion(total_domain_pred, domain_label)
            total_loss = c_loss + left_ld_loss + right_ld_losss + global_ld_loss
            
            metric.update(torch.argmax(label_pred, dim=1), labels, f_criterion(label_pred, labels).item())
            train_bar.set_postfix_str(f"Discriminator: {(total_loss-c_loss).item():.2f}")
            total_loss.backward()
            optimizer.step()

        print("\033[32m train state: " + metric.value())
        metric_value = evaluate(model, data_loader_val, device, metrics, f_criterion, loss_func, loss_param)
        for m in metrics:
            # if metric is the best, save the model state
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch+1, metric=m)
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")['model'])
    metric_value = evaluate(model, data_loader_test, device, metrics, f_criterion, loss_func, loss_param)
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
        targets = torch.argmax(targets, dim=1)


        # perform emotion recognition
        outputs, _, _, _ = model(samples, samples)

        # calculate the loss value
        loss = criterion(outputs, targets) + (0 if loss_func is None else loss_func(loss_param))
        # one hot code
        # loss = criterion(outputs, targets)
        metric.update(torch.argmax(outputs, dim=1), targets, loss.item())

    print("\033[34m eval state: " + metric.value())
    return metric.values

