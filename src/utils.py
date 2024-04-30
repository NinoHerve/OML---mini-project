import yaml
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.optim.lr_scheduler as schedulers
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Precision, Recall, F1Score



# ------------------------------ retrieve functions -------------------------------
def retrieve_setup(model_name, dataset_name):

    # retrieve model & data
    model = retrieve_model(model_name)
    training_data, test_data = retrieve_dataset(dataset_name)

    # change number of input channels
    num_channels = training_data[0][0].shape[0]
    model.features[0][0] = torch.nn.Conv2d(num_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)

    # change last layer of model
    in_features = model.classifier[-1].in_features
    out_features = len(training_data.classes)           # Does this attribute exist for any dataset?
    model.classifier[-1] = torch.nn.Linear(in_features=in_features, out_features=out_features)
    
    # dataset
    dataset = dict(train=training_data, test=test_data)

    return model, dataset


def retrieve_parameters(fname):
    with open(fname, "r") as file:
        data = yaml.safe_load(file)
    return data


def retrieve_dataset(dataset_name):

    dataset_kwargs = dict(
        root=f"data/{dataset_name}",    # directory where dataset will be loaded
        download=True,                  # downloads data if data not in given directory
    )

    # CIFAR10 dataset
    if dataset_name == "CIFAR10":
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))     # very ugly hard coding
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        training_data = torchvision.datasets.CIFAR10(train=True, transform=train_transforms, **dataset_kwargs)        
        test_data = torchvision.datasets.CIFAR10(train=False, transform=test_transforms, **dataset_kwargs)
    # FashionMNIST dataset
    elif dataset_name == "FashionMNIST":
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(28, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

        training_data = torchvision.datasets.FashionMNIST(train=True, transform=train_transforms, **dataset_kwargs)        
        test_data = torchvision.datasets.FashionMNIST(train=False, transform=train_transforms, **dataset_kwargs)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not implemented.")
    
    return training_data, test_data


def retrieve_model(model_name):
    if model_name.lower() == "mobilenetv3small":
        model = torchvision.models.mobilenet_v3_small(weights=None)
        # model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
    else:
        raise ValueError(f"Model '{model_name}' not implemented.")
    return model 

def retrieve_training_params(model, dataset_name, scheduler_name, file="parameters.yml"):
    """
    Make torch objects from .yml parameter file.
    """
    params = retrieve_parameters(file)

    # Optimizer
    opt_type = params["training"]["optimizer"]
    opt_kwargs = params[dataset_name][opt_type]
    optimizer = make_optimizer(opt_type, model, **opt_kwargs)

    # Learning rate scheduler
    lr_schduler_dict = params[dataset_name]["learning_rate_scheduler"][scheduler_name]
    lr_scheduler_kwargs = {k: eval(v) if "lambda" in k else v for k, v in lr_schduler_dict.items()}
    lr_scheduler = make_lr_scheduler(optimizer, scheduler_name, lr_scheduler_kwargs)

    # Loss function
    loss_type = params[dataset_name]["loss"]
    loss = make_loss(loss_type)

    # Training parameters
    n_epochs = params["training"]["n_epochs"]
    batch_size = params["training"]["batch_size"]

    return optimizer, lr_scheduler, loss, n_epochs, batch_size


# ----------------------------- make functions -------------------------------------

def make_loss(loss_type):
    if loss_type.lower() == "bce":
        loss = torch.nn.BCELoss()
    elif loss_type.lower() == "crossentropy":
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss '{loss_type}' not implemented.")
    return loss
    

def make_optimizer(opt_type, model, **opt_kwargs):
    if opt_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **opt_kwargs)
    elif opt_type.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), **opt_kwargs)
    else:
        raise ValueError(f"Optimizer '{opt_type}' not implemented.")
    return optimizer


def make_lr_scheduler(optimizer: torch.optim.Optimizer, lr_type: str, kwargs) -> torch.optim.lr_scheduler.LRScheduler:
    supported_schedulers = {
        "FixLR": schedulers.LambdaLR,
        "LinearLR": schedulers.LinearLR,
        "OneCycleLR": schedulers.OneCycleLR,
        "CyclicLR": schedulers.CyclicLR,
        "CyclicLR2": schedulers.CyclicLR,
        # Add more schedulers here as needed
    }

    if lr_type not in supported_schedulers:
        raise ValueError(f"Unsupported learning rate scheduler type: {lr_type}")

    scheduler_cls = supported_schedulers[lr_type]
    return scheduler_cls(optimizer, **kwargs)


# ---------------------- metrics -------------------------

def get_metrics(num_classes, device):
    accuracy = Accuracy(task="multiclass", num_classes=num_classes, average="macro").to(device)
    precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
    recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_metrics(metric_path, metrics, y_pred, y_true, tb_writer, n_iter):
    scores = {}
    for metric_name, metric in metrics.items():
        score = metric(y_pred, y_true)
        # print(f"{metric_name}: {score}")
        tb_writer.add_scalar(f"{metric_path}/{metric_name}", score, n_iter)
        scores[metric_name] = score.item()
    return scores


# ----------------------- training ----------------------

def training_loop(model, dataset, scheduler, optimizer, loss_fn, n_epochs=1, batch_size=32,
                train_strategy=("", 1), test_strategy=("", 1), scheduler_strategy="iter", 
                file_name="", device=torch.device("cpu")):

    # data loader
    train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size, shuffle=True) 
    eval_loader = torch.utils.data.DataLoader(dataset["test"], batch_size, shuffle=True) 
    
    # tensorboard
    tb_writer = SummaryWriter(f"./tensorboard/{file_name}")
    num_classes = len(dataset["train"].classes)
    metrics = get_metrics(num_classes, device)
    metric_tr_path = f"{train_loader.dataset.root.split('/')[-1]}/train"
    metric_te_path = f"{eval_loader.dataset.root.split('/')[-1]}/test"
    train_log = train_strategy[1] if train_strategy[0] == "iter" else train_strategy[1] * len(train_loader) // train_loader.batch_size
    eval_log = test_strategy[1] if test_strategy[0] == "iter" else test_strategy[1] * len(train_loader) // train_loader.batch_size

    # storage
    metrics_hist = []
    
    # Training loop
    model.to(device)
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}")
        for tr_iter, (X_tr_batch, y_tr_batch) in enumerate(train_loader):
            X_tr_batch, y_tr_batch = X_tr_batch.to(device), y_tr_batch.to(device)

            model.train()
            optimizer.zero_grad()
            output = model(X_tr_batch)
            loss = loss_fn(output, y_tr_batch)
            loss.backward()
            optimizer.step()
            
            iter = epoch * len(train_loader) + tr_iter

            tb_writer.add_scalar(f"{metric_tr_path}/lr", scheduler.get_last_lr()[0], iter)
            # optimizer.param_groups[0]["lr"]
            
            if iter % train_log == 0:
                # print("train log metrics")
                tb_writer.add_scalar(f"{metric_tr_path}/loss", loss.item(), iter)
                _, y_pred = torch.max(output, 1)
                tr_metric = compute_metrics(metric_tr_path, metrics, y_pred, y_tr_batch, tb_writer, iter)
                metrics_hist.append({**tr_metric, "loss": loss.item(), "lr": scheduler.get_last_lr()[0], "iter": iter, "source": "train"})

            
            if iter % eval_log == 0:
                # print("test log metrics")
                model.eval()
                with torch.no_grad():
                    all_outputs = []
                    all_predictions = []
                    all_targets = []
                    for te_iter, (X_te_batch, y_te_batch) in enumerate(eval_loader):
                        X_te_batch, y_te_batch = X_te_batch.to(device), y_te_batch.to(device)

                        output = model(X_te_batch)
                        _, y_pred = torch.max(output, 1)

                        all_outputs.append(output)
                        all_predictions.append(y_pred)
                        all_targets.append(y_te_batch)
                    
                    all_outputs = torch.cat(all_outputs, dim=0)
                    all_predictions = torch.cat(all_predictions, dim=0)
                    all_targets = torch.cat(all_targets, dim=0)

                    loss = loss_fn(all_outputs, all_targets)
                    tb_writer.add_scalar(f"{metric_te_path}/loss", loss.item(), iter)

                    te_metric = compute_metrics(metric_te_path, metrics, all_predictions, all_targets, tb_writer, iter)
                    metrics_hist.append({**te_metric, "loss": loss.item(), "lr": scheduler.get_last_lr()[0], "iter": iter, "source": "test"})

            #update scheduler
            if scheduler_strategy == "iter":
                scheduler.step()

        if scheduler_strategy == "epoch":
            scheduler.step()

    metrics_hist = np.array(metrics_hist)
    pd.DataFrame(metrics_hist.tolist()).to_csv(f"./metrics/{file_name}.csv")

    tb_writer.flush()
    tb_writer.close()
    return model, metrics_hist