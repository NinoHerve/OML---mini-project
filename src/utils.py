import yaml
import torch
import torchvision


# ------------------------------ retrieve functions -------------------------------
def retrieve_setup(model_name, dataset_name):

    # retrieve model & data
    model = retrieve_model(model_name)
    training_data, test_data = retrieve_dataset(dataset_name)

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

    if dataset_name == "CIFAR10":

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))            # very ugly hard coding
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        training_data = torchvision.datasets.CIFAR10(train=True, transform=train_transforms, **dataset_kwargs)        
        test_data = torchvision.datasets.CIFAR10(train=False, transform=test_transforms, **dataset_kwargs)

    else:
        raise ValueError(f"Dataset '{dataset_name}' not implemented.")
    
    return training_data, test_data


def retrieve_model(model_name):

    if model_name.lower() == "mobilenetv3small":
        model = torchvision.models.mobilenet_v3_small(weights=None)

    else:
        raise ValueError(f"Model '{model_name}' not implemented.")

    return model 


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
    
    else:
        raise ValueError(f"Optimizer '{opt_type}' not implemented.")
    
    return optimizer


def make_from_file(file, model, dataset_name):
    """
    Make torch objects from .yml parameter file.
    """
    params = retrieve_parameters(file)

    opt_type = params["training"]["optimizer"]
    opt_kwargs = params[dataset_name][opt_type]
    optimizer = make_optimizer(opt_type, model, **opt_kwargs)

    loss_type = params[dataset_name]["loss"]
    loss = make_loss(loss_type)

    n_epochs = params["training"]["n_epochs"]
    
    batch_size = params["training"]["batch_size"]

    return optimizer, loss, n_epochs, batch_size


# ----------------------------- train function -------------------------------------

