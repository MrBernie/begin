import dependency as d
import lightning_module as m
import model

transform = d.transforms.ToTensor()
train_set = d.datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = d.datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

dataset = d.MNIST(d.os.getcwd(), download=True, transform=d.transforms.ToTensor())
train_loader = d.DataLoader(dataset)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = d.torch.Generator().manual_seed(42)
train_set, valid_set = d.data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

train_loader = d.DataLoader(train_set)
valid_loader = d.DataLoader(valid_set)

# model
autoencoder = m.LitAutoEncoder(model.Encoder(), model.Decoder())
# train model
trainer = d.L.Trainer()

# # train the model
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# # validate the model
trainer.fit(model, train_loader, valid_loader)

# test the model
# trainer.test(autoencoder, dataloaders=d.DataLoader(test_set))


