import string

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
from gcRNN.models import GCRNN
from gcRNN.datasets import GCDataset
from gcRNN.tools.preprocess import preprocess


learning_rate = 0.005
train_counter = 0
valid_counter = 0


def train_step(rnn, criterion, device, dataloader, dataset, optimizer, writer):

    rnn.train()

    global train_counter
    running_loss = 0.0
    batch = 1
    for gender, name in tqdm(dataloader, desc="train", unit="batch"):
        gender.to(device)
        name.to(device)

        hidden = rnn.initHidden(gender.size(0))

        optimizer.zero_grad()

        for i in range(name.size(1)):
            output, hidden = rnn(name[:, i], hidden)
        loss = criterion(output.squeeze(1), gender.squeeze(1))

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * gender.size(0)

        if train_counter % 100 == 0:
            writer.add_scalar("Loss/train", running_loss/ batch * gender.size(0), train_counter)

        batch +=1
        train_counter += 1

    return running_loss / len(dataset)


def validation_step(rnn, criterion, device, dataloader, dataset, writer):
    rnn.eval()

    global valid_counter
    running_loss = 0.0
    batch = 1
    with torch.no_grad():
        for gender, name in tqdm(dataloader, desc="val", unit="batch"):
            gender.to(device)
            name.to(device)

            hidden = rnn.initHidden(gender.size(0))

            for i in range(name.size(1)):
                output, hidden = rnn(name[:, i], hidden)
            loss = criterion(output.squeeze(1), gender.squeeze(1))

            running_loss += loss.item() * gender.size(0)

            if valid_counter % 100 == 0:
                writer.add_scalar("Loss/train", running_loss / batch * gender.size(0), train_counter)

            batch += 1
            valid_counter += 1

    return running_loss / len(dataset)


def main(args):
    (args.model / "GCRNN").mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    criterion = nn.NLLLoss()
    df = preprocess(args.data)
    dataset = GCDataset(df)

    num_train = int(len(dataset)*0.7)
    num_val = len(dataset) - num_train

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    n_letters = len(string.ascii_letters) + 1

    rnn = GCRNN(n_letters, 128, 2)

    optimizer = Adam(rnn.parameters(), lr=1e-4)

    writer = SummaryWriter()

    if args.checkpoint:

        def map_location(storage, _):
            return storage.cuda() if torch.cuda.is_available() else storage.cpu()

        chkpt = torch.load(args.checkpoint, map_location=map_location)
        rnn.load_state_dict(chkpt["state_dict"])

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch, args.num_epochs - 1))
        print("-" * 10)

        train_loss = train_step(rnn, criterion=criterion, device=device, dataloader=train_loader,
                                dataset=train_dataset, optimizer=optimizer, writer=writer)

        print("train loss: {:.4f}".format(train_loss))

        val_loss = validation_step(rnn, criterion=criterion, device=device, dataloader=val_loader,
                                   dataset=val_dataset, writer=writer)

        print("val loss: {:.4f}".format(val_loss))

        checkpoint = args.model / "GCRNN" / "checkpoint-{:05d}-of-{:05d}.pth".format(epoch + 1, args.num_epochs)
        states = {"epoch": epoch + 1, "state_dict": rnn.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(states, checkpoint)

    writer.close()
