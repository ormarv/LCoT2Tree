import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from gatv2 import build_dataloader, GAT

def setup(rank, world_size)->None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup()->None:
    dist.destroy_process_group()


def prepare(rank, world_size, dataset):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    return sampler

def _load_and_train_parallel(rank, world_size, train_features, train_graphs, train_labels, eval_features, eval_graphs, eval_labels, batch_size, in_channels:int, out_channels:int, hidden:int, parent_dir:str, epochs:int, lr:float):
    setup(rank=rank, world_size=world_size)

    train_loader = build_dataloader(train_features, train_graphs, train_labels, batch_size, parallel=True, rank=rank, world_size=world_size)
    eval_loader = build_dataloader(eval_features, eval_graphs, eval_labels, batch_size, parallel=True, rank=rank, world_size=world_size)
    model = GAT(in_channels=in_channels, out_channels=out_channels, hidden=hidden).to(rank)
    ddp_model = DDP(module=model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("    Training")
    for epoch in range(epochs):
        print(f"-------------------------------EPOCH N°{epoch}-------------------------------")
        train_loader.sampler.set_epoch(epoch)
        model.train()
        loss_all = 0
        total_correct_train = 0
        total_train = 0
        for j, data in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = torch.nn.functional.nll_loss(output, data.y)
            loss_eval += loss.item()
            prediction = output.argmax(dim=1)
            correct = int((prediction == data.y).sum())
            acc = correct/len(data.y)
            print(f"    Batch {j}. Loss: {loss.item()}. Accuracy: {acc}")
            total_correct_eval += correct
            total_eval += len(data.y)

    

