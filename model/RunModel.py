#!/usr/bin/env python3
# coding: utf-8

# Get the current script directory
from pathlib import Path
DIR = Path(__file__).parent.resolve()

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(str(DIR / 'pvfinder.mplstyle'))
# See https://matplotlib.org/users/style_sheets.html


import numpy as np
import argparse
import torch
import os


from collectdata import DataCollector
from loss import Loss
from training import trainNet, select_gpu
import models

# This bit of black magic pulls out all the Torch Models by name from the loaded models file.
MODELS = {x for x in dir(models)
              if not x.startswith("_")
                 and isinstance(getattr(models, x), type)
                 and torch.nn.Module in getattr(models, x).mro()}

def main(n_epochs, name, datafile, batch_size, learning_rate, model, output, gpu=None):

    device = select_gpu(gpu)

    Model = getattr(models, model)

    collector = DataCollector(datafile, 120_000, 10_000)
    train_loader = collector.get_training(batch_size, 120_000, device=device, shuffle=True)
    val_loader = collector.get_validation(batch_size, 10_000, device=device, shuffle=False)

    model = Model()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    output.mkdir(exist_ok=True)

    # Create our optimizer function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create our loss function
    loss = Loss()

    # Run the epochs
    for results in trainNet(model, optimizer, loss,
                            train_loader, val_loader,
                            n_epochs,
                            notebook = False):

        # Any prints in here are per iteration

        # Save each model state dictionary
        if output:
            torch.save(model.state_dict(), output / f'{name}_{results.epoch}.pyt')

    torch.save(model.state_dict(), output / f'{name}_final.pyt')

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(np.arange(len(results.cost))+1, results.cost, 'o-',color='r',label='Train')
    ax.plot(np.arange(len(results.val))+1, results.val, 'o-' , color='b', label='Validation')
    ax.set_xlabel('Number of epoch')
    ax.set_ylabel('Average cost per bin of a batch')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(str(output / f'{name}.png'))


if __name__ == '__main__':
    # Handy: https://gist.github.com/dsc/3855240

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Run example: CUDA_VISIBLE_DEVICES=0 ./RunModel.py 20180801_30000_2layer --model SimpleCNN2Layer")
    parser.add_argument('-e', '--epochs', type=int, default=200, help="Set the number of epochs to run")
    parser.add_argument('name', help="The name, such as date_numevents_model or similar")
    parser.add_argument('-d', '--data', default='/data/schreihf/PvFinder/Aug_15_140000.npz',
                        help="The data to read in, in npz format (for now)")
    parser.add_argument('-b', '--batch-size', type=int, default=32, dest='batch', help="The batch size")
    parser.add_argument('--learning-rate', type=float, default=1e-3, dest='learning', help="The learning rate")
    parser.add_argument('--model', required=True, choices=MODELS, help="Model to train on")
    parser.add_argument('--output', type=Path, default=Path('output'), help="Output directory")
    parser.add_argument('--gpu', help="Pick a GPU by bus order (you can use CUDA_VISIBLE_DEVICES instead)")

    args = parser.parse_args()
    main(args.epochs, args.name, args.data, args.batch, args.learning, args.model, args.output, args.gpu)
