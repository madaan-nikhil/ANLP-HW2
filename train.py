import argparse
import os
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from trainer import *
from model import *
from dataloaders.conll_dataloader import *


def _load_model(model_dir, filename, model, device):
    """
    Load the model checkpoint
    
    Args:
    - model_dir: checkpoints local directory
    - filename: name of the checkpoint file
    - rest: args used for creating the model instance
    """
    print(f'Loading model :{filename} {model_dir}')
    with open(os.path.join(model_dir, filename), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device)
    return model


if __name__ == '__main__': 
    
    NUM_LABELS = 15
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50, help="num of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--lr', type=float, default=0.00001, help="learning rate")
    parser.add_argument('--use-cuda', type=bool, default=False)# not used
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    parser.add_argument('--resume_from_epoch', type=int, default=0, help="checkpoint to resume")
    parser.add_argument('--training', type=int, default=1, help='Flag to train the model')
    parser.add_argument('--inference', type=int, default=0, help='inference flag')
    parser.add_argument('--data_dir', help="data directory")
    parser.add_argument('--model_dir', help="model directory")
    parser.add_argument('--model_file', help="model_file")
    parser.add_argument('--model_name', type=str, default="allenai/scibert_scivocab_cased", help="resume training")


    args, _ = parser.parse_known_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f'Loading data')
    file_path = "dataloaders/project-2-at-2022-10-22-19-26-4e2271c2.conll" # set args.data_dir
    train_dataset, val_dataset = get_loaders(file_path=file_path, 
                                            val_size=0.2, 
                                            tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding="longest"))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=args.batch_size, 
                                           shuffle=False, 
                                           collate_fn=SciDataset.collate_batch)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                           batch_size=args.batch_size, 
                                           shuffle=False, 
                                           collate_fn=SciDataset.collate_batch)
    print(f'Data Loaded')
    
    model_name = args.model_name 
    # Download pytorch model
    feature_extractor = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding="longest")
    model = TokenClassification(feature_extractor,
                                device,
                                NUM_LABELS,
                                feature_extractor.config.hidden_size,
                                dropout=0.1)
    
    if args.resume:
        model = _load_model(args.model_dir, filename=args.model_file, model=model, device=device)
    
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, min_lr=1e-8)
    criterion = torch.nn.CrossEntropyLoss()

    print("Model Training begins")
    if args.training:
        model = model.to(device)
        trainer = Trainer(model,
                        args.epochs,
                        optimizer,
                        schedular,
                        criterion,
                        train_loader,
                        val_loader,
                        device,
                        args.model_dir,
                        NUM_LABELS,
                        resume = args.resume)

        trainer.fit()