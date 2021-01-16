import json
import argparse
import numpy as np
from pathlib import Path
from skimage.io import imsave

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from src.dataset import SuperMarioKartDataset
from src.processing import get_transforms
from src.modeling import autoencoders

"""
ToDos:
    1. Add context loss
    2. Add noise loss
    3. Try MLP network 
    4. Use pretrained VGG layers
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--save_path", default=None, type=Path)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = load_json('./configs', args.config_name)
    config = config['pca'] # take config info for PCA

    # ===================Create data loaders=====================
    transforms = get_transforms(roi=config['roi'], grayscale=config['grayscale'])

    train_set = SuperMarioKartDataset(
        Path(args.data_path) / 'train_files.txt', transforms=transforms
    )
    valid_set = SuperMarioKartDataset(
        Path(args.data_path) / 'valid_files.txt', transforms=transforms
    )

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], num_workers=4)
    valid_loader = DataLoader(train_set, batch_size=1, num_workers=4)

    # ===================Export some val images=====================
    if not args.save_path:
        save_path = args.data_path / 'results'
    else:
        save_path = args.save_path
    save_path.mkdir(exist_ok=True)
    
    for i, x_t in enumerate(valid_loader):
        # save original files to have comparions
        imsave(save_path / f'sample_{i}.png', x_t.numpy()[0][0], check_contrast=False)
        if i == 5:
            break

    # ===================Instantiate models=====================
    img_shape = train_set.image_shape
    network = autoencoders.PCA(img_shape, config['latent_dim']).cuda()

    optimizer = torch.optim.Adam(network.parameters())
    criterion = nn.MSELoss()

    best_val_loss = 999
    max_bad_epocs = 3
    cnt_bad_epocs = 0

    for epoch in range(1, config['n_epochs'] + 1):
        train_losses, valid_losses = [], []
        epc_save_path = save_path / f'epoch_{epoch}'
        epc_save_path.mkdir(exist_ok=True)
        # ===================Training=====================
        for x_t in train_loader:
            x_t = x_t.cuda()

            encoding, decoding = network(x_t)
            loss = criterion(decoding, x_t)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================Validation=====================
        for i, x_t in enumerate(valid_loader):
            x_t = x_t.cuda()
            
            with torch.no_grad():
                encoding, decoding = network(x_t)
                loss = criterion(decoding, x_t)

            if i < 10:
                decoded_img = decoding.cpu().numpy()[0][0].astype(np.uint8)
                imsave(epc_save_path / f'decoding_{i}.png', decoded_img, check_contrast=False)

            valid_losses.append(loss.item())

        avg_trn_loss = round(np.mean(train_losses), 4)
        avg_val_loss = round(np.mean(valid_losses), 4)

        print(f'Epoch - {epoch} | Avg Train Loss - {avg_trn_loss} | Avg Val Loss - {avg_val_loss}')

        # ===================Checkpointing=====================
        if best_val_loss > avg_val_loss:
            cnt_bad_epocs = 0
            best_val_loss = avg_val_loss
            torch.save(network.state_dict(), save_path / 'best_encoder.pt')
        
        # ===================Early stopping=====================
        if best_val_loss < avg_val_loss:
            cnt_bad_epocs += 1

        if cnt_bad_epocs == max_bad_epocs:
            print('Network is not improving. Stopping training...') 
            break

def load_json(fdir, name):
    """
    Load json as python object
    """
    path = Path(fdir) / "{}.json".format(name)
    if not path.exists():
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj

if __name__ == '__main__':
    main()