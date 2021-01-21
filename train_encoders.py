import json
import argparse
import numpy as np
from pathlib import Path
from skimage.io import imsave

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToPILImage

from piq import SSIMLoss # image quality losses

from src.dataset import SuperMarioKartDataset
from src.processing import get_transforms
from src.modeling import autoencoders

"""
ToDos:
    1. Resize images
    2. Add noise loss
    3. Refactor code
    4. Use pretrained VGG layers
    5. Consider adding FOV - accuracy decreases with distance
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--save_path", default=None, type=Path)
    parser.add_argument("--load_path", default=None, type=Path)

    parser.add_argument("--config_name", required=True)
    parser.add_argument("--encoder_type", required=True, type=str)
    
    parser.add_argument("--viz_samples", default=5, type=int)
    parser.add_argument("--patience", default=10, type=int)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = load_json('./configs', args.config_name)
    config = config[args.encoder_type] 
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
        print('CUDA is not available, using CPU instead...')

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
        save_path = args.data_path / (args.encoder_type + '_results')
    else:
        save_path = args.save_path
    save_path.mkdir(exist_ok=True)

    # ===================Instantiate models=====================
    img_shape = train_set.image_shape
    if args.encoder_type == 'pca':
        network = autoencoders.PCA(img_shape, config['latent_dim']).cuda()
    elif args.encoder_type == 'mlp':
        network = autoencoders.MLP(img_shape, config['latent_dim']).cuda()

    optimizer = torch.optim.Adam(network.parameters())
    ssim_loss = SSIMLoss(data_range=1.)
    mse_loss = nn.MSELoss()

    load_path = args.load_path
    if load_path and load_path.exists():
        network.load_state_dict(torch.load(load_path / "best_encoder.pt"))

    best_val_loss = 999
    cnt_bad_epocs = 0

    for epoch in range(1, config['n_epochs'] + 1):
        train_losses, valid_losses = [], []
        epc_save_path = save_path / f'epoch_{epoch}'
        epc_save_path.mkdir(exist_ok=True)
        # ===================Training=====================
        for x_t in train_loader:
            x_t = x_t.cuda()

            encoding, decoding = network(x_t)
            step_mse = mse_loss(decoding, x_t)
            loss = mse_loss(decoding, x_t) + 0.1 * ssim_loss(decoding, x_t)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================Validation=====================
        for i, x_t in enumerate(valid_loader):
            x_t = x_t.cuda()
            
            with torch.no_grad():
                encoding, decoding = network(x_t)
                step_mse = mse_loss(decoding, x_t)
                step_ssim = 0.1 * ssim_loss(decoding, x_t)
                loss = step_mse + step_ssim

            if i < args.viz_samples: # display only the first 10 samples
                input_image = np.array(ToPILImage()(x_t.cpu()[0]))
                decoded_image = np.array(ToPILImage()(decoding.cpu()[0]))
                merged_image = np.concatenate((input_image, decoded_image))
                imsave(epc_save_path / f'result_{i}.png', merged_image)

            valid_losses.append(step_mse.item())

        avg_trn_loss = round(np.mean(train_losses), 4)
        avg_val_loss = round(np.mean(valid_losses), 4)

        print(f'Epoch - {epoch} | Avg Train MSE - {avg_trn_loss} | Avg Val MSE - {avg_val_loss}')

        # ===================Checkpointing=====================
        if best_val_loss > avg_val_loss:
            cnt_bad_epocs = 0
            best_val_loss = avg_val_loss
            torch.save(network.state_dict(), save_path / 'best_encoder.pt')
        
        # ===================Early stopping=====================
        if best_val_loss < avg_val_loss:
            cnt_bad_epocs += 1

        if cnt_bad_epocs == args.patience:
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