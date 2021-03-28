import json
import argparse
import numpy as np
from pathlib import Path
from skimage.io import imsave

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToPILImage

from src.dataset import SuperMarioKartDataset
from src.processing import get_transforms
from src.modeling.autoencoders import get_encoder 

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
    parser.add_argument("--data_path",  required=True, type=Path, help="Parent of train/valid txt files")
    parser.add_argument("--save_path", default=None,  type=Path, help="Where to save models and visu")
    parser.add_argument("--load_path", default=None,  type=Path, help="From where to load pretrained")

    parser.add_argument("--config_path", required=True, type=Path, help="Encoder information")
    
    parser.add_argument("--viz_samples", default=5, type=int)
    parser.add_argument("--patience", default=10, type=int)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = load_json(args.config_path)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================Create data loaders=====================
    transforms = get_transforms(**config['preprocessing'])

    train_set = SuperMarioKartDataset(
        Path(args.data_path) / 'train_files.txt', transforms=transforms
    )
    valid_set = SuperMarioKartDataset(
        Path(args.data_path) / 'valid_files.txt', transforms=transforms
    )

    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'])
    valid_loader = DataLoader(train_set, batch_size=1)

    # ===================Encoder outputs visualization=====================
    if not args.save_path:
        save_path = args.data_path / (config['encoder']['enc_type'] + '_results')
    else:
        save_path = args.save_path
    save_path.mkdir(exist_ok=True)

    # ===================Instantiate models=====================
    input_shape = train_set.image_shape
    network = get_encoder(input_shape=input_shape, device=DEVICE, **config['encoder'])

    if args.load_path and args.load_path.exists():
        network.load_state_dict(torch.load(args.load_path / "best_encoder.pt"))

    best_val_loss = 999
    cnt_bad_epocs = 0

    for epoch in range(1, config['training']['epochs'] + 1):
        train_losses, valid_losses = [], []
        # ===================Training=====================
        for x_t in train_loader:
            x_t = x_t.to(DEVICE)
            train_loss = network.train(x_t)
            train_losses.append(train_loss)
        # ===================Validation=====================
        for i, x_t in enumerate(valid_loader):
            x_t = x_t.to(DEVICE)
            
            valid_loss, encoding, decoding = network.evaluate(x_t)
            valid_losses.append(valid_loss)

            if not epoch % 5 and i < args.viz_samples: # saving samples
                # create save path
                epc_save_path = save_path / f'epoch_{epoch}'
                epc_save_path.mkdir(exist_ok=True)
                # format images
                input_image = np.array(ToPILImage()(x_t.cpu()[0]))
                decoded_image = np.array(ToPILImage()(decoding.cpu()[0]))
                # merge images and save
                merged_image = np.concatenate((input_image, decoded_image))
                imsave(epc_save_path / f'result_{i}.png', merged_image)

        avg_trn_loss = round(np.mean(train_losses), 4)
        avg_val_loss = round(np.mean(valid_losses), 4)

        print(f'Epoch - {epoch} | Avg Train loss - {avg_trn_loss} | Avg Val loss - {avg_val_loss}')

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

def load_json(path):
    """
    Load json as python object
    """
    if not path.exists():
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj

if __name__ == '__main__':
    main()