from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import pandas as pd

def eval(model, split, seq_length, n_cpu, disp):
    dataset = GolfDB(data_file='/content/drive/MyDrive/Github/joeklein-gaalevy-wemarti-fp/data/val_split_{}.pkl'.format(split),
                     vid_dir='/content/drive/MyDrive/Github/joeklein-gaalevy-wemarti-fp/data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)
    
    df = pd.read_pickle('/content/drive/MyDrive/Github/joeklein-gaalevy-wemarti-fp/data/val_split_{}.pkl'.format(split))

    correct = []

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        _, _, _, _, c = correct_preds(probs, labels.squeeze())
        correct.append(c)

    # add PCE to data frame for reporting
    df[['address', 'toe_up', 'mid_backswing', 'top', 'mid_downswing', 'impact', 'mid_follow_through', 'finish']] = pd.DataFrame(correct, index = df.index)
    df['N'] = 1

    # sex report
    sex_df = df.groupby('sex').agg(N = ('N', 'count'),
                                  address = ('address', 'mean'), 
                                  toe_up = ('toe_up', 'mean'), 
                                  mid_backswing = ('mid_backswing', 'mean'), 
                                  top = ('top', 'mean'), 
                                  mid_downswing = ('mid_downswing', 'mean'), 
                                  impact = ('impact', 'mean'), 
                                  mid_follow_through = ('mid_follow_through', 'mean'), 
                                  finish = ('finish', 'mean')).reset_index()

    # club report
    club_df = df.groupby('club').agg(N = ('N', 'count'),
                                    address = ('address', 'mean'), 
                                    toe_up = ('toe_up', 'mean'), 
                                    mid_backswing = ('mid_backswing', 'mean'), 
                                    top = ('top', 'mean'), 
                                    mid_downswing = ('mid_downswing', 'mean'), 
                                    impact = ('impact', 'mean'), 
                                    mid_follow_through = ('mid_follow_through', 'mean'), 
                                    finish = ('finish', 'mean')).reset_index()

    # view report
    view_df = df.groupby('view').agg(N = ('N', 'count'),
                                    address = ('address', 'mean'), 
                                    toe_up = ('toe_up', 'mean'), 
                                    mid_backswing = ('mid_backswing', 'mean'), 
                                    top = ('top', 'mean'), 
                                    mid_downswing = ('mid_downswing', 'mean'), 
                                    impact = ('impact', 'mean'), 
                                    mid_follow_through = ('mid_follow_through', 'mean'), 
                                    finish = ('finish', 'mean')).reset_index()

    # slow report
    slow_df = df.groupby('slow').agg(N = ('N', 'count'),
                                    address = ('address', 'mean'), 
                                    toe_up = ('toe_up', 'mean'), 
                                    mid_backswing = ('mid_backswing', 'mean'), 
                                    top = ('top', 'mean'), 
                                    mid_downswing = ('mid_downswing', 'mean'), 
                                    impact = ('impact', 'mean'), 
                                    mid_follow_through = ('mid_follow_through', 'mean'), 
                                    finish = ('finish', 'mean')).reset_index()

    PCE = np.mean(correct)
    return PCE, df, sex_df, club_df, view_df, slow_df
    

if __name__ == '__main__':

    split = 1
    seq_length = 64
    n_cpu = 1

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load('models/swingnet_2000.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    PCE, df, sex_df, club_df, view_df, slow_df = eval(model, split, seq_length, n_cpu, True)
    
    print('Average PCE: {}'.format(PCE))
    display(sex_df)
    display(club_df)
    display(view_df)
    display(slow_df)