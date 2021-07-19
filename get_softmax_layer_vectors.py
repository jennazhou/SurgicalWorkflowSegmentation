#! /usr/bin/env python
# script to train the model with cross-validation and get softmax layer vector outputs saved

# imports
import pickle
import os
import glob
import numpy as np

from model import Endo3D
import torch
import torch.nn as nn
from torch.utils import data
import json
from sequence_loder import sequence_loder


# not need f1 scores for now
# get predicted labels, true labels and softmax layer probability output for the given video name
def sequence_maker(model, video_name, div, device):
    sacro = sequence_loder(div, device)
    sacro.whole_len_output(video_name)
    whole_loder = data.DataLoader(sacro, 20)
    seq_pre = []
    seq_true = []
#     fc_list = []
    output_list = []
    for labels_val, inputs_val in whole_loder:
        inputs_val = inputs_val.float().to(device)
        labels_val = labels_val.long().to(device)
        # output is the log of softmax output probabilities
        output, x_fc = model.forward_cov(inputs_val)
        _, predicted_labels = torch.max(output.cpu().data, 1)
        for i in range(predicted_labels.numpy().shape[0]):
            seq_pre.append(predicted_labels.numpy()[i])
            seq_true.append(labels_val.cpu().numpy()[i])
            output_list.append(output.cpu()[i])
#             fc_list.append(x_fc[i, :])
    return seq_pre, seq_true, output_list





######### code to run the model on test data to get softmax layer output #########
# to load onto GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ALL PATHS
paramPath = "/home/yitong/venv_yitong/sacro_wf_analysis/params"
path_prefix= '/home/yitong/venv_yitong/sacro_wf_analysis/'
whole_path = path_prefix + 'data/sacro_sequence/whole'
videos_path = path_prefix + 'data/sacro_jpg'
save_path_prefix = '/home/jenna/jennaCode/model_outputs/'
    
folders = ['folder1', 'folder2', 'folder3', 'folder4', 'folder5', 'folder6', 'folder7']
div = ['div1', 'div2', 'div3', 'div4', 'div5', 'div6', 'div7']

for counter, folder in enumerate(folders):
    print('The current folder is:' + folder)
    # whole data sequence data path
    whole_folder_path = os.path.join(whole_path, folder)
    # get divisions for cross validation
    with open(os.path.join(videos_path, 'dataset_' + div[counter] + '.json'), 'r') as json_data:
        temp = json.load(json_data)
    video_list = temp['test']
    print('The current test video list: {}'.format(video_list))
    
    # construct model 
    vanilla_net = Endo3D().to(device)
    Endo3D_state_dict = vanilla_net.state_dict()
    # update the state dict based on stored parameters
    pre_state_dict = torch.load(path_prefix + 'params/cross_validation/' + div[counter] + '/params_c3d.pkl')
    new_state_dict = {k: v for k, v in pre_state_dict.items() if k in Endo3D_state_dict}
    Endo3D_state_dict.update(new_state_dict)
    vanilla_net.load_state_dict(Endo3D_state_dict)


    for video_name in video_list:
        save_path = os.path.join(save_path_prefix, video_name)
        # create the path if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with torch.no_grad():
            seq_pred, seq_true, output_list = sequence_maker(vanilla_net, video_name, div[counter], device)
            # change the values in output list from log values to positive probabilities
            probs_list = []
            for tensor in output_list:
                probs = torch.exp(tensor)
                probs_list.append(probs.numpy())
            # change the list to numpy array
            probs_list = np.array(probs_list)
            print("The probability vector list of video {} is:".format(video_name))
            print(probs_list)
            
            # save the array of probability vectors as well as the prediction labels into files for furture references
            prob_file = open(os.path.join(save_path, 'prob_vectors.pickle'), 'wb+')
            pickle.dump(probs_list, prob_file)
            prob_file.close()
            
            pred_file = open(os.path.join(save_path, 'seq_pred.pickle'), 'wb+')
            pickle.dump(seq_pred, pred_file)
            pred_file.close()