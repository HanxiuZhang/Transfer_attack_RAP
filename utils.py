## get hyperparameters

import argparse

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str, default="./save/")

    parser.add_argument('--source_model', type=str, default='resnet50', choices=['resnet50', 'inception-v3', 'densenet121', 'vgg16bn'])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_iterations', type=int, default=400)

    parser.add_argument('--loss_function', type=str, default='CE', choices=['CE','MaxLogit'])

    parser.add_argument('--targeted', action='store_true')

    parser.add_argument('--m1', type=int, default=1, help='number of randomly sampled images')
    parser.add_argument('--m2', type=int, default=1, help='num of copies')
    parser.add_argument('--strength', type=float, default=0)

    parser.add_argument('--adv_perturbation', action='store_true')

    parser.add_argument('--adv_loss_function', type=str, default='CE', choices=['CE', 'MaxLogit'])

    parser.add_argument('--adv_epsilon', type=eval, default=16/255)
    parser.add_argument('--adv_steps', type=int, default=8)

    parser.add_argument('--transpoint', type=int, default=0)

    parser.add_argument('--seed', type=int, default=0)


    parser.add_argument('--MI', action='store_true')
    parser.add_argument('--DI', action='store_true')
    parser.add_argument('--TI', action='store_true')
    parser.add_argument('--SI', action='store_true')
    parser.add_argument('--random_start', action='store_true')


    parser.add_argument('--save', action='store_true')

    parser.add_argument('--device', type=int, default=0)


    arg = parser.parse_args()
    
    return arg

import os

def makedir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('----------- new folder ------------')
        print('------------ ok ----------------')
    
    else:
        print('----------- There is this folder! -----------')


def logging(s, print_=True, log_=True, path = ''):

    if print_:
        print(s)

    if log_:
        with open(os.path.join(path, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')

##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list