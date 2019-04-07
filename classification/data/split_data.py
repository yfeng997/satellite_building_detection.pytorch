from shutil import copyfile
import random
import os

for label in ['residential', 'non_residential']:
    for finetune_size in ['500', '1000', '2000', '5000', '10000']:
        for root, dirs, files in os.walk('/data/feng/building-classify/wc_finetune/' + finetune_size + '/' + label):
            for filename in files:
                r = random.random()
                target_dir = ''
                if r < 0.1:
                    target_dir = 'test'
                elif r < 0.2:
                    target_dir = 'val'
                else:
                    target_dir = 'train'
                target_filename = '/data/feng/building-classify/wc_finetune/' + finetune_size + '/' + target_dir + '/' + label + '/' + filename
                if not os.path.exists('/data/feng/building-classify/wc_finetune/' + finetune_size + '/' + target_dir + '/' + label + '/'):
                    os.makedirs('/data/feng/building-classify/wc_finetune/' + finetune_size + '/' + target_dir + '/' + label + '/')
                source_filename = '/data/feng/building-classify/wc_finetune/' + finetune_size + '/' + label + '/' + filename
                copyfile(source_filename, target_filename)