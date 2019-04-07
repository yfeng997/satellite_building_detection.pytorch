import os
import random

train_config_file = './config/train_config.txt'
test_config_file = './config/test_config.txt'
images_dir = '/data/feng/building/images'

f = open(train_config_file, 'w')
t = open(test_config_file, 'w')

for root, dirs, files in os.walk(images_dir):
    for filename in files:
        if filename.endswith('.jpg'):
            if random.random() < 0.05:
                # write as test data
                t.write(os.path.join(root, filename)+'\n')
            else:
                # write as train data
                f.write(os.path.join(root, filename)+'\n')

f.close()
t.close()