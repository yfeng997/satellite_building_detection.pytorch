import os
from shutil import copyfile

sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
output_dir = 'wc_finetune'

for size in sizes:
    # take as much as we could from non-residential data. If not enough, fill with residential data
    # output path: wc_finetune/size/residential(non_residential)
    res_output = output_dir + '/' + str(size) + '/train/' + 'residential'
    nonres_output = output_dir + '/' + str(size) + '/train/' + 'non_residential'
    if not os.path.exists(res_output):
        os.makedirs(res_output)
    if not os.path.exists(nonres_output):
        os.makedirs(nonres_output)
    nonres_size = size / 2
    nonres_count = 0
    for root, dirs, files in os.walk('images/train/non_residential'):
        for filename in files:
            source = root + '/' + filename
            target = nonres_output + '/' + filename
            copyfile(source, target)
            nonres_count += 1
            if nonres_count >= nonres_size:
                break
    res_size = size - nonres_count
    res_count = 0
    for root, dirs, files in os.walk('images/train/residential'):
        for filename in files:
            source = root + '/' + filename
            target = res_output + '/' + filename
            copyfile(source, target)
            res_count += 1
            if res_count >= res_size:
                break
    