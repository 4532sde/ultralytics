import os
import random

#ROOT = '/data/home/202221000635/ultralytics-main/trans/datasets/Fruit/'
ROOT = '/data/home/202221000635/ultralytics-main/trans/redproject/Fruit/'
trainval_percent = 0.9
train_percent = 0.9
xmlfilepath = os.path.join(ROOT, 'Annotations')  # 使用os.path.join()拼接路径
txtsavepath = os.path.join(ROOT, 'ImageSets')   # 使用os.path.join()拼接路径

# 确保目录存在，如果不存在则创建
os.makedirs(txtsavepath, exist_ok=True)

total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')  # 使用os.path.join()拼接路径
ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')          # 使用os.path.join()拼接路径
ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')        # 使用os.path.join()拼接路径
fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')            # 使用os.path.join()拼接路径

for i in list:
    name = total_xml[i][:-4] + '\n'
    print (name)
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
