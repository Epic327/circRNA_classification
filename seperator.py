import sys
import random

data = list()
with open('data/circRNA_dataset.bed', 'r') as f:
    for line in f:
        data.append([line, 'pos'])

with open('data/negative_dataset.bed', 'r') as f:
    for line in f:
        data.append([line, 'neg'])

random.shuffle(data)
length = len(data)
train_val_ind = int(length * 0.85)
train_val_data = data[:train_val_ind]
test_data = data[train_val_ind:]

pos_train = open('data/pos_train.bed', 'w+')
neg_train = open('data/neg_train.bed', 'w+')

for line in train_val_data:
    if line[1] == 'pos':
        pos_train.write(line[0])
    elif line[1] == 'neg':
        neg_train.write(line[0])
    else:
        print('fail')
        sys.exit()

pos_train.close()
neg_train.close()

test_file = open('data/true_test.bed', 'w+')
test_label = open('data/true_test_label.txt', 'w+')

for line in test_data:
    test_file.write(line[0])
    test_label.write(line[1]+'\n')

test_file.close()
test_label.close()