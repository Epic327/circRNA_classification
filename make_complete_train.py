import random
import os

# combine positive and negative training file into one

def main():
	neg = open(os.path.join('data', 'neg_train.txt'), 'r')
	pos = open(os.path.join('data', 'pos_train.txt'), 'r')
	complete = open(os.path.join('data','train.txt'), 'w')

	labels = list()
	sequence = list()

	# extract the information
	for line in neg:
		if line.startswith('>'):
			labels.append(line)
		else:
			sequence.append(line)
	
	for line in pos:
		if line.startswith('>'):
			labels.append(line)
		else:
			sequence.append(line)

	neg.close()
	pos.close()

	# combine sequence and label information in one list
	pairs = list()

	for label, seq in zip(labels, sequence):
		pairs.append((label, seq))

	random.shuffle(pairs)

	# save the information
	for elem in pairs:
		complete.write(elem[0])
		complete.write(elem[1])

	complete.close()

if __name__ == '__main__':
	main()
