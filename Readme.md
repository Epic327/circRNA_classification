# Circular RNA classifaction with circDeep dataset

This file describes how to perform our experiments using the dataset provided by [circDeep](https://github.com/UofLBioinformatics/circDeep)


## Installation

Requirements:
* pysam, only needed for the creation of the ground truth files. It requires Ubuntu. However, if the files are directly downloaded from this repository, it is not needed.
* numpy
* keras
* tensorflow-gpu (or the backend which you prefer)
* scikit-learn

TODO: add version numbers

Installation instructions are given for Anaconda with Tensorflow GPU Backend.

```
conda config --add channels bioconda
conda config --add channels anaconda
conda config --add channels conda-forge
conda create --name circRNA pysam numpy tensorflow-gpu keras scikit-learn
conda activate circRNA
```

## Additional Data

Due to the large file size the genome file has to be downloaded seperately from my [google drive](https://drive.google.com/open?id=1boQCOPhm_Ht6ldzIppkqWG825l-EEa0R) and needs to be put into the folder 'data'.
Alternatively it can be downloaded from the [UCSC Genome Browser](https://genome.ucsc.edu/)

## Simple Instructions

In order to recreate the results simply run the following command:
```
python classifier.py --name y --in_len 200
```

Additionally if you want to run the different experimental setups, as they can be found in the data folder, simply copy the files of the respective folder and paste them into the 'circRNA_classification' folder (The folder which contains classifier.py). 
Lastly if you do not wish to train the model but simply want to reproduce the results, just copy the files from one of the experimental setups (found in the 'data') folder and paste them into the 'circRNA_classification' folder. Furthermore comment line 31 in classifier.py ('train_net(x_train, y_train, name, in_len)'). Now the code will not train a model but simply load the model and weights from the files in the 'circRNA_classification' folder and perform evaluation based on them.

If you wish to recreate the ground truth files yourself, first run these commands:

```
python seperator.py
python make_data.py
python make_complete_train.py
```

## Process

This section describes how to use the provided data to recreate the results.


### Creating the Dataset

circDeep provided two files for the dataset, one for the [positive instances](data/circRNA_dataset.bed) and one for the [negative ones](data/negative_dataset.bed).

The positive dataset contains **31939** samples and the negative one contains **19683** samples which is equals the number of samples mentioned in their [paper (Section 3.1)](https://academic.oup.com/bioinformatics/article/36/1/73/5527751).

However, the provided files only mention the samples and have not been divided into training and test test. In order to do that we will have to call [seperator.py](seperator.py)

Tasks of this script are:
* reading the positive and negative file into one list
* randomly shuffle the list
* split the shuffled list into training and testset, using 85% as the training size
* it will create a file for the [negative training](data/neg_train.bed) samples, a file for the [positive training](data/pos_train.bed) samples, one for the [testing](data/true_test.bed) samples and a label file, in which each line corresponds to the same respective line in the testset and contains the label of said sample. However, the last file ist not needed for further processing.

The created positive train file contains **27134** instances and the test file contains **4805** instances, which add up the initial **31939** instances. The same applies for the negative train file with **16744** samples and the **2939** negative instances in the test file, which add up to the initial **19683** samples.
The numbers can slightly vary after performing the operation yourself, but the sum should always match.

### Processing the data files

Using the information provided by the previously created bed files, we now have to create files which actually contain the sequence information. This can be done using [make_data.py](make_data.py).

This script was adapted from the method bed_to_fasta, provided by the implementation of circDeep. The tasks are:
* Creating for every previously created bed file a corresponding text file which contains the sequence information
* These files are: [pos_train.txt](data/pos_train.txt), [neg_train.txt](data/neg_train.txt) and [test.txt](data/test.txt)
* The output is written in a format that can be used by iLearn:
* >ID|Label|Set\n nt_seq
* where ID is just a numbred id, label is 0 for negative samples and 1 for positive ones, Set is either training or testing and nt_seq is the nucleotide sequence.
* This file also contains parameters to adapt how the sequence should be extracted. Namely which padding technique, sequence length etc.

### Combining both training files into one

[make_complete_train.py](make_complete_train.py) will combine both negative and positive training files into one.

It will also randomly shuffle the cmobined training file.

### Training and Evaluation

[classifier.py](classifier.py) performs the training and evaluation.
It will create a list of the one hot encoded sequences and another list containing the labels for the train and test set seperately.
Then training is performed using our autoencoder architecture and the training lists.
The model files are saved.
Finally, using this model file predictions are made on the test list and the accuracy is printed.
