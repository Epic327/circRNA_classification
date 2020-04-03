import pysam
import sys
import os
import math

#creates text files containing the sequence information for training and test set

def main():
    #load files
    pos = os.path.join('data', 'pos_train.bed')
    neg = os.path.join('data', 'neg_train.bed')
    test = os.path.join('data', 'true_test.bed')
    out1 = os.path.join('data', 'pos_train.txt')
    out2 = os.path.join('data', 'neg_train.txt')
    out3 = os.path.join('data', 'test.txt')
    genome = os.path.join('data', 'hg38.fa')
    i = 0
    # set parameters which settings should be used for creating the file
    adjacency_flag = True
    padding =  2  # 0: pre, 1: post, 2:middle
    seq_len = 1000
    adj_len = 100
    print('Pos')
    i = bed_to_fasta(pos, out1, genome, 'pos', i, adjacency_flag, padding, seq_len, adj_len)
    print('Neg')
    i = bed_to_fasta(neg, out2, genome, 'neg', i, adjacency_flag, padding, seq_len, adj_len)
    print('Test')
    i = bed_to_fasta(test, out3, genome, 'test', i, adjacency_flag, padding, seq_len, adj_len)


def bed_to_fasta(bed_file, fasta_file, genome_fasta, label, counter, adjacency_flag, padding, seq_len, adj_len):
    # the main method to create the sequence files
    bed = open(bed_file, 'r');

    output = open(fasta_file, 'w');
    genome_fa = pysam.FastaFile(genome_fasta)

	# if adjacent nucleotides should be used
    if adjacency_flag:
        for line in bed:
			# get the location information about the sequences and fetch it
            values = line.split()

            chr_n = values[0]
            start = int(values[1])
            end = int(values[2])
            strand = values[3]
            seq = genome_fa.fetch(chr_n, start-adj_len, end+adj_len)

            seq = seq.upper()

			# set label and set information
            if label == 'test':
                if values[4].startswith('hsa'):
                    label1 = '1'
                else:
                    label1 = '0'
                set_var = 'testing'
            else:
                set_var = 'training'
                if label == 'pos':
                    label1 = '1'
                else:
                    label1 = '0'

			# reverse complement the sequence if necessary
            if strand == '-':
                seq = reverse_complement(seq)

			# Use appropriate padding technique and add/remove nucleotides
            # 0: pre, 1: post, 2:middle
            if padding == 0:

                if len(seq) > seq_len:
                    to_remove =  len(seq) - seq_len
                    seq = seq[to_remove:]
                elif len(seq) < seq_len:
                    to_add = seq_len - len(seq)
                    seq = 'Z'*to_add + seq 

            elif padding == 1:
                
                if len(seq) > seq_len:
                    to_remove =  len(seq) - seq_len
                    seq = seq[:-to_remove]
                elif len(seq) < seq_len:
                    to_add = seq_len - len(seq)
                    seq =  seq + 'Z'*to_add

            elif padding == 2:

                if len(seq) > seq_len:
                    to_keep = seq_len/2.0
                    seq = seq[:math.floor(to_keep)] + seq[-math.ceil(to_keep):]
                elif len(seq) < seq_len:
                    to_add = seq_len - len(seq)
                    middle = int(len(seq)/2)
                    seq =  seq[:middle] + 'Z'*to_add + seq[middle:]
            else:
                print('error')
                sys.exit()

			# write output line
            output.write('>' + '|'.join([str(counter), label1, set_var]) + '\n')
            output.write(seq + '\n')
            counter += 1
	# similar procedure if no adjacent nucleotides should be used
    else:
        for line in bed:

            values = line.split()

            chr_n = values[0]
            start = int(values[1])
            end = int(values[2])
            strand = values[3]
            seq = genome_fa.fetch(chr_n, start, end)

            seq = seq.upper()

            if label == 'test':
                if values[4].startswith('hsa'):
                    label1 = '1'
                else:
                    label1 = '0'
                set_var = 'testing'
            else:
                set_var = 'training'
                if label == 'pos':
                    label1 = '1'
                else:
                    label1 = '0'

            if strand == '-':
                seq = reverse_complement(seq)

            # 0: pre, 1: post, 2:middle
            if padding == 0:

                if len(seq) > seq_len:
                    to_remove =  len(seq) - seq_len
                    seq = seq[to_remove:]
                elif len(seq) < seq_len:
                    to_add = seq_len - len(seq)
                    seq = 'Z'*to_add + seq 

            elif padding == 1:
                
                if len(seq) > seq_len:
                    to_remove =  len(seq) - seq_len
                    seq = seq[:-to_remove]
                elif len(seq) < seq_len:
                    to_add = seq_len - len(seq)
                    seq =  seq + 'Z'*to_add

            elif padding == 2:

                if len(seq) > seq_len:
                    to_keep = seq_len/2.0
                    seq = seq[:math.floor(to_keep)] + seq[-math.ceil(to_keep):]
                elif len(seq) < seq_len:
                    to_add = seq_len - len(seq)
                    middle = int(len(seq)/2)
                    seq =  seq[:middle] + 'Z'*to_add + seq[middle:]
            else:
                print('error')
                sys.exit()

            output.write('>' + '|'.join([str(counter), label1, set_var]) + '\n')
            output.write(seq + '\n')
            counter += 1

    return counter


def reverse_complement(seq):
    # reverse complement a given sequence
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))


def complement(seq):
    # definition of the complement nucleotides
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    complseq = [complement[base] for base in seq]
    return complseq


if __name__ == '__main__':
    main()