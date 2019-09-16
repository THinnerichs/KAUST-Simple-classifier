### KAUST: Simple classifier

This project aims to classify true and false intron and exon 5' segregators with a basic classifier. As written, it is mainly used as a toy project to work on data preprocessing and neural network model fitting.
This work was performed at KAUST, Saudi Arabia.

#### Description of the preprocessing and model

The original data is given as a large file of records of the DNA of Arabidopsis thaliana in fasta format. 

In this project the libraries Biopython, SciKit-Learn and Keras were used. 

The initial data is parsed with the SeqIO module of Biopython. Afterwards a One Hot Encoding of the nucleotide sequences is performed. For the direct preparation for the machine learning model and its robustness a k-fold cross validation is performed. 

Different kinds of neural networks like shallow, intermediate and deep were applied and their performance was documented in the form of mean and standard deviation. 

#### Usage

The Convert_fa_to_fasta.py script converts the raw data to files that are handable by the SeqIO parser and needs to performed at first.

Second, perform the Prepare_data.py script to store the given, preprocessed data in .npy files. 

This data is used in the third step in the simple_classifier.py script to actually execute the ML model. 

#### Results

Results are stored in the results directory.

