# MLproject

Here is the implementation of 3 different decoder algorithms. Each decoder predicts the spatial position of a rat from
the neural activity in the rat's hippocampus. I motivate the approaches and evaluate their accuracy [in this report](mcclain_ML_project.pdf)

The coordinate_decoder file is the main file that coordinates the others. The the bayesian_decoder, nn_decoder, and simple_regression
files have objects for each of the different types of decoder. To switch between decoders you must change the instantiation of
the decoder in coordinate_decoder. The analysis file holds some utilities and basic computations that are used throughout the
other files. The data_processing file contains an object that loads the data and performs preprocessing steps on it. The 
tuning_properties file holds and object for computing the average firing rates of the cells as a function of space. This is used in the bayesian_decoder.
