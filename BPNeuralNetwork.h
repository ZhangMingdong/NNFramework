#pragma once


#include "LabeledVector.h"
#include <vector>

class NNImageList;
class NNImage;

/*** The neural network data structure.  The network is assumed to
be a fully-connected feedforward three-layer network.
Unit 0 in each layer of units is the threshold unit; this means
that the remaining units are indexed from 1 to n, inclusive.
***/
class BPNeuralNetwork
{
private:
	BPNeuralNetwork(int n_in, int n_hidden, int n_out);
public:
	~BPNeuralNetwork();
public:
	int input_n;                  /* number of input units */
	int hidden_n;                 /* number of hidden units */
	int output_n;                 /* number of output units */

	double *input_units;          /* the input units */
	double *hidden_units;         /* the hidden units */
	double *output_units;         /* the output units */

	double *hidden_delta;         /* storage for hidden unit error */
	double *output_delta;         /* storage for output unit error */

	double *target;               /* storage for target vector */

	double **input_weights;       /* weights from input to hidden layer */
	double **hidden_weights;      /* weights from hidden to output layer */

								  /*** The next two are for momentum ***/
	double **input_prev_weights;  /* previous change on input to hidden wgt */
	double **hidden_prev_weights; /* previous change on hidden to output wgt */
public:
	// create a new network
	static BPNeuralNetwork* Create(int n_in, int n_hidden, int n_out);
	// read the network from a file
	static BPNeuralNetwork* Read(char *filename);
	// initialize the random seed
	static void Initialize(unsigned int seed);
public:
	/* 
		train the network for one epoch
			eo: error of output layer
			eh: error of hidden layer
	*/
	void UpdateBackward(double eta, double dmomentum, double *eo, double *eh);
	// run the network
	void FeedForward();
	// save the model
	void Save(char *filename);

	// calculate the performance on the image list
	void CalculatePerformance(const std::vector<LabeledVector*>& vecLV, int list_errors);

	// load data for the input layer and labels
	void LoadInputData(const LabeledVector* pLV);

	// calculate label of the current input layer
	int CalculateLabel();

	// train the net
	void TrainNet(int nEpochs, const std::vector<LabeledVector*>& vecLV, double eta, double dmomentum);


private:
	// return whether the result is agree with the lable and compute the loss function
	int evaluatePerformance(double *err);
};