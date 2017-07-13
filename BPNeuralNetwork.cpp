#include "BPNeuralNetwork.h"

#include <stdio.h>
#include<stdlib.h>
#include <string>
#include <iostream>

#include <math.h>

#include "NNImage.h"

#define TARGET_HIGH 0.9
#define TARGET_LOW 0.1


#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register int _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

/*** Return random number between 0.0 and 1.0 ***/
double drnd()
{
	return ((double)rand() / RAND_MAX);
}

/*** Return random number between -1.0 and 1.0 ***/
double dpn1()
{
	return ((drnd() * 2.0) - 1.0);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/
double squash(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

/*** Allocate 2d array of doubles ***/
double** Create2DArray(int m, int n)
{
	double** buf = new double*[m];
	for (int i = 0; i < m; i++) {
		buf[i] = new double[n];
	}

	return buf;
}

void bpnn_randomize_weights(double **w, int m, int n)
{
	for (int i = 0; i <= m; i++) {
		for (int j = 0; j <= n; j++) {
			w[i][j] = 0.1 * dpn1();
		}
	}
}

void bpnn_zero_weights(double **w, int m, int n)
{
	for (int i = 0; i <= m; i++) {
		for (int j = 0; j <= n; j++) {
			w[i][j] = 0.0;
		}
	}
}

void bpnn_layerforward(double *l1, double *l2, double **conn, int n1, int n2)
{
	/*** Set up thresholding unit ***/
	l1[0] = 1.0;

	/*** For each unit in second layer ***/
	for (int j = 1; j <= n2; j++) {

		/*** Compute weighted sum of its inputs ***/
		double sum = 0.0;
		for (int k = 0; k <= n1; k++) {
			sum += conn[k][j] * l1[k];
		}
		l2[j] = squash(sum);
	}

}

// o(1-o)(t-o)
void bpnn_output_error(double *delta, double *target, double *output, int nj, double *err)
{
	double errsum = 0.0;
	for (int j = 1; j <= nj; j++) {
		double o = output[j];
		double t = target[j];
		delta[j] = o * (1.0 - o) * (t - o);
		errsum += abs(delta[j]);
	}
	*err = errsum;
}

void bpnn_hidden_error(double *delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err)
{
	double errsum = 0.0;
	for (int j = 1; j <= nh; j++) {
		double h = hidden[j];
		double sum = 0.0;
		for (int k = 1; k <= no; k++) {
			sum += delta_o[k] * who[j][k];
		}
		delta_h[j] = h * (1.0 - h) * sum;
		errsum += abs(delta_h[j]);
	}
	*err = errsum;
}

void bpnn_adjust_weights(double *delta, int ndelta, double *ly, int nly, double **w, double **oldw, double eta, double momentum)
{
	ly[0] = 1.0;
	for (int j = 1; j <= ndelta; j++) {
		for (int k = 0; k <= nly; k++) {
			double new_dw = ((eta * delta[j] * ly[k]) + (momentum * oldw[k][j]));
			w[k][j] += new_dw;
			oldw[k][j] = new_dw;
		}
	}
}



void BPNeuralNetwork::Initialize(unsigned int seed)
{
	printf("Random number generator seed: %d\n", seed);
	srand(seed);
}

BPNeuralNetwork::BPNeuralNetwork(int n_in, int n_hidden, int n_out)
{
	this->input_n = n_in;
	this->hidden_n = n_hidden;
	this->output_n = n_out;

	this->input_units = new double[n_in + 1];
	this->hidden_units = new double[n_hidden + 1];
	this->output_units = new double[n_out + 1];

	this->hidden_delta = new double[n_hidden + 1];
	this->output_delta = new double[n_out + 1];
	this->target = new double[n_out + 1];

	this->input_weights = Create2DArray(n_in + 1, n_hidden + 1);
	this->hidden_weights = Create2DArray(n_hidden + 1, n_out + 1);

	this->input_prev_weights = Create2DArray(n_in + 1, n_hidden + 1);
	this->hidden_prev_weights = Create2DArray(n_hidden + 1, n_out + 1);
}

void BPNeuralNetwork::FeedForward()
{
	int nIn = this->input_n;
	int nHid = this->hidden_n;
	int nOut = this->output_n;

	/*** Feed forward input activations. ***/
	bpnn_layerforward(this->input_units, this->hidden_units, this->input_weights, nIn, nHid);
	bpnn_layerforward(this->hidden_units, this->output_units, this->hidden_weights, nHid, nOut);

}

void BPNeuralNetwork::Train(double eta, double momentum, double *eo, double *eh)
{
	int nIn = this->input_n;
	int nHid = this->hidden_n;
	int nOut = this->output_n;

	// 0.Feed forward input activations.
	bpnn_layerforward(this->input_units, this->hidden_units, this->input_weights, nIn, nHid);
	bpnn_layerforward(this->hidden_units, this->output_units, this->hidden_weights, nHid, nOut);

	// 1.Compute error and delta on output and hidden units.
	bpnn_output_error(this->output_delta, this->target, this->output_units, nOut, eo);
	bpnn_hidden_error(this->hidden_delta, nHid, this->output_delta, nOut, this->hidden_weights, this->hidden_units, eh);

	// 2.Adjust input and hidden weights.
	bpnn_adjust_weights(this->output_delta, nOut, this->hidden_units, nHid, this->hidden_weights, this->hidden_prev_weights, eta, momentum);
	bpnn_adjust_weights(this->hidden_delta, nHid, this->input_units, nIn, this->input_weights, this->input_prev_weights, eta, momentum);

}

void BPNeuralNetwork::Save(char *filename)
{
	int n1, n2, n3, i, j, memcnt;
	double dvalue, **w;
	char *mem;
	FILE* fd;

	if ((fd = fopen(filename, "wb")) == NULL) {
		printf("BPNeuralNetwork_SAVE: Cannot create '%s'\n", filename);
		return;
	}

	n1 = this->input_n;  n2 = this->hidden_n;  n3 = this->output_n;
	printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
	fflush(stdout);

	fwrite((char *)&n1, sizeof(int), 1, fd);
	fwrite((char *)&n2, sizeof(int), 1, fd);
	fwrite((char *)&n3, sizeof(int), 1, fd);

	memcnt = 0;
	w = this->input_weights;
	mem = (char *)malloc((unsigned)((n1 + 1) * (n2 + 1) * sizeof(double)));
	for (i = 0; i <= n1; i++) {
		for (j = 0; j <= n2; j++) {
			dvalue = w[i][j];
			fastcopy(&mem[memcnt], &dvalue, sizeof(double));
			memcnt += sizeof(double);
		}
	}
	fwrite(mem, (n1 + 1) * (n2 + 1) * sizeof(double), 1, fd);
	free(mem);

	memcnt = 0;
	w = this->hidden_weights;
	mem = (char *)malloc((unsigned)((n2 + 1) * (n3 + 1) * sizeof(double)));
	for (i = 0; i <= n2; i++) {
		for (j = 0; j <= n3; j++) {
			dvalue = w[i][j];
			fastcopy(&mem[memcnt], &dvalue, sizeof(double));
			memcnt += sizeof(double);
		}
	}
	fwrite(mem, (n2 + 1) * (n3 + 1) * sizeof(double), 1, fd);
	free(mem);

	fclose(fd);
	return;
}

BPNeuralNetwork* BPNeuralNetwork::Create(int n_in, int n_hidden, int n_out) {

	BPNeuralNetwork *newnet = new BPNeuralNetwork(n_in, n_hidden, n_out);

	//#define INITZERO

#ifdef INITZERO
	bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
	bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
	bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
	bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
	bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);

	return (newnet);
}

BPNeuralNetwork* BPNeuralNetwork::Read(char *filename) {
	int n1, n2, n3;
	FILE *fd = fopen(filename, "rb");

	if (fd == NULL) {
		return (NULL);
	}

	printf("Reading '%s'\n", filename); 

	fread((char *)&n1, sizeof(int), 1, fd);
	fread((char *)&n2, sizeof(int), 1, fd);
	fread((char *)&n3, sizeof(int), 1, fd);
	BPNeuralNetwork *newptr = new BPNeuralNetwork(n1, n2, n3);

	printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
	printf("Reading input weights...");  

	int memcnt = 0;
	char *mem = (char *)malloc((unsigned)((n1 + 1) * (n2 + 1) * sizeof(double)));
	fread(mem, (n1 + 1) * (n2 + 1) * sizeof(double), 1, fd);
	for (int i = 0; i <= n1; i++) {
		for (int j = 0; j <= n2; j++) {
			fastcopy(&(newptr->input_weights[i][j]), &mem[memcnt], sizeof(double));
			memcnt += sizeof(double);
		}
	}
	free(mem);

	printf("Done\nReading hidden weights...");  

	memcnt = 0;
	mem = (char *)malloc((unsigned)((n2 + 1) * (n3 + 1) * sizeof(double)));
	fread(mem, (n2 + 1) * (n3 + 1) * sizeof(double), 1, fd);
	for (int i = 0; i <= n2; i++) {
		for (int j = 0; j <= n3; j++) {
			fastcopy(&(newptr->hidden_weights[i][j]), &mem[memcnt], sizeof(double));
			memcnt += sizeof(double);
		}
	}
	free(mem);
	fclose(fd);

	printf("Done\n"); 

	bpnn_zero_weights(newptr->input_prev_weights, n1, n2);
	bpnn_zero_weights(newptr->hidden_prev_weights, n2, n3);

	return (newptr);
}

BPNeuralNetwork::~BPNeuralNetwork()
{
	int n1, n2, i;

	n1 = this->input_n;
	n2 = this->hidden_n;

	free((char *)this->input_units);
	free((char *)this->hidden_units);
	free((char *)this->output_units);

	free((char *)this->hidden_delta);
	free((char *)this->output_delta);
	free((char *)this->target);

	for (i = 0; i <= n1; i++) {
		free((char *)this->input_weights[i]);
		free((char *)this->input_prev_weights[i]);
	}
	free((char *)this->input_weights);
	free((char *)this->input_prev_weights);

	for (i = 0; i <= n2; i++) {
		free((char *)this->hidden_weights[i]);
		free((char *)this->hidden_prev_weights[i]);
	}
	free((char *)this->hidden_weights);
	free((char *)this->hidden_prev_weights);
}

int BPNeuralNetwork::evaluatePerformance(double *err)
{
	*err = 0;
	int nResult = 1;
	for (size_t i = 1; i <= output_n; i++)
	{
		double delta = this->target[i] - this->output_units[i];

		*err  +=(0.5 * delta * delta);
		nResult = nResult && (this->target[i] > 0.5 == this->output_units[i] > 0.5);
	}

	return nResult;
}


/*** Computes the performance of a net on the images in the imagelist. ***/
/*** Prints out the percentage correct on the image set, and the
average error between the target and the output units for the set. ***/
void BPNeuralNetwork::CalculatePerformance(NNImageList *il, int list_errors)
{

	double err = 0.0;			// total error
	int correct = 0;			// number of correct
	int n = il->size();
	if (n <= 0){
		if (!list_errors)
			printf("0.0 0.0 ");
		return;
	}

	// calculate every input image
	for (int i = 0; i < n; i++) {
		// 0.Load the image into the input layer.
		LoadInputImage((*il)[i]);

		// 1.Run the network on this input.
		this->FeedForward();

		/*** Set up the target vector for this image. **/
		this->LoadTarget((*il)[i]);

		/*** See if it got it right. ***/
		//chaged by sjt here.
		//if (evaluate_performance(this, &val, 0)) {

		double val;
		if (this->evaluatePerformance(&val)) {
			correct++;
		}
		else if (list_errors) {
			printf("%s - outputs ", (*il)[i]->_arrName);
			for (int j = 1; j <= this->output_n; j++) {
				printf("%.3f ", this->output_units[j]);
			}
			putchar('\n');
		}
		err += val;
	}

	// calculate average
	err = err / (double)n;

	if (!list_errors)
		/* bthom==================================
		this line prints part of the ouput line
		discussed in section 3.1.2 of homework
		*/
		printf("%g %g ", ((double)correct / (double)n) * 100.0, err);

}

void BPNeuralNetwork::LoadInputImage(NNImage *img)
{
	int nr = img->_nRows;
	int nc = img->_nCols;
	int nt = img->_nTuls;
	int imgsize = nr * nc*nt;
	if (imgsize != this->input_n) {
		printf("LOAD_INPUT_WITH_NNImage: This image has %d pixels,\n", imgsize);
		printf("   but your net has %d input units.  I give up.\n", this->input_n);
		exit(-1);
	}

	double *units = this->input_units;
	int k = 1;
	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < nc; j++) {
			for (size_t t = 0; t < nt; t++)
			{
				units[k] = ((double)img->GetPixel(i, j, t)) / 255.0;
				k++;
			}
		}
	}
}

/*** This is the target output encoding for a network with one output unit.
It scans the image name, and if it's an image of me (js) then
it sets the target unit to HIGH; otherwise it sets it to LOW.
Remember, units are indexed starting at 1, so target unit 1
is the one to change....  ***/
void BPNeuralNetwork::LoadTarget(NNImage *img)
{
	int scale;
	char userid[40], head[40], expression[40], eyes[40], photo[40];

	userid[0] = head[0] = expression[0] = eyes[0] = photo[0] = '\0';

	/*** scan in the image features ***/
	sscanf(img->_arrName, "%[^_]_%[^_]_%[^_]_%[^_]_%d.%[^_]",
		userid, head, expression, eyes, &scale, photo);

	char *p = strrchr(eyes, '.');
	if (p != NULL)
	{
		userid[0] = head[0] = expression[0] = eyes[0] = photo[0] = '\0';

		sscanf(img->_arrName, "%[^_]_%[^_]_%[^_]_%[^.].%[^_]", userid, head, expression, eyes, photo);
		scale = 1;
	}

	p = strrchr(userid, '\\');
	if (p != NULL)
		p++;

	switch (g_targetProperty)
	{
	case P_User:
		if (!strcmp(p, "an2i")) {
			this->target[1] = TARGET_HIGH;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "at33")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_HIGH;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "boland")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_HIGH;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "bpm")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_HIGH;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "ch4f")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_HIGH;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "cheyer")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_HIGH;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "choon")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_HIGH;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "danieln")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_HIGH;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "glickman")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_HIGH;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "karyadi")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_HIGH;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "kawamura")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_HIGH;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "kk49")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_HIGH;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "megak")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_HIGH;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "mitchell")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_HIGH;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "night")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_HIGH;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "phoebe")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_HIGH;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "saavik")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_HIGH;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "steffi")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_HIGH;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "sz24")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_HIGH;
			this->target[20] = TARGET_LOW;
		}
		else if (!strcmp(p, "tammo")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
			this->target[5] = TARGET_LOW;
			this->target[6] = TARGET_LOW;
			this->target[7] = TARGET_LOW;
			this->target[8] = TARGET_LOW;
			this->target[9] = TARGET_LOW;
			this->target[10] = TARGET_LOW;
			this->target[11] = TARGET_LOW;
			this->target[12] = TARGET_LOW;
			this->target[13] = TARGET_LOW;
			this->target[14] = TARGET_LOW;
			this->target[15] = TARGET_LOW;
			this->target[16] = TARGET_LOW;
			this->target[17] = TARGET_LOW;
			this->target[18] = TARGET_LOW;
			this->target[19] = TARGET_LOW;
			this->target[20] = TARGET_HIGH;
		}
		else {
			std::cout << p << std::endl;
			std::cout << "error" << std::endl;
			exit(0);
		}
		break;
	case P_Head:
		if (!strcmp(head, "up")) {
			this->target[1] = TARGET_HIGH;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
		}
		else if (!strcmp(expression, "left")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_HIGH;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
		}
		else if (!strcmp(expression, "straight")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_HIGH;
			this->target[4] = TARGET_LOW;
		}
		else {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_HIGH;
		}
		break;
	case P_Expression:
		if (!strcmp(expression, "angry")) {
			this->target[1] = TARGET_HIGH;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
		}
		else if (!strcmp(expression, "happy")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_HIGH;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_LOW;
		}
		else if (!strcmp(expression, "neutral")) {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_HIGH;
			this->target[4] = TARGET_LOW;
		}
		else {
			this->target[1] = TARGET_LOW;  /* it's me, set target to HIGH */
			this->target[2] = TARGET_LOW;
			this->target[3] = TARGET_LOW;
			this->target[4] = TARGET_HIGH;
		}
		break;
	case P_Eye:
		if (!strcmp(eyes, "sunglasses")) {
			this->target[1] = TARGET_HIGH;  /* it's me, set target to HIGH */
		}
		else {
			this->target[1] = TARGET_LOW;   /* not me, set it to LOW */
		}
		break;
		
	default:
		break;
	}

	//if (!strcmp(p, "glickman")) {


	//   std::cout<<p<<std::endl;

	


	//if (!strcmp(p, "glickman")) {




}

void BPNeuralNetwork::LoadTargetNew(NNImage *img)
{
	for (size_t i = 0; i < g_nClass; i++)
	{
		this->target[i+1] = TARGET_LOW;
	}
	this->target[img->_nLabel+1] = TARGET_HIGH;

}

void BPNeuralNetwork::CalculatePerformanceNew(NNImageList *il, int list_errors)
{
	double err = 0.0;			// total error
	int correct = 0;			// number of correct
	int n = il->size();			// number of images
	if (n <= 0) {
		if (!list_errors)
			printf("0.0 0.0 ");
		return;
	}

	// calculate every input image
	for (int i = 0; i < n; i++) {
		// 0.Load the image into the input layer.
		LoadInputImage((*il)[i]);

		// 1.Run the network on this input.
		FeedForward();

		// 2.Set up the target vector for this image.
		LoadTargetNew((*il)[i]);

		// 3.check the result and loss function
		double val;
		if (evaluatePerformance(&val)) {
			correct++;
		}
		else if (list_errors) {
			printf("%s - outputs ", (*il)[i]->_arrName);
			for (int j = 1; j <= this->output_n; j++) {
				printf("%.3f ", this->output_units[j]);
			}
			putchar('\n');
		}
		err += val;
	}

	// calculate average
	err = err / (double)n;

	if (!list_errors)
		/* bthom==================================
		this line prints part of the ouput line
		discussed in section 3.1.2 of homework
		*/
		printf("%g %g ", ((double)correct / (double)n) * 100.0, err);

}
