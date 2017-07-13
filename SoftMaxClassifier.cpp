#include "SoftMaxClassifier.h"
#include <math.h>
#include <iostream>

using namespace std;


SoftMaxClassifier::SoftMaxClassifier():_pW(0),_pB(0)
{
}

void SoftMaxClassifier::initializeImp() {

	_pW = new MyMatrix(_nD, _nClass);
	_pB = new MyMatrix(1, _nClass);
	_pB->InitZero();
	_pW->InitRandom(0.01);
}

SoftMaxClassifier::~SoftMaxClassifier()
{
	if (_pW)delete _pW;
	if (_pB)delete _pB;
}

void SoftMaxClassifier::Train(const MyMatrix* pInput, const int* pLabel) {
	_pI = pInput;
	_arrLabel = pLabel;
	// train
	double dbStepSize = 0.01;// 1.0;
	double dbReg = 0.003;
	// set step to 1 to show the training procedure
	int nEpochs = 200;
	for (size_t i = 0; i < nEpochs; i++)
	{
		trainStep(dbStepSize, dbReg);
	}
}

void SoftMaxClassifier::trainStep(double dbStepSize, double dbReg) {
	// 1.evaluate class scores
	MyMatrix scores(_nPoints, _nClass);
	evaluateScore(&scores);

	// 2.compute the class probabilities
	MyMatrix expScores(&scores, exp);
	MyMatrix expScoreSum;
	expScores.Sum(&expScoreSum, 1);
	MyMatrix prob;
	expScores.Div(&prob, &expScoreSum);

	// 3.compute the loss: average cross-entropy loss and regularization
	double dbDataLoss = 0;
	for (size_t i = 0; i < _nPoints; i++)
	{
		double dbCorrectLogProb = -log(prob.GetValue(i,_arrLabel[i]));// -log(arrProb[i][_arrLabel[i]]);
		dbDataLoss += dbCorrectLogProb;
	}
	dbDataLoss /= _nPoints;
	double dbRegLoss = 0.5*dbReg*_pW->Norm2();
	double dbLoss = dbRegLoss + dbDataLoss;
	cout << "Loss:\t" << dbLoss << "\t";

	// 4.compute the  gradient on scores
	MyMatrix dScore(&prob);
	for (size_t j = 0; j < _nPoints; j++)
	{
		dScore.SetValue(j, _arrLabel[j], (prob.GetValue(j, _arrLabel[j]) - 1));
	}
	dScore.Multi(1.0 / _nPoints);
	// 5.backpropagate the gradient to the parameters (W,b)
	MyMatrix inputT(_pI, true);
	MyMatrix dW(&inputT, &dScore);
	MyMatrix dB;
	dScore.Sum(&dB, 0);

	// 6.regularization gradient
	for (size_t j = 0; j < _nD; j++) {
		for (size_t k = 0; k < _nClass; k++) {
			dW.SetValue(j, k, dW.GetValue(j, k) + dbReg*_pW->GetValue(j, k));
		}
	}
	// 7.perform parameter update
	_pW->Linear(&dW, -dbStepSize);
	_pB->Linear(&dB, -dbStepSize);

	// 8.calculate score and accuracy
	int nPredicted = 0;
	for (size_t i = 0; i < _nPoints; i++)
	{
		if (CalcLabel(_pI->GetRow(i)) == _arrLabel[i]) nPredicted++;
	}
	cout << "Accuracy:\t" << nPredicted / (double)(_nPoints) << endl;;
}


int SoftMaxClassifier::CalcLabel(const double* X) {
	MyMatrix score(1, _nClass);
	MyMatrix input(1, _nD);
	for (size_t i = 0; i < _nD; i++)
	{
		input.SetValue(0, i, X[i]);
	}
	score.Formula(&input, _pW, _pB);

	int nResult = 0;
	double dbMaxScore = score.GetValue(0,0);
	for (size_t i = 1; i < _nClass; i++)
	{
		double dbValue = score.GetValue(0, i);
		if (dbValue > dbMaxScore) {
			nResult = i;
			dbMaxScore = dbValue;
		}
	}
	return nResult;
}

void SoftMaxClassifier::evaluateScore(MyMatrix* pScores) {
	pScores->Formula(_pI, _pW, _pB);
}