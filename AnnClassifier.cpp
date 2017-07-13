#include "AnnClassifier.h"
#include <math.h>
#include <iostream>

using namespace std;


AnnClassifier::AnnClassifier(int nHidden):_pW1(0)
,_pB1(0)
,_pW2(0)
,_pB2(0)
,_pHidden(0)
,_nHidden(nHidden)
{

}


void AnnClassifier::initializeImp() {
	_pW1 = new MyMatrix(_nD, _nHidden);
	_pB1 = new MyMatrix(1, _nHidden);
	_pW2 = new MyMatrix(_nHidden, _nClass);
	_pB2 = new MyMatrix(1, _nClass);
	_pHidden = new MyMatrix(_nPoints, _nHidden);

	_pB1->InitZero();
	_pW1->InitRandom(0.01);
	_pB2->InitZero();
	_pW2->InitRandom(0.01);
}

AnnClassifier::~AnnClassifier()
{
	if (_pW1) delete _pW1;
	if (_pB1) delete _pB1;
	if (_pW2) delete _pW2;
	if (_pB2) delete _pB2;
	if (_pHidden) delete _pHidden;
}

void AnnClassifier::Train(const MyMatrix* pInput, const int* pLabel) {
	_pI = pInput;
	_arrLabel = pLabel;
	// train
	double dbStepSize = 1.0;
	double dbReg = 0.001;
	// set step to 1 to show the training procedure
	int nEpochs = 10000;
	for (size_t i = 0; i < nEpochs; i++)
	{
		trainStep(dbStepSize, dbReg);
	}
}

void AnnClassifier::trainStep(double dbStepSize, double dbReg) {
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
		double dbCorrectLogProb = -log(prob.GetValue(i, _arrLabel[i]));// -log(arrProb[i][_arrLabel[i]]);
		dbDataLoss += dbCorrectLogProb;
	}
	dbDataLoss /= _nPoints;
	double dbRegLoss = 0.5*dbReg*(_pW1->Norm2()+_pW2->Norm2());
	double dbLoss = dbRegLoss + dbDataLoss;
	cout << "Loss:\t" << dbLoss << "\t";

	// 4.compute the  gradient on scores
	MyMatrix dScore(&prob);
	for (size_t j = 0; j < _nPoints; j++)
	{
		dScore.SetValue(j, _arrLabel[j], (prob.GetValue(j, _arrLabel[j]) - 1));
	}
	dScore.Multi(1.0 / _nPoints);

	// 5.backpropagate the gradient to the parameters
	// first backprop into parameters w2 and b2
	MyMatrix hiddenT(_pHidden, true);
	MyMatrix dW2(&hiddenT, &dScore);
	MyMatrix dB2;
	dScore.Sum(&dB2, 0);

	// next backprop into hidden layer
	MyMatrix W2T(_pW2, true);
	MyMatrix dHidden(&dScore, &W2T);
	// backprop the ReLU non-linearity
	for (size_t i = 0; i < _nPoints; i++)
	{
		for (size_t j = 0; j < _nHidden; j++)
		{
			if (_pHidden->GetValue(i,j)<=0)
			{
				dHidden.SetValue(i, j, 0);
			}
		}
	}

	MyMatrix inputT(_pI, true);
	MyMatrix dW1(&inputT, &dHidden);
	MyMatrix dB1;
	dHidden.Sum(&dB1, 0);

	// 6.regularization gradient
	for (size_t j = 0; j < _nHidden; j++) {
		for (size_t k = 0; k < _nClass; k++) {
			dW2.SetValue(j, k, dW2.GetValue(j, k) + dbReg*_pW2->GetValue(j, k));
		}
	}	
	for (size_t j = 0; j < _nD; j++) {
		for (size_t k = 0; k < _nHidden; k++) {
			dW1.SetValue(j, k, dW1.GetValue(j, k) + dbReg*_pW1->GetValue(j, k));
		}
	}
	// 7.perform parameter update
	_pW1->Linear(&dW1, -dbStepSize);
	_pB1->Linear(&dB1, -dbStepSize);
	_pW2->Linear(&dW2, -dbStepSize);
	_pB2->Linear(&dB2, -dbStepSize);

	// 8.calculate score and accuracy
	int nPredicted = 0;
	for (size_t i = 0; i < _nPoints; i++)
	{
		if (CalcLabel(_pI->GetRow(i)) == _arrLabel[i]) nPredicted++;
	}
	cout << "Accuracy:\t" << nPredicted / (double)(_nPoints) << endl;;
}


int AnnClassifier::CalcLabel(const double* X) {
	MyMatrix score(1, _nClass);
	MyMatrix input(1, _nD);
	for (size_t i = 0; i < _nD; i++)
	{
		input.SetValue(0, i, X[i]);
	}
	MyMatrix hidden(1, _nHidden);
	hidden.Formula(&input, _pW1, _pB1);
	hidden.TrimNegative();
	score.Formula(&hidden, _pW2, _pB2);

	int nResult = 0;
	double dbMaxScore = score.GetValue(0, 0);
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

void AnnClassifier::evaluateScore(MyMatrix* pScores) {
	_pHidden->Formula(_pI, _pW1, _pB1);
	_pHidden->TrimNegative();
	pScores->Formula(_pHidden, _pW2, _pB2);
}