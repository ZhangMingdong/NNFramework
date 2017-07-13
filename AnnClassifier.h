#pragma once
#include"IMyClassifier.h"
/*
	Artificial Neural Network classifier
*/
class AnnClassifier:public IMyClassifier
{
public:
	AnnClassifier(int nHidden);
	virtual ~AnnClassifier();	
private:
	// W1 
	MyMatrix* _pW1;
	// b1
	MyMatrix* _pB1;
	// W2 
	MyMatrix* _pW2;
	// b2
	MyMatrix* _pB2;
	// hidden layer
	MyMatrix* _pHidden;
	// number of hidden layers
	int _nHidden = 100;
protected:
	// initialized the classifer
	void initializeImp();
public:
	// training
	void Train(const MyMatrix* pInput, const int* pLabel);
	// calculate the label of point
	int CalcLabel(const double* X);
private:
	// an epoch of the training
	void trainStep(double dbStepSize, double dbReg);
	// evaluate scores of all the instance
	void evaluateScore(MyMatrix* pScores);
};

