#pragma once
#include"IMyClassifier.h"
/*
	SoftMaxClassifier
*/
class SoftMaxClassifier:public IMyClassifier
{
public:
	SoftMaxClassifier();
	virtual ~SoftMaxClassifier();
private:
	// W 
	MyMatrix* _pW;
	// b
	MyMatrix* _pB;

protected:
	// initialized the classifer
	virtual void initializeImp();
public:
	// training
	virtual void Train(const MyMatrix* pInput, const int* pLabel);
	// calculate the label of point
	virtual int CalcLabel(const double* X);
private:
	// an epoch of the training
	void trainStep(double dbStepSize, double dbReg);
	// evaluate scores of all the instance
	void evaluateScore(MyMatrix* pScores);
	// save the parameters
	void saveParams();
};

