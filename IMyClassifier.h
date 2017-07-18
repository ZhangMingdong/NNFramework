#pragma once

#include "MyMatrix.h"
/*
	Interface of my classifiers
	Mingdong
	2017/07/13
*/
class IMyClassifier
{
public:
	IMyClassifier();
	virtual ~IMyClassifier();
public:
	// training
	virtual void Train(const MyMatrix* pInput, const int* pLabel)=0;
	// calculate the label of point
	virtual int CalcLabel(const double* X)=0;
public:
	// test on the test dataset
	double Test(const MyMatrix* pTestData, const int* pTestLabel);
protected:
	// input
	const MyMatrix* _pI;
	// output
	const int* _arrLabel;
	// number of classes
	int _nClass;
	// dimension
	int _nD;
	// number of training points
	int _nPoints;

protected:
	// initialized the classifer
	void initialize(int nPoints, int nD, int nClass);
protected:
	// initialization of the implementation classes
	virtual void initializeImp() = 0;
public:
	enum Enum_Classifier
	{
		SoftMax,
		Ann
	};
	// create a classifier
	static IMyClassifier* CreateClassifier(Enum_Classifier type, int nPoints, int nD, int nClass,int nHiddden=0);
};

