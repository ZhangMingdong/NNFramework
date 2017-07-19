#include "IMyClassifier.h"
#include "SoftMaxClassifier.h"
#include "AnnClassifier.h"

#include<iostream>

using namespace std;

IMyClassifier::IMyClassifier() :_nPoints(0)
, _nD(0)
, _nClass(0)
{
}

void IMyClassifier::initialize(int nPoints, int nD, int nClass) {
	_nPoints = nPoints;
	_nD = nD;
	_nClass = nClass;
	initializeImp();
}

IMyClassifier::~IMyClassifier()
{
}

IMyClassifier* IMyClassifier::CreateClassifier(Enum_Classifier type, int nPoints, int nD, int nClass, int nHiddden) {
	IMyClassifier* pClassifier = 0;
	switch (type)
	{
	case IMyClassifier::SoftMax:
		pClassifier = new SoftMaxClassifier();
		break;
	case IMyClassifier::Ann:
		pClassifier = new AnnClassifier(nHiddden);
		break;
	default:
		break;
	}
	if(pClassifier)
		pClassifier->initialize(nPoints, nD, nClass);
	return pClassifier;
}


// test on the test dataset
double IMyClassifier::Test(const MyMatrix* pTestData, const int* pTestLabel) {
	int nPredicted = 0;
	int nPoints = pTestData->Rows();
	for (size_t i = 0; i < nPoints; i++)
	{
//		cout << CalcLabel(pTestData->GetRow(i)) << "\t" << pTestLabel[i] << endl;
		if (CalcLabel(pTestData->GetRow(i)) == pTestLabel[i]) nPredicted++;
	}
	return nPredicted / (double)(nPoints);
}