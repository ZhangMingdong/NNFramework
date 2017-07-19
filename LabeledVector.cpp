#include "LabeledVector.h"

#include <assert.h>

LabeledVector::LabeledVector(int nD):_nD(nD)
{
	_arrData = new double[_nD];
}


LabeledVector::~LabeledVector()
{
	if (_arrData)
		delete _arrData;
}

double LabeledVector::GetData(int nIndex)const {
	assert(nIndex < _nD);
	assert(_arrData);
	return _arrData[nIndex];
}
void LabeledVector::SetData(int nIndex, double dbData) {
	assert(nIndex < _nD);
	assert(_arrData);
	_arrData[nIndex] = dbData;
}