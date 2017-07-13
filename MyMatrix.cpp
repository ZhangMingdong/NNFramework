#include "MyMatrix.h"
#include <assert.h>
#include <random>


MyMatrix::MyMatrix(int r, int c):_nR(r),_nC(c)
{
	allocateSpace();
}

MyMatrix::MyMatrix(const MyMatrix* m1, const MyMatrix* m2)
{
	_nR = m1->_nR;
	_nC = m2->_nC;
	allocateSpace();
	Multi(m1, m2);
}

MyMatrix::MyMatrix(const MyMatrix* m1, const MyMatrix* m2, const MyMatrix* b) {
	_nR = m1->_nR;
	_nC = m2->_nC;
	allocateSpace();
	Formula(m1, m2, b);
}

MyMatrix::MyMatrix(const MyMatrix* m1, bool bTranspose) {
	if (bTranspose)
	{
		_nR = m1->_nC;
		_nC = m1->_nR;
	}
	else {
		_nR = m1->_nR;
		_nC = m1->_nC;
	}
	allocateSpace();
	for (size_t i = 0; i < _nR; i++)
	{
		for (size_t j = 0; j < _nC; j++)
		{
			if (bTranspose)
			{
				_data[i][j] = m1->GetValue(j, i);
			}
			else {
				_data[i][j] = m1->GetValue(i, j);
			}
		}
	}
}


MyMatrix::MyMatrix(const MyMatrix* m1, double(*f)(double)) {
	_nR = m1->_nR;
	_nC = m1->_nC;
	allocateSpace();
	for (size_t i = 0; i < _nR; i++)
	{
		for (size_t j = 0; j < _nC; j++)
		{
			_data[i][j] = f(m1->_data[i][j]);
		}
	}
}


MyMatrix::MyMatrix():_data(0) {
	
}

MyMatrix::~MyMatrix()
{
	if (_data)
	{
		for (size_t i = 0; i < _nR; i++)
		{
			delete[] _data[i];
		}
		delete[]_data;

	}
}

// get a value
double MyMatrix::GetValue(int i, int j)const {
	return _data[i][j];
}

// set a value
void MyMatrix::SetValue(int i, int j, double v) {
	_data[i][j] = v;
}


void MyMatrix::allocateSpace() {
	_data = new double*[_nR];
	for (size_t i = 0; i < _nR; i++)
	{
		_data[i] = new double[_nC];
	}
}

void MyMatrix::TrimNegative() {
	for (size_t i = 0; i < _nR; i++)
	{
		for (size_t j = 0; j < _nC; j++)
		{
			if (_data[i][j] < 0) _data[i][j] = 0;
		}
	}
}

void MyMatrix::Multi(double dbFactor) {
	for (size_t i = 0; i < _nR; i++)
	{
		for (size_t j = 0; j < _nC; j++)
		{
			_data[i][j] *= dbFactor;
		}
	}
}

double MyMatrix::Norm2() {
	double dbNorm = 0;
	for (size_t i = 0; i < _nR; i++)
	{
		for (size_t j = 0; j < _nC; j++) {
			dbNorm += _data[i][j] * _data[i][j];
		}
	}
	return dbNorm;
}

void MyMatrix::Multi(const MyMatrix* m1, const MyMatrix* m2) {
	if (m1->_nC == m2->_nR)
	{
		for (size_t i = 0; i < _nR; i++)
		{
			for (size_t j = 0; j < _nC; j++)
			{
				_data[i][j] = 0;
				for (size_t k = 0; k < m1->_nC; k++) {
					_data[i][j] += m1->GetValue(i, k)*m2->GetValue(k, j);
				}
			}
		}
	}
}

void MyMatrix::Formula(const MyMatrix* m1, const MyMatrix* m2, const MyMatrix* b) {
	if (m1->_nC == m2->_nR&&m2->_nC == b->_nC)
	{
		for (size_t i = 0; i < _nR; i++)
		{
			for (size_t j = 0; j < _nC; j++)
			{
				_data[i][j] = b->GetValue(0, j);
				for (size_t k = 0; k < m1->_nC; k++) {
					_data[i][j] += m1->GetValue(i, k)*m2->GetValue(k, j);
				}
			}
		}
	}
}

void MyMatrix::Linear(const MyMatrix* m1, double dbScale) {
	assert(m1->_nR == _nR&&m1->_nC == _nC);
	for (size_t i = 0; i < _nR; i++)
	{
		for (size_t j = 0; j < _nC; j++) {
			_data[i][j] += m1->_data[i][j] * dbScale;
		}
	}
}

// initialize value randomly
void MyMatrix::InitRandom(double dbScale) {
	for (size_t i = 0; i < _nR; i++)
	{
		for (size_t j = 0; j < _nC; j++) {
			_data[i][j] = dbScale*rand() / (double)RAND_MAX;
		}
	}
}

// initialize value to 0
void MyMatrix::InitZero() {
	for (size_t i = 0; i < _nR; i++)
	{
		for (size_t j = 0; j < _nC; j++) {
			_data[i][j] = 0;
		}
	}

}

const double* MyMatrix::GetRow(int r) const {
	return _data[r];
}

void MyMatrix::ApplyFun(double(*f)(double)) {
	for (size_t i = 0; i < _nR; i++)
	{
		for (size_t j = 0; j < _nC; j++) {
			_data[i][j] = f(_data[i][j]);
		}
	}
}

void MyMatrix::Sum(MyMatrix* pResult, int nAxis) {
	if (nAxis==0)
	{
		pResult->_nR = 1;
		pResult->_nC = _nC;
	}
	else if (nAxis == 1) {
		pResult->_nR = _nR;
		pResult->_nC = 1;
	}
	pResult->allocateSpace();
	if (nAxis == 0)
	{
		for (size_t i = 0; i < _nC; i++)
		{
			double dbSum = 0;
			for (size_t j = 0; j < _nR; j++)
			{
				dbSum += _data[j][i];
			}
			pResult->_data[0][i] = dbSum;
		}
	}
	else if (nAxis == 1) {
		for (size_t i = 0; i < _nR; i++)
		{
			double dbSum = 0;
			for (size_t j = 0; j < _nC; j++)
			{
				dbSum += _data[i][j];
			}
			pResult->_data[i][0] = dbSum;
		}

	}
}

double MyMatrix::Sum(double(*f)(double)) {
	double dbResult = 0;
	for (size_t i = 0; i < _nR; i++)
	{
		for (size_t j = 0; j < _nC; j++)
		{
			if (f) {
				dbResult += f(_data[i][j]);
			}
			else dbResult += _data[i][j];
		}
	}
	return dbResult;
}

void MyMatrix::Div(MyMatrix* pResult, MyMatrix* pDivisor) {
	pResult->_nR = _nR;
	pResult->_nC = _nC;
	pResult->allocateSpace();

	if (pDivisor->_nR == 1 && pDivisor->_nC == _nC)
	{
		for (size_t i = 0; i < _nR; i++)
		{
			for (size_t j = 0; j < _nC; j++)
			{
				pResult->_data[i][j] = _data[i][j] / pDivisor->_data[0][j];
			}
		}
	}
	else if (pDivisor->_nR == _nR && pDivisor->_nC == 1) {
		for (size_t i = 0; i < _nR; i++)
		{
			for (size_t j = 0; j < _nC; j++)
			{
				pResult->_data[i][j] = _data[i][j] / pDivisor->_data[i][0];
			}
		}
	}
	else {
		assert(false);
	}
}