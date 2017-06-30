#include "ImageLayer.h"

#include <QGLWidget>
#include <gl/GLU.h>


#include <QFile>

#include <iostream>

using namespace std;

const int g_nFocusedIndex = 1;

ImageLayer::ImageLayer() :_dataTexture(0)
, _dataTextureMean(0)
, _arrW(0)
, _dbDelta(1.0)
, _dbLambda(1.0)
{
	// 0.read data
	const char* _strFile1 = "..\\cifar-10-batches-bin\\data_batch_2.bin";
	const char* _strFile0 = "..\\cifar-10-batches-bin\\test_batch.bin";
	readData(1, _strFile1);
	readData(0, _strFile0);

	// 1.statistic
	statistic();

	// initialize W
	initW();

	// calculate the loss
	calculateLoss();

	// test result
	test(1);

	// 2.test
//	test(0,TM_Mean);

	// generate texture
	generateTexture();
}

ImageLayer::~ImageLayer()
{
	if (_dataTexture)
	{
		delete[]_dataTexture;
	}
	if (_dataTextureMean)
	{
		delete[]_dataTextureMean;
	}
	if (_arrW) delete[] _arrW;
}

void ImageLayer::Draw() {
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	// draw raw image matrix
	glBindTexture(GL_TEXTURE_2D, texID[0]);

	glBegin(GL_QUADS);
	float _fLeft = -2;
	float _fRight = 2;
	float _fBottom = -2;
	float _fTop = 2;
	glTexCoord2f(0.0f, 0.0f); glVertex2f(_fLeft, _fBottom);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(_fRight, _fBottom);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(_fRight, _fTop);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(_fLeft, _fTop);

	glEnd();

	// draw mean
	glBindTexture(GL_TEXTURE_2D, texID[1]);

	glBegin(GL_QUADS);
	_fLeft = -2;
	_fRight = 2;
	_fBottom = -2.5;
	_fTop = -2.1;
	glTexCoord2f(0.0f, 0.0f); glVertex2f(_fLeft, _fBottom);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(_fRight, _fBottom);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(_fRight, _fTop);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(_fLeft, _fTop);

	glEnd();


	glDisable(GL_TEXTURE_2D);
}

void ImageLayer::readData(int nIndex, const char* strFile) {
	
	// length of the data is 30730000
	// which is (1+1024*3)*10000
	QFile file(strFile);
	if (!file.open(QIODevice::ReadOnly)) {
		return;
	}

	char* temp = new char[file.size()];
	file.read(temp, file.size());

	unsigned char* arrData=(unsigned char*)temp;
	int x = file.size();
	for (size_t i = 0; i < g_nImgs; i++)
	{
		_arrLabels[nIndex][i] = *arrData++;
		for (size_t k = 0; k < 3; k++)
		{
			for (size_t j = 0; j < g_nPixels; j++)
			{
				_arrPixels[nIndex][i][j][k]= *arrData++;
			}
		}
	}
}

void ImageLayer::statistic() {
	// count imgs of each class
	int arrCount[g_nClass];
	for (size_t i = 0; i < g_nClass; i++) arrCount[i] = 0;

	for (size_t i = 0; i < g_nClass; i++)
	{
		for (size_t j = 0; j < g_nPixels; j++)
		{
			for (size_t k = 0; k < 3;k++)
				_arrMean[i][j][k] = 0;
		}
	}

	for (size_t i = 0; i < g_nImgs; i++)
	{
		int nClassIndex = _arrLabels[1][i];
		arrCount[nClassIndex]++;

		for (size_t j = 0; j < g_nPixels; j++)
		{
			for (size_t k = 0; k < 3; k++) {
				_arrMean[nClassIndex][j][k] += _arrPixels[1][i][j][k];
			}
		}
	}

	// normalization
	for (size_t i = 0; i < g_nClass; i++)
	{
		for (size_t j = 0; j < g_nPixels; j++)
		{
			for (size_t k = 0; k < 3; k++) {
				_arrMean[i][j][k] /= arrCount[i];
			}
		}
	}
}

void ImageLayer::test(int nTestIndex, EnumTestMode mode) {
	int nRight = 0;
	// test each image
	switch (mode)	
	{
	case TM_Mean:
		for (size_t i = 0; i < g_nImgs; i++)
		{
			if (classifyByMean(i) == _arrLabels[nTestIndex][i])
			{
				nRight++;
			}
		}
		break;
	case TM_W:
		for (size_t i = 0; i < g_nImgs; i++)
		{
			if (classifyByW(i) == _arrLabels[nTestIndex][i])
			{
				nRight++;
			}
		}
		break;
	default:
		break;
	}
//	cout << "the right rate is: " << nRight / (double)g_nImgs << endl;
	cout << nRight / (double)g_nImgs << endl;
}

int ImageLayer::classifyByMean(int nIndex) {
	double arrBias[g_nClass];
	for (size_t i = 0; i < g_nClass; i++) arrBias[i] = 0;
	bool bL2Distance = true;
	if (bL2Distance)
	{
		for (size_t i = 0; i < g_nClass; i++)
		{
			for (size_t j = 0; j < g_nPixels; j++)
			{
				for (size_t k = 0; k < 3; k++)
				{
					arrBias[i] += pow(_arrPixels[0][nIndex][j][k] - _arrMean[i][j][k],2);
				}
			}
			arrBias[i] = sqrt(arrBias[i]);
		}

	}
	else {
		for (size_t i = 0; i < g_nClass; i++)
		{
			for (size_t j = 0; j < g_nPixels; j++)
			{
				for (size_t k = 0; k < 3; k++)
				{
					arrBias[i] += abs(_arrPixels[0][nIndex][j][k] - _arrMean[i][j][k]);
				}
			}
		}
	}
	// find the least bias
	int nLeastIndex = 0;
	double dbLeast = arrBias[0];
	for (size_t i = 1; i < g_nClass; i++)
	{
		if (arrBias[i] < dbLeast) {
			dbLeast = arrBias[i];
			nLeastIndex = i;
		}
	}
	return nLeastIndex;
}

void ImageLayer::generateTexture() {
	// 3.generate texture
	_dataTexture = new GLubyte[4 * g_nRow*g_nCol* g_nImgRow*g_nImgCol];
	for (size_t ii = 0; ii < g_nImgRow; ii++)
	{
		for (size_t jj = 0; jj < g_nImgCol; jj++)
		{
			for (size_t i = 0; i < g_nRow; i++)
			{
				for (size_t j = 0; j < g_nCol; j++)
				{
					int nReversedI = g_nRow - 1 - i;
					int nImgIndex = ii * g_nImgCol + jj;
					int nPixelIndex = (ii*g_nRow + i) * g_nImgCol * g_nCol + jj * g_nCol + j;
					int nPixelInImg = i * g_nCol + j;
					_dataTexture[4 * nPixelIndex + 0] = _arrPixels[g_nFocusedIndex][nImgIndex][nReversedI * 32 + j][0];
					_dataTexture[4 * nPixelIndex + 1] = _arrPixels[g_nFocusedIndex][nImgIndex][nReversedI * 32 + j][1];
					_dataTexture[4 * nPixelIndex + 2] = _arrPixels[g_nFocusedIndex][nImgIndex][nReversedI * 32 + j][2];
					_dataTexture[4 * nPixelIndex + 3] = (GLubyte)255;
				}
			}
		}
	}


	glGenTextures(1, &texID[0]);
	glBindTexture(GL_TEXTURE_2D, texID[0]);
	// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_nImgCol*g_nCol, g_nImgRow*g_nRow, 0, GL_RGBA, GL_UNSIGNED_BYTE, _dataTexture);


	// generate texture for means
	_dataTextureMean = new GLubyte[4 * g_nRow*g_nCol* g_nClass];

	for (size_t l = 0; l < g_nClass; l++)
	{
		for (size_t i = 0; i < g_nRow; i++)
		{
			for (size_t j = 0; j < g_nCol; j++)
			{
				int nReversedI = g_nRow - 1 - i;
				int nPixelIndex = i * g_nCol*g_nClass + l * g_nCol + j;
				int nPixelInImg = i * g_nCol + j;
				_dataTextureMean[4 * nPixelIndex + 0] = _arrMean[l][nReversedI * 32 + j][0];
				_dataTextureMean[4 * nPixelIndex + 1] = _arrMean[l][nReversedI * 32 + j][1];
				_dataTextureMean[4 * nPixelIndex + 2] = _arrMean[l][nReversedI * 32 + j][2];
				_dataTextureMean[4 * nPixelIndex + 3] = (GLubyte)255;
			}
		}
	}


	glGenTextures(1, &texID[1]);
	glBindTexture(GL_TEXTURE_2D, texID[1]);
	// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_nClass*g_nCol, g_nRow, 0, GL_RGBA, GL_UNSIGNED_BYTE, _dataTextureMean);


}


void ImageLayer::initW() {
	int nLen = g_nClass*(g_nPixels * 3 + 1);
	_arrW = new double[nLen];
	for (size_t i = 0; i < nLen; i++)
	{
		_arrW[i] = 1;
		_arrW[i] = rand()/(double)RAND_MAX;
	}
}


double ImageLayer::calculateLoss() {
	double dbLossRegularization = 0;

	// loss of the parameter
	int nLen = g_nClass*(g_nPixels * 3 + 1);
	for (size_t i = 0; i < nLen; i++)
	{
		dbLossRegularization += _arrW[i]*_arrW[i];
	}
	dbLossRegularization = sqrt(dbLossRegularization);

	double dbLossData = 0;
	for (size_t i = 0; i < g_nImgs; i++)
	{
		dbLossData += calculateDataLoss(i);
	}
	dbLossData /= g_nImgs;

	double dbLoss = dbLossRegularization + dbLossData;
//	cout << "the regularization loss is: " << dbLossRegularization << endl;
//	cout << "the data loss is: " << dbLossData << endl;
//	cout << "the loss is: " << dbLoss << endl;

	cout << dbLoss << "\t";
	return dbLoss;
}


double ImageLayer::calculateDataLoss(int nIndex) {
	int nLabel = _arrLabels[g_nFocusedIndex][nIndex];
//	cout << "label:" << nLabel << endl;
	double dbLoss = 0;
	double arrScore[g_nClass];
	calculateScore(nIndex, arrScore);
//	for (size_t i = 0; i < g_nClass; i++)
//	{
//		cout << arrScore[i] << endl;
//	}

	for (size_t i = 0; i < g_nClass; i++)
	{
//		cout << endl;
		if (i == nLabel) continue;
		double dbBias = arrScore[i] - arrScore[nLabel] + _dbDelta;
		if (dbBias > 0) dbLoss += dbBias;
//		cout << dbBias;
	}
//	cout << endl;

	return dbLoss;
}


void ImageLayer::calculateScore(int nIndex, double* arrScore) {
	double* pW = _arrW;
	for (size_t i = 0; i < g_nClass; i++)
	{
		arrScore[i] = 0;
		for (size_t j = 0; j < g_nPixels; j++)
		{
			for (size_t k = 0; k < 3; k++)
			{
				arrScore[i] += (_arrPixels[g_nFocusedIndex][nIndex][j][k] * pW[j * 3 + k]);
			}
			arrScore[i] += pW[g_nPixels*3];			// b_i
		}
		pW += (g_nPixels * 3 + 1);
	}
}


int ImageLayer::classifyByW(int nIndex) {
	double arrScore[g_nClass];
	calculateScore(nIndex, arrScore);
	int nLabel = 0;
	double dbMaxScore = arrScore[0];
	for (size_t i = 1; i < g_nClass; i++)
	{
		if (arrScore[i] > dbMaxScore) {
			dbMaxScore = arrScore[i];
			nLabel = i;
		}
	}
//	cout << "classify->label:" << nLabel << endl;
//	for (size_t i = 0; i < g_nClass; i++)
//	{
//		cout << arrScore[i] << endl;
//	}
	return nLabel;	
}










