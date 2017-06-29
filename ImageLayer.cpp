#include "ImageLayer.h"

#include <QGLWidget>
#include <gl/GLU.h>


#include <QFile>

#include <iostream>

using namespace std;

const int g_nFocusedIndex = 1;

ImageLayer::ImageLayer() :_dataTexture(0)
, _dataTextureMean(0)
{
	// 0.read data
	const char* _strFile1 = "..\\cifar-10-batches-bin\\data_batch_2.bin";
	const char* _strFile0 = "..\\cifar-10-batches-bin\\test_batch.bin";
	readData(1, _strFile1);
	readData(0, _strFile0);

	// 1.statistic
	statistic();

	// 2.test
	testMean();

	// 3.generate texture
	int nImgRow = 100;
	int nImgCol = 100;
	int nRow = 32;
	int nCol = 32;
	_dataTexture = new GLubyte[4 * nRow*nCol* nImgRow*nImgCol];
	for (size_t ii = 0; ii < nImgRow; ii++)
	{
		for (size_t jj = 0; jj < nImgCol; jj++)
		{
			for (size_t i = 0; i < nRow; i++)
			{
				for (size_t j = 0; j < nCol; j++)
				{					
					int nReversedI = nRow - 1 - i;
					int nImgIndex = ii * nImgCol + jj;
					int nPixelIndex = (ii*nRow + i) * nImgCol * nCol + jj * nCol + j;
					int nPixelInImg = i * nCol + j;
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

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nImgCol*nCol, nImgRow*nRow, 0, GL_RGBA, GL_UNSIGNED_BYTE, _dataTexture);


	// generate texture for means
	_dataTextureMean = new GLubyte[4 * g_nRow*g_nCol* g_nClass];

	for (size_t l = 0; l < g_nClass; l++)
	{
		for (size_t i = 0; i < nRow; i++)
		{
			for (size_t j = 0; j < nCol; j++)
			{
				int nReversedI = nRow - 1 - i;
				int nPixelIndex = i * nCol*g_nClass + l * nCol + j;
				int nPixelInImg = i * nCol + j;
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
	for (size_t i = 0; i < 10000; i++)
	{
		_arrLabels[nIndex][i] = *arrData++;
		for (size_t k = 0; k < 3; k++)
		{
			for (size_t j = 0; j < 1024; j++)
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

void ImageLayer::testMean() {
	int nRight = 0;
	// test each image
	for (size_t i = 0; i < g_nImgs; i++)
	{
		if (classify(i)==_arrLabels[0][i])
		{
			nRight++;
		}
	}
	cout << "the right rate is: " << nRight / (double)g_nImgs << endl;
}
int ImageLayer::classify(int nIndex) {
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