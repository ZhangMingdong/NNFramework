#include "ImageLayer.h"

#include <QGLWidget>
#include <gl/GLU.h>

#include <QFile>

#include <iostream>
#include <fstream>

#include "BPNeuralNetwork.h"
#include "NNImage.h"
#include "IMyClassifier.h"

//#define NOT_USE_ESIST_MODEL

using namespace std;

const int g_nFocusedIndex = 1;
ofstream output("log_20170701.txt");



ImageLayer::ImageLayer() :_dataTexture(0)
, _dataTextureMean(0)
, _arrW(0)
, _dbDelta(1.0)
, _dbLambda(1.0)
, _pBPNN(NULL)
{
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

	if (_pBPNN) delete _pBPNN;
}


void ImageLayer::Initialize() {
	// 0.read data
	const char* _arrFiles[g_nFiles] = {
		"..\\cifar-10-batches-bin\\test_batch.bin"
		,"..\\cifar-10-batches-bin\\data_batch_1.bin"
		,"..\\cifar-10-batches-bin\\data_batch_2.bin"
		,"..\\cifar-10-batches-bin\\data_batch_3.bin"
		,"..\\cifar-10-batches-bin\\data_batch_4.bin"
		,"..\\cifar-10-batches-bin\\data_batch_5.bin"
	};
	for (size_t i = 0; i < g_nFiles; i++)
	{
		readData(i, _arrFiles[i]);
	}


	long t1 = GetTickCount();

//	testMeanClassifier();
	trainNN(g_nFocusedIndex);
//	testNearestNeighbor();
//	testMSVM();
//	testLossFun();
//	testRandomLocalSearch();
//	testSoftMax();
//	testAnn();



	int t = GetTickCount() - t1;
	cout << "Computing time: \t" << t << endl;

	// generate texture
	generateTexture();
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
	for (size_t iteF = 1; iteF < g_nFiles; iteF++)
	{
		for (size_t i = 0; i < g_nImgs; i++)
		{
			int nClassIndex = _arrLabels[1][i];
			arrCount[nClassIndex]++;

			for (size_t j = 0; j < g_nPixels; j++)
			{
				for (size_t k = 0; k < 3; k++) {
					_arrMean[nClassIndex][j][k] += _arrPixels[iteF][i][j][k];
				}
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
			if (classifyByMean(nTestIndex,i) == _arrLabels[nTestIndex][i])
			{
				nRight++;
			}
		}
		break;
	case TM_W:
		for (size_t i = 0; i < g_nImgs; i++)
		{
			if (classifyByW(nTestIndex, i) == _arrLabels[nTestIndex][i])
			{
				nRight++;
			}
		}
		break;
	default:
		break;
	}
//	cout << "the right rate is: " << nRight / (double)g_nImgs << endl;
	output << nRight / (double)g_nImgs << endl;
	cout << nRight / (double)g_nImgs << endl;
}

int ImageLayer::classifyByMean(int nFileIndex, int nImgIndex) {
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
					arrBias[i] += pow(_arrPixels[nFileIndex][nImgIndex][j][k] - _arrMean[i][j][k],2);
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
					arrBias[i] += abs(_arrPixels[nFileIndex][nImgIndex][j][k] - _arrMean[i][j][k]);
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

void ImageLayer::initW(bool bRandom) {
	if (true) delete[] _arrW;

	int nWLen = g_nClass*(g_nPixels * 3 + 1);
	_arrW = new double[nWLen];
	if (bRandom) {
		double dbRange = 10;
		for (size_t i = 0; i < nWLen; i++)
		{
			_arrW[i] = rand() / (double)RAND_MAX*dbRange * 2 - dbRange;
		}
	} 
	else 
	{
		for (size_t i = 0; i < nWLen; i++)
		{
			_arrW[i] = 1;
		}
	}
	

}

double ImageLayer::calculateLoss(int nFileIndex, const double *pW) {
	double dbLossRegularization = 0;

	// loss of the parameter
	int nWLen = g_nClass*(g_nPixels * 3 + 1);
	for (size_t i = 0; i < nWLen; i++)
	{
		dbLossRegularization += pW[i]* pW[i];
	}
	dbLossRegularization = sqrt(dbLossRegularization);

	double dbLossData = 0;
	for (size_t i = 0; i < g_nImgs; i++)
	{
		dbLossData += calculateDataLoss(nFileIndex,i,pW);
	}
	dbLossData /= g_nImgs;

	double dbLoss = dbLossRegularization + dbLossData;
//	cout << "the regularization loss is: " << dbLossRegularization << endl;
//	cout << "the data loss is: " << dbLossData << endl;
//	cout << "the loss is: " << dbLoss << endl;

//	cout << dbLoss << "\t";
	return dbLoss;
}

double ImageLayer::calculateDataLoss(int nFileIndex, int nImgIndex,const double* pW) {
	int nLabel = _arrLabels[nFileIndex][nImgIndex];
//	cout << "label:" << nLabel << endl;
	double dbLoss = 0;
	double arrScore[g_nClass];
	calculateScore(nFileIndex,nImgIndex,pW, arrScore);
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

void ImageLayer::calculateScore(int nFileIndex, int nImgIndex, const double* pW, double* arrScore) {
	const double* pWLine = pW; // W of each line
	// for each line
	for (size_t i = 0; i < g_nClass; i++)
	{
		// initialize the score to 0
		arrScore[i] = 0;
		for (size_t j = 0; j < g_nPixels; j++)
		{
			for (size_t k = 0; k < 3; k++)
			{
				arrScore[i] += (_arrPixels[nFileIndex][nImgIndex][j][k] * pWLine[j * 3 + k]);
			}		
		}
		arrScore[i] += pWLine[g_nPixels*3];		// b_i
		pWLine += (g_nPixels * 3 + 1);
	}
}

int ImageLayer::classifyByW(int nFileIndex, int nImgIndex) {
	double arrScore[g_nClass];
	calculateScore(nFileIndex,nImgIndex,_arrW, arrScore);
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

void ImageLayer::trainRandomLocal(int nFileIndex) {
	double dbStep = 0.001;
	
	int nWLen = g_nClass*(g_nPixels * 3 + 1);
	double* arrTempW = new double[nWLen];
	for (size_t i = 0; i < nWLen; i++)
	{
		arrTempW[i] = _arrW[i];
	}
	double dbCurrentLoss = calculateLoss(nFileIndex,arrTempW); // current loss function value
	int nUpdated = 0;		// times of updated
	int nTried = 0;			// times of trials
	while(nUpdated<20)	// run 100 steps
	{		
		nTried++;
		if (nTried > 10000) break;		// break while tried 10000 times and cannot find a direction
		// generate a direction randomly, and change w a step
		int nDirection = rand() / (double)RAND_MAX*nWLen;
		int nForwardOrBack = rand() / (double)RAND_MAX>0.5 ? 1 : -1;	// move forward or move back
		arrTempW[nDirection] += dbStep*nForwardOrBack;

		// calculate bias of loss function
		double dbNewLoss= calculateLoss(nFileIndex,arrTempW);


		if (dbNewLoss>dbCurrentLoss) {
			arrTempW[nDirection] -= dbStep*nForwardOrBack;
//			cout << "fail" << endl;;
		}
		else {
			nUpdated++;
			dbCurrentLoss = dbNewLoss;
			cout << nTried << "\t";
			nTried = 0;
			output << dbCurrentLoss << endl;
			cout << dbCurrentLoss << endl;
		}
	}

	for (size_t i = 0; i < nWLen; i++)
	{
		_arrW[i] = arrTempW[i];
	}

	delete[] arrTempW;
}

void ImageLayer::trainNN(int nDataSetIndex) {


}


void ImageLayer::testMeanClassifier() {

	// 1.statistic
	statistic();
	// 2.test
	test(0, TM_Mean);
}

void ImageLayer::testNearestNeighbor() {
	int nRight = 0;
	for (size_t i = 0; i < g_nImgs; i++)
	{
		int nLabel = getLabelByNN(_arrPixels[0][i]);
		if (nLabel == _arrLabels[0][i]) nRight++;
		cout << i << "\t" << nRight / (double)(i + 1) << endl;
	}
	cout << nRight / (double)g_nImgs << endl;
}

int calcL1Dis(unsigned char pImg1[g_nPixels][3], unsigned char pImg2[g_nPixels][3]) {
	int nDis = 0;
	for (size_t i = 0; i < g_nPixels; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			nDis += abs(pImg1[i][j] - pImg2[i][j]);
		}
	}
	return nDis;
}

double calcL2Dis(unsigned char pImg1[g_nPixels][3], unsigned char pImg2[g_nPixels][3]) {
	double dbDis = 0;
	for (size_t i = 0; i < g_nPixels; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			double dbDif = pImg1[i][j] - pImg2[i][j];
			dbDis += dbDif*dbDif;
		}
	}
	return sqrt(dbDis);
}

int ImageLayer::getLabelByNN(unsigned char pImg[1024][3]) {
	double dbDisMin = 10000000.0;
	int nLabelMin = -1;
	for (size_t iteF = 1; iteF < g_nFiles; iteF++)
	{
		for (size_t i = 0; i < g_nImgs; i++)
		{
			double dbDis = calcL2Dis(pImg, _arrPixels[iteF][i]);
			if (dbDis < dbDisMin) {
				dbDisMin = dbDis;
				nLabelMin = _arrLabels[iteF][i];
			}
		}

	}
	return nLabelMin;
}


/*
update the parateters
nClass=1/-1;
*/
void ImageLayer::updateW(double* arrData, int nLabel, double dbDelta, double dbLambda) {
	double dbLoss = 0;
	int nParamLen = g_nPixels * 3 + 1;
	// calculate the score for each class about this instance
	double arrScore[g_nClass];
	for (size_t i = 0; i < g_nClass; i++)
	{
		arrScore[i] = 0;
		for (size_t j = 0; j < nParamLen; j++)
		{
			arrScore[i] += arrData[j] * _arrW[i*nParamLen + j];
		}
	}
	// check each score and update W
	for (size_t i = 0; i < g_nClass; i++)
	{		
		if (i == nLabel) continue;
		double dbBias = arrScore[i] - arrScore[nLabel] + dbDelta;
		dbBias = dbBias > 0 ? 1.0 : -1.0;
		if (dbBias>0)
		{
			// update W
			for (size_t j = 0; j < nParamLen; j++)
			{
				_arrW[i*nParamLen + j] -= arrData[j] * dbBias*dbLambda;
				_arrW[nLabel*nParamLen + j] -= arrData[nLabel] * dbBias*dbLambda;
			}
		}
	}
}

void ImageLayer::testMSVM() {

	// initialize W
	initW();
	test(0);


	int nEpochs = 100;
	double dbDelta = 0.1;
	double dbLambda = 0.0000001;
	const int nPixelLen = g_nPixels * 3 + 1;
	double arrData[nPixelLen];
	for (size_t epoch = 0; epoch < nEpochs; epoch++)
	{
		for (size_t fileIndex = 1; fileIndex < g_nFiles; fileIndex++)
		{
			for (size_t i = 0; i < g_nImgs; i++)
			{
				for (size_t j = 1; j < g_nPixels; j++)
				{
					for (size_t k = 0; k < 3; k++)
					{
						arrData[3 * j + k] = _arrPixels[fileIndex][i][j][k];
					}
				}
				arrData[nPixelLen - 1] = 1;
				updateW(arrData, _arrLabels[fileIndex][i], dbDelta, dbLambda);
			}
		}

		cout << epoch << "\t";
		test(0);
	}




}


void ImageLayer::testLossFun() {
	// calculate the loss
	/*
	for (size_t i = 0; i < 100; i++)
	{
	train(g_nFocusedIndex);
	// test result
	test(1);
	test(0);

	}*/
	for (size_t i = 0; i < 10; i++)
	{

		initW(true);
		cout << "loss: " << calculateLoss(0,_arrW) << "\t";
		test(0);
	}
}



void ImageLayer::testRandomLocalSearch() {
	int nEpochs = 100;
	initW();
	cout << "loss: " << calculateLoss(0, _arrW) << "\t";
	test(0);
	for (size_t i = 0; i < nEpochs; i++)
	{
		for (size_t j = 1; j < g_nFiles; j++)
		{
			trainRandomLocal(j);
		}
		cout << "loss: " << calculateLoss(0, _arrW) << "\t";
		test(0);
	}
}


void ImageLayer::testSoftMax() {
	// -1.create test data
	MyMatrix testData(g_nImgs, g_nPixels * 3);
	for (size_t j = 0; j < g_nImgs; j++)
	{
		for (size_t k = 0; k < g_nPixels; k++)
		{
			for (size_t l = 0; l < 3; l++)
			{
				testData.SetValue(j, k * 3 + l, _arrPixels[0][j][k][l] / (double)255);
			}
		}
	}

	// 0.create classifier
//	int nPoints = g_nImgs;
	int nPoints = (g_nFiles - 1)*g_nImgs;
	int nD = g_nPixels * 3;
	IMyClassifier* pClassifier = IMyClassifier::CreateClassifier(IMyClassifier::SoftMax, nPoints, nD, g_nClass);
	// 1.create training data
	MyMatrix input(nPoints, nD);
	int* arrLabels = new int[nPoints];
	int nIndex = 0;
	for (size_t i = 1; i < g_nFiles; i++)
	{
		for (size_t j = 0; j < g_nImgs; j++)
		{
			arrLabels[nIndex] = _arrLabels[i][j];
			for (size_t k = 0; k < g_nPixels; k++)
			{
				for (size_t l = 0; l < 3; l++)
				{
					input.SetValue(nIndex, k * 3 + l, _arrPixels[i][j][k][l]/(double)255);
				}
			}
			nIndex++;
		}


	}
	int nEpochs = 1;
	for (size_t i = 0; i < nEpochs; i++)
	{
		// 2.train
		pClassifier->Train(&input, arrLabels);

		// 3.test
		cout << "Epoch:\t"<<i<<"\tAccuracy on the test data: " << pClassifier->Test(&testData, _arrLabels[0]) << endl;

	}
	// 3.release resource
	delete pClassifier;
	delete[] arrLabels;
}


void ImageLayer::testAnn() {
	// -1.create test data
	MyMatrix testData(g_nImgs, g_nPixels * 3);
	for (size_t j = 0; j < g_nImgs; j++)
	{
		for (size_t k = 0; k < g_nPixels; k++)
		{
			for (size_t l = 0; l < 3; l++)
			{
				testData.SetValue(j, k * 3 + l, _arrPixels[0][j][k][l] / (double)255);
			}
		}
	}

	// 0.create classifier
	int nPoints = g_nImgs;
//	int nPoints = (g_nFiles - 1)*g_nImgs;
	int nD = g_nPixels * 3;
	IMyClassifier* pClassifier = IMyClassifier::CreateClassifier(IMyClassifier::Ann, nPoints, nD, g_nClass, 1000);
	// 1.create training data
	MyMatrix input(nPoints, nD);
	int* arrLabels = new int[nPoints];
	int nIndex = 0;
//	for (size_t i = 1; i < g_nFiles; i++)
	for (size_t i = 1; i < 2; i++)
	{
		for (size_t j = 0; j < g_nImgs; j++)
		{
			arrLabels[nIndex] = _arrLabels[i][j];
			for (size_t k = 0; k < g_nPixels; k++)
			{
				for (size_t l = 0; l < 3; l++)
				{
					input.SetValue(nIndex, k * 3 + l, _arrPixels[i][j][k][l] / (double)255);
				}
			}
			nIndex++;
		}


	}
	int nEpochs = 1;
	for (size_t i = 0; i < nEpochs; i++)
	{
		// 2.train
		pClassifier->Train(&input, arrLabels);

		// 3.test
		cout << "Epoch:\t" << i << "\tAccuracy on the test data: " << pClassifier->Test(&testData, _arrLabels[0]) << endl;

	}
	// 3.release resource
	delete pClassifier;
	delete[] arrLabels;
}