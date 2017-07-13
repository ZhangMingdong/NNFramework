#pragma once
#include "ILayer.h"
#include <vector>
#include <QGLWidget>

#include "setting.h"
/*
	Layer to show the 2D sequence
	Mingdong
	2017/06/21
	*/
class ImageLayer :
	public ILayer
{
public:
	ImageLayer();
	virtual ~ImageLayer();
public:
	virtual void Draw();
private:

	// mode of the test
	enum EnumTestMode
	{
		TM_Mean
		, TM_W
	};
private:
	// read the test data set 1
	void readData(int nIndex,const char* strFile);

	// basic statistic, calculate mean
	void statistic();

	// test classification by mean, using nTestIndex's data as test data
	void test(int nTestIndex,EnumTestMode mode=TM_W);

	void generateTexture();

	// classify an image whose index is nIndex, return its label
	int classifyByMean(int nFileIndex,int nImgIndex);

	// classify an image whose index is nIndex, return its label
	int classifyByW(int nFileIndex, int nImgIndex);

	// initialize W matrix
	void initW(bool bRandom=false);

	// calculate loss of current w
	double calculateLoss(int nFileIndex,const double *pW);

	// calculate data loss of the nIndex's image
	double calculateDataLoss(int nFileIndex, int nImgIndex, const double* pW);

	// calculate the score vector of the nIndex's image of the nImgIndex's file
	void calculateScore(int nFileIndex,int nImgIndex, const double* pW, double* arrScore);

	// training random locally
	void trainRandomLocal(int nFileIndex);

	// train neural network
	void trainNN(int nDataSetIndex);

	// test mean classifier
	void testMeanClassifier();

	// test nearest neighbor classifier
	void testNearestNeighbor();

	// get the label of the image by NN
	int getLabelByNN(unsigned char pImg[1024][3]);

	// test the algorithm of Multiclass SVM
	void testMSVM();

	// update W for a training instance
	void updateW(double* arrData, int nLabel, double dbDelta, double dbLambda);

	// test the loss function
	void testLossFun();

	// test random local search
	void testRandomLocalSearch();

	// test SoftMax classifier
	void testSoftMax();
private:


private:
	// texture data and id
	GLubyte* _dataTexture;
	GLuint texID[2];
	GLubyte* _dataTextureMean;

	// array of labels
	unsigned char _arrLabels[g_nFiles][g_nImgs];
	// array of pixels
	unsigned char _arrPixels[g_nFiles][g_nImgs][g_nPixels][3];	// the array[0] is the test data

	// mean of each class
	double _arrMean[g_nClass][g_nPixels][3];

	// the w matrix;
	double* _arrW;
	double _dbDelta;
	double _dbLambda;
};

