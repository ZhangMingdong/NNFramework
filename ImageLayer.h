#pragma once
#include "ILayer.h"
#include <vector>
#include <QGLWidget>


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
	// read the test data set 1
	void readData(int nIndex,const char* strFile);

	// basic statistic, calculate mean
	void statistic();

	// test classification by mean
	void testMean();

	// classify an image whose index is nIndex, return its label
	int classify(int nIndex);
private:
	static const int g_nImgRow = 100;
	static const int g_nImgCol = 100;
	static const int g_nRow = 32;
	static const int g_nCol = 32;
	static const int g_nClass = 10;
	static const int g_nImgs = g_nImgRow*g_nImgCol;
	static const int g_nPixels = g_nRow*g_nCol;

	static const int g_nFiles = 6;					// number of files

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
};

