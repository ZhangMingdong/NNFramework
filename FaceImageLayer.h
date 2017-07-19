#pragma once
#include "ImageLayer.h"
class NNImageList;

class FaceImageLayer :
	public ImageLayer
{
public:
	FaceImageLayer();
	virtual ~FaceImageLayer();
public:

	virtual void Initialize();
	virtual void Draw();
private:

	// train neural network using face data
	void trainNNFace();
	// generate the texture of the image
	virtual void generateTexture();
	// load the training data and test data
	void loadData();

	// test my ann classifier
	void testMyAnn();
private:
	// 0-training data;1,2-testing data
	NNImageList* _arrImage[3];
	// size of training set
	int _nTrainSize;
	// number of columns
	int _nCols;
	int _nRows;
	int _nImgCols;
	int _nImgRows;
};

