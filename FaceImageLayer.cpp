#include "FaceImageLayer.h"

#include "BPNeuralNetwork.h"
#include "NNImage.h"

#include "MyMatrix.h"
#include "IMyClassifier.h"

#include <iostream>

using namespace std;

// get the head label from the file name
int GetHeadLabel(char* strFileName) {
	int scale;
	char userid[40], head[40], expression[40], eyes[40], photo[40];

	userid[0] = head[0] = expression[0] = eyes[0] = photo[0] = '\0';

	/*** scan in the image features ***/
	sscanf(strFileName, "%[^_]_%[^_]_%[^_]_%[^_]_%d.%[^_]",
		userid, head, expression, eyes, &scale, photo);

	char *p = strrchr(eyes, '.');
	if (p != NULL)
	{
		userid[0] = head[0] = expression[0] = eyes[0] = photo[0] = '\0';

		sscanf(strFileName, "%[^_]_%[^_]_%[^_]_%[^.].%[^_]", userid, head, expression, eyes, photo);
		scale = 1;
	}

	p = strrchr(userid, '\\');
	if (p != NULL)
		p++;
	if (!strcmp(eyes, "sunglasses")) {
		return 0;
	}
	else {
		return 1;
	}
	/*
	if (!strcmp(head, "up")) {
		return 0;
	}
	else if (!strcmp(head, "left")) {
		return 1;
	}
	else if (!strcmp(head, "straight")) {
		return 2;
	}
	else {
		return 3;
	}*/
}

FaceImageLayer::FaceImageLayer()
{
	for (size_t i = 0; i < 3; i++)
	{
		_arrImage[i] = NULL;
	}
}

FaceImageLayer::~FaceImageLayer()
{
	for (size_t i = 0; i < 3; i++)
	{
		if (_arrImage[i])delete (_arrImage[i]);
	}
}

void FaceImageLayer::Initialize() {
	// 1.load the face image data
	loadData();
	// 2.train the classifier
	trainNNFace();
	// 3.generate face texture
	generateTexture();

//	testMyAnn();
}

void FaceImageLayer::loadData() {
	/*** Create imagelists ***/
	char* arrFileName[3] = {
		"trainset/all_scale1_train.list"
		,"trainset/all_scale1_test1.list"
		,"trainset/all_scale1_test2.list"
	};
	for (size_t i = 0; i < 3; i++)
	{
		_arrImage[i] = new NNImageList();
		// load train, test1, or test2 sets
		_arrImage[i]->LoadFromFile(arrFileName[i]);
		for (size_t j = 0,length= (*_arrImage[i]).size(); j < length; j++)
		{
			NNImage* pImage = (*_arrImage[i])[j];
			int nD = pImage->_nCols*pImage->_nRows*pImage->_nTuls;
			LabeledVector*pLV=new LabeledVector(nD);
			pImage->SetLabeledVector(pLV);
			_arrInputVector[i].push_back(pLV);
		}
	}


	_nTrainSize = _arrImage[0]->size();
	_nImgCols = sqrt(_nTrainSize) + 1;
	_nImgRows = _nTrainSize / _nImgCols + 1;
	_nCols = (*_arrImage[0])[0]->_nCols;
	_nRows = (*_arrImage[0])[0]->_nRows;
}

void FaceImageLayer::trainNNFace() {
	int seed = 102194;   /*** today's date seemed like a good default ***/
	int epochs = 200;
	int savedelta = 100;
	int list_errors = 0;

	char* netname = g_pModelFileName;
	

	/*** Initialize the neural net package ***/
	BPNeuralNetwork::Initialize(seed);

	/*** Show number of images in train, test1, test2 ***/
	printf("%d images in training set\n", _arrImage[0]->size());
	printf("%d images in test1 set\n", _arrImage[1]->size());
	printf("%d images in test2 set\n", _arrImage[2]->size());
	if (epochs > 0) {
		printf("Training underway (going to %d epochs)\n", epochs);
		printf("Will save network every %d epochs\n", savedelta);
	}



	// 0.Create network
	if (_pBPNN)
	{
		delete _pBPNN;
		_pBPNN = NULL;
	}


	NNImage *iimg = (*_arrImage[0])[0];
	int imgsize = iimg->_nRows*iimg->_nCols;
	_pBPNN = BPNeuralNetwork::Create(imgsize, g_nHiddenLayers, g_nOutputLayers);


	/*** Print out performance before any epochs have been completed. ***/
	printf("0 0.0 ");

	_pBPNN->CalculatePerformance(_arrInputVector[0], 0);
	_pBPNN->CalculatePerformance(_arrInputVector[1], 0);
	_pBPNN->CalculatePerformance(_arrInputVector[2], 0);
	printf("\n");

	// 1.train
	for (int epoch = 1; epoch <= epochs; epoch++) {

		printf("%d ", epoch);

		double out_err;
		double hid_err;
		double sumerr = 0.0;
		for (int i = 0; i < _nTrainSize; i++) {

			/** Set up input units and target vector on net with the i'th data **/
			_pBPNN->LoadInputData(_arrInputVector[0][i]);

			/** Run backprop, learning rate 0.3, momentum 0.3 **/
			_pBPNN->Train(0.3, 0.3, &out_err, &hid_err);

			sumerr += (out_err + hid_err);
		}
		printf("%g ", sumerr);

		/*** Evaluate performance on train, test, test2, and print perf ***/
//		_pBPNN->CalculatePerformance(_arrImage[0], 0);
//		_pBPNN->CalculatePerformance(_arrImage[1], 0);
//		_pBPNN->CalculatePerformance(_arrImage[2], 0);
		_pBPNN->CalculatePerformance(_arrInputVector[0], 0);
		_pBPNN->CalculatePerformance(_arrInputVector[1], 0);
		_pBPNN->CalculatePerformance(_arrInputVector[2], 0);
		printf("\n");  

		/*** Save network every 'savedelta' epochs ***/
		if (!(epoch % savedelta)) {
			_pBPNN->Save(netname);
		}
	}
	printf("\n");

	// 3.save the network
	if (epochs > 0) {
		_pBPNN->Save(netname);
	}
}

void FaceImageLayer::Draw() {
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

void FaceImageLayer::generateTexture() {

	// 3.generate texture
	_dataTexture = new GLubyte[4 * _nCols*_nRows* _nImgRows*_nImgCols];
	for (size_t l = 0; l < _nTrainSize; l++)
	{
		int jj = l%_nImgCols;
		int ii = l / _nImgCols;
		for (size_t i = 0; i < _nRows; i++)
		{
			for (size_t j = 0; j < _nCols; j++)
			{
				int iii = ii*_nRows + i;
				int jjj = jj * _nCols + j;
				iii = _nRows*_nImgRows-1 - iii;	// up-down;
				int nPixelIndex = iii * _nImgCols * _nCols + jjj; 

				_dataTexture[4 * nPixelIndex + 0] = (*_arrImage[0])[l]->GetPixel(i, j, 0);
				_dataTexture[4 * nPixelIndex + 1] = (*_arrImage[0])[l]->GetPixel(i, j, 1);
				_dataTexture[4 * nPixelIndex + 2] = (*_arrImage[0])[l]->GetPixel(i, j, 2);
				_dataTexture[4 * nPixelIndex + 3] = (GLubyte)255;

//				cout << (*_arrImage[0])[l]->GetPixel(i, j, 0) << "\t" 
//					<< (*_arrImage[0])[l]->GetPixel(i, j, 1) << "\t" 
//					<< (*_arrImage[0])[l]->GetPixel(i, j, 2) << endl;
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

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _nImgCols*_nCols, _nImgRows*_nRows, 0, GL_RGBA, GL_UNSIGNED_BYTE, _dataTexture);


	/*
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
	*/

}

void FaceImageLayer::testMyAnn() {
	// number of pixels in each image
	int nPixels = _nCols*_nRows;
	int nTestSize1 = (*_arrImage[1]).size();
	int nTestSize2 = (*_arrImage[2]).size();
	int* arrLabels1 = new int[nTestSize1];
	int* arrLabels2 = new int[nTestSize2];
	// -1.create test data
	MyMatrix testData1(nTestSize1, nPixels);
	MyMatrix testData2(nTestSize2, nPixels);
	for (size_t j = 0; j < nTestSize1; j++)
	{
		arrLabels1[j] = GetHeadLabel((*_arrImage[1])[j]->_arrName);
		for (size_t k = 0; k < nPixels; k++)
		{
			int ii = k / _nCols;
			int jj = k%_nCols;
			testData1.SetValue(j, k, (*_arrImage[1])[j]->GetPixel(ii, jj, 0) / (double)255);
		}
	}


	for (size_t j = 0; j < nTestSize2; j++)
	{
		arrLabels2[j] = GetHeadLabel((*_arrImage[2])[j]->_arrName);
		for (size_t k = 0; k < nPixels; k++)
		{
			int ii = k / _nCols;
			int jj = k%_nCols;
			testData2.SetValue(j, k, (*_arrImage[2])[j]->GetPixel(ii, jj, 0) / (double)255);
		}
	}

	// 0.create classifier
	int nD = nPixels;
	int nClass = 2;
	IMyClassifier* pClassifier = IMyClassifier::CreateClassifier(IMyClassifier::Ann, _nTrainSize, nD, nClass, 100);
	// 1.create training data
	MyMatrix input(_nTrainSize, nD);
	int* arrLabels = new int[_nTrainSize];

	for (size_t j = 0; j < _nTrainSize; j++)
	{
		arrLabels[j] = GetHeadLabel((*_arrImage[0])[j]->_arrName);
		for (size_t k = 0; k < nPixels; k++)
		{
			int ii = k / _nCols;
			int jj = k%_nCols;
			double dbValue = (*_arrImage[0])[j]->GetPixel(ii, jj, 0) / (double)255;
			input.SetValue(j, k, dbValue);
//			cout << dbValue << endl;
		}
	}
//	input.Print();
	int nEpochs = 1;
	for (size_t i = 0; i < nEpochs; i++)
	{
		// 2.train
		pClassifier->Train(&input, arrLabels);

		// 3.test
		cout << "Epoch:\t" << i << "\tAccuracy on the test data: " << pClassifier->Test(&testData1, arrLabels1) << endl;
		cout << "Epoch:\t" << i << "\tAccuracy on the test data: " << pClassifier->Test(&testData2, arrLabels2) << endl;

	}
	// 3.release resource
	delete pClassifier;
	delete[] arrLabels;
	delete[] arrLabels1;
	delete[] arrLabels2;
}