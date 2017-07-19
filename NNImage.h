#pragma once
#include <vector>
/*
	class used to represent the image data used in neural network
	Mingdong
	2017/07/02
*/
class NNImage
{
public:
	NNImage();
	~NNImage();
public:
	int _nRows = 0;
	int _nCols = 0;
	int _nTuls = 1;
	char *_arrName = 0;
	int *_arrData = 0;
	int _nLabel = -1;				// used in new dataset
	int _nK = 0;					// number of classes
public:
	void SetPixel(int r, int c,int t, int val);
	int GetPixel(int r, int c,int t);
public:
	// read an image from file
	static NNImage* Read(char *filename);
protected:
	// set the label and number of classes from the file name
	void setLabelFromName();
};


struct NNImageList : public std::vector<NNImage*> {
public:
	// original method to load the data from file
	void LoadFromFile(char *filename);
	// method for new data
	void LoadFromFileNew(const char *filename);
};
