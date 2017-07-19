#pragma once

/*
	class to represent a vector with label
*/
class LabeledVector
{
protected:
	// the data buffer
	double* _arrData;
	// the label
	int _nLabel;
	// dimension
	int _nD;
	// number of class
	int _nK;
public:
	LabeledVector(int nD);
	virtual ~LabeledVector();
	double GetData(int nIndex)const;
	void SetData(int nIndex,double dbData);
	int GetLabel()const { return _nLabel; }
	int GetDimension() const { return _nD; }
	int GetK()const { return _nK; }
	void SetLabel(int nLabel) { _nLabel = nLabel; }
	void SetK(int nK) { _nK = nK; }
};

