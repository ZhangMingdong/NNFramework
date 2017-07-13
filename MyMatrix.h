#pragma once
// maxtrix structure
class MyMatrix
{
public:
	// empty matrix
	MyMatrix();
	// construct by r and c
	MyMatrix(int r,int c);
	// construct from multiplying two matrix
	MyMatrix(const MyMatrix* m1, const MyMatrix* m2);
	// create a matrix from another one
	MyMatrix(const MyMatrix* m1, bool bTranspose = false);

	// construct by apply a function on the element of m1
	MyMatrix(const MyMatrix* m1,double(*f)(double));

	// construct matrix using pattern m1*m2+b
	MyMatrix(const MyMatrix* m1, const MyMatrix* m2,const MyMatrix* b);
	virtual ~MyMatrix();
private:
	int _nR;				// rows
	int _nC;				// columns
	double** _data;			// the data
public:
	// get a value
	double GetValue(int i, int j)const;
	// set a value
	void SetValue(int i, int j, double v);
	// trim the negative to 0
	void TrimNegative();
	// multiply a factor
	void Multi(double dbFactor);
	// calculate the norm2
	double Norm2();
	// initialize value randomly
	void InitRandom(double dbScale);
	// initialize value to 0
	void InitZero();
public:		// modify
	// using the result of multiplication
	void Multi(const MyMatrix* m1, const MyMatrix* m2);
	// using the result of the formula
	void Formula(const MyMatrix* m1, const MyMatrix* m2, const MyMatrix* b);

	// linear combination: m=m+m1*k
	void Linear(const MyMatrix* m1, double dbScale);
	
	// get the reference of the row of given index
	const double* GetRow(int r) const;

	// apply a function to each element
	void ApplyFun(double(*f)(double));

	// sum the matrix along the axis
	void Sum(MyMatrix* pResult, int nAxis);

	// calculate sum of the element, apply f if need
	double Sum(double(*f)(double));

	// divide by pDivisor
	void Div(MyMatrix* pResult, MyMatrix* pDivisor);

	int Rows() { return _nR; }
	int Cols() { return _nC; }


private:
	// allocate space for _data
	void allocateSpace();

};

