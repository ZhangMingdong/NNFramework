// ==================================================
//  name:	MathTypes   
//	date:		09/09/2009
//	demo:	
//	creator:	Rigel
// ==================================================
#pragma once
//--------------------------------------------------------------------------------
#include <math.h>
#include <vector>
#include <string>
//--------------------------------------------------------------------------------
#ifndef PId
 #define PId	3.14159265358979323846264338
 #define PIf 	3.14159265358979323846264338f
 #define PI2d 	6.28318530717958647692528676
 #define PI2f 	6.28318530717958647692528676f
 #define PID2d	1.57079632679489661923132169
 #define PID2f	1.57079632679489661923132169f
 #define PID3d	1.04719755119659774615421446
#endif
//--------------------------------------------------------------------------------
//三维点的数据结构
//20111214
struct DPoint3{
	DPoint3() { x = y = z = 0.0f; }
	DPoint3(double fx, double fy, double fz) { x=fx; y=fy; z=fz; }
	DPoint3 &operator=(const DPoint3 &v) { x = v.x; y = v.y; z = v.z; return *this; }
	double x, y, z;
};//--------------------------------------------------------------------------------
//二维点的数据结构
//DPoint3用来表示世界坐标系中的点，而IPoint2用来表示屏幕坐标系中的点，
//因此允许两点相加，这在世界坐标系中是没有实际意义的，但在屏幕坐标系中
//可以用来描述一些显示要素
//20111214
struct IPoint2 {
	IPoint2() {x = 0; y = 0;}
	IPoint2(int ix, int iy) { x=ix; y=iy; }
	IPoint2 &operator=(const IPoint2 &v) { x = v.x; y = v.y; return *this; }
	int x, y;

	IPoint2 operator +(const IPoint2 &v) const { return IPoint2(x+v.x, y+v.y); }
};//--------------------------------------------------------------------------------
//颜色定义
//20111214
struct RGBAf{
	RGBAf() {}
	RGBAf(float _r, float _g, float _b) { r = _r; g = _g; b = _b; a = 1.0f; }
	RGBAf(float _r, float _g, float _b, float _a) { r = _r; g = _g; b = _b; a = _a; }
	// assignment
	RGBAf &operator=(const RGBAf &v) {
		r = v.r;
		g = v.g;
		b = v.b; 
		a=v.a; 
		return *this; 
	}
	float r, g, b, a;
};
//暂不细考
//20111214
struct FBox3
{
public:
	FBox3() {}
	FBox3(float x1, float y1, float z1, float x2, float y2, float z2)
	{
		_x1 = x1;
		_x2 = x2;
		_y1 = y1;
		_y2 = y2;
	}

	void Set(float x1, float y1, float z1, float x2, float y2, float z2)
	{
		_x1 = x1;
		_x2 = x2;
		_y1 = y1;
		_y2 = y2;
	}


	void GrowToContainBox(const FBox3 &box)
	{
		if (box._x1 < _x1) _x1 = box._x1;
		if (box._y1 < _y1) _y1 = box._y1;
		if (box._x2 > _x2) _x2 = box._x2;
		if (box._y2 > _y2) _y2 = box._y2;
	}
	void GrowToContainPoint(const DPoint3 &point)
	{
		if (point.x < _x1) _x1 = point.x;
		if (point.y < _y1) _y1 = point.y;
		if (point.x > _x2) _x2 = point.x;
		if (point.y > _y2) _y2 = point.y;
	}

//	FPoint3	min, max;
	float _x1;
	float _x2;
	float _y1;
	float _y2;
	float _z1;
	float _z2;
	float MidX(){return (_x1+_x2)/2;};
	float MidY(){return (_y1+_y2)/2;};
	void operator *=(float f)
	{
		_x1 *= f;
		_x2 *= f;
		_y1 *= f;
		_y2 *= f;
	}
	void operator /=(float f)
	{
		_x1 /= f;
		_x2 /= f;
		_y1 /= f;
		_y2 /= f;
	}
};//--------------------------------------------------------------------------------

class ILayer;
struct IPoint2;
// 距离和索引，用于返回寻找最近点对的结果
struct DistanceAndIndices
{
	double _dbDis;
	int _nIndex1;
	int _nIndex2;
};


class Point : public DPoint3
{
public:
	Point() {};
	Point(DPoint3& p) :DPoint3(p), _ulPos(0) {}
	int _ulPos;
};
