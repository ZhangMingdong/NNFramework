/*!
* \file ZCalcFuncs.h
* 
* Copyright (c) 2009,2010 by z / HJFXXY

algorithm com from LYD LingHang domain app
all angle parameters are rad, distance is m

*/
#pragma once
#include <math.h>
#include <vector>

using namespace std;

// define the standard trig values
#ifdef PI
#undef PI
#undef PI_2
#undef PI_4
#endif
const double PI   = 3.14159265358979323846;
const double PI_2 = 1.57079632679489661923;		//!< PI/2
const double PI_4 = 0.78539816339744830962;
const double PI_180=0.017453292519943295769236907684886;	//!< PI/180
const double PI_180_1=57.295779513082320876798154814105;	//!< 180/PI
const double ER_WGS84=6378137.0;		//!< earth radius WGS84

namespace calculator
{

	inline double Deg2Rad(double deg)
	{
		return deg*PI_180;
	}

	inline double Rad2Deg(double rad)
	{
		return rad*PI_180_1;
	}

	inline double formatAngle(double angle){
		if(angle<0)
			return angle+360;
		if(angle>=360)
			return angle-360;
		return angle;
	}
	inline float formatAngle(float angle){
		if(angle<0)
			return angle+360;
		if(angle>=360)
			return angle-360;
		return angle;
	}

	//return angle (degree)
	inline double angleDyDx(double dx, double dy)
	{
		if(0==dx)
		{
			if(dy>=0)
				return 0;
			else
				return 180;
		}
		double h=Rad2Deg(atan(dy/dx));
		if(dx>0)
			h=90-h;
		else
			h=270+h;
		return h;
	}
	inline void mirrorLatLon(double& lon, double& lat
		, double lon1,double lat1
		,double lon2,double lat2
		,double lon3,double lat3)
	{
		double lon23 = (lon2+lon3)/2;
		double lat23 = (lat2+lat3)/2;
		double p = (lat3 - lat2)/(lon3 - lon2);
		double lon0 = (p*p*lon1-p*lat1+p*lat23+lon23)/(1+p*p);
		double lat0 = (p*p*lat23-p*lon1+p*lon23+lat1)/(1+p*p);
		lon = 2*lon0 - lon1;
		lat = 2*lat0 - lat1;
	}

	inline double distanceLatLon(double lon1, double lat1, double lon2, double lat2)
	{
		double dLon=fabs(lon2-lon1);
		double dLat=fabs(lat2-lat1);

		double cosValue=sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(dLon);
		double resultBigCircle=acos(cosValue)*ER_WGS84;
		if(resultBigCircle>=1000000)
			return resultBigCircle;

		double dPhi=log(tan(lat2/2+PI_4)/tan(lat1/2+PI_4));
		double q;
		if(fabs(dPhi)>1e-10)
			q=dLat/dPhi;
		else
			q=cos(lat1);
		if(fabs(dLon)>PI)
			dLon=dLon>0?(dLon-2*PI):(2*PI+dLon);
		double d=sqrt(dLat*dLat+q*q*dLon*dLon)*ER_WGS84;
		return d;
	}
	inline int directionLatLon(double lon1,double lat1, double lon2, double lat2)
	{
		double yy=distanceLatLon(lon1, lat1
			,lon1,lat2);
		double xx=distanceLatLon(lon1, lat1
			,lon2,lat1);

		lon1=xx*((lon2-lon1)>0?1:-1);
		lat1=yy*((lat2-lat1)>0?1:-1);
		lat2=lat1/sqrt(lon1*lon1+lat1*lat1);

		return lon1>0? ((int)(acos(lat2)*180/PId+0.5)):(360-(int)(acos(lat2)*180/PId+0.5));
	}
	inline void extendLatLon(double& lonA, double& latA
		,double& lonB,double& latB
		, double lon1,double lat1
		,double lon2,double lat2
		,double lon3,double lat3)//对于中点进行平行扩展
	{
		double k = (lat3-lat1)/(lon3-lon1);
		double dis = sqrt((lon2-lon3)*(lon2-lon3)+(lat2-lat3)*(lat2-lat3))/10;
		double disy = dis*sin(atan(k));
		double disx = disy/k;
		if (k>0)
		{
			lonA = lon2 - disx;
			lonB = lon2 + disx;
			latA = lat2 - disy;
			latB = lat2 + disy;
		}
		else
		{
			lonA = lon2 + disx;
			lonB = lon2 - disx;
			latA = lat2 + disy;
			latB = lat2 - disy;
		}

	}
	inline int sweepAngleLatLon(double lon1,double lat1
		, double lon2, double lat2
		, double lon3, double lat3)
	{
		int A1 = directionLatLon(lon2,lat2,lon1,lat1);
		int A2 = directionLatLon(lon2,lat2,lon3,lat3);
		int A = A1-A2;

		return A>0?A:(360-A);
	}

	inline void toLatLonFromDistanceHeading(double& retLon, double& retLat, double lon, double lat, double dist, double heading)
	{
		retLat=retLon=0;
		retLat=lat+dist/ER_WGS84*cos(heading);
		double dphi=log(tan(retLat/2.0+PI_4)/tan(lat/2.0+PI_4));
		double q;
		if(fabs(dphi)>1e-10)
			q=(retLat-lat)/dphi;
		else
			q=cos(lat);
		double dLon=(dist/ER_WGS84)*sin(heading)/q;
		if(fabs(retLat)>PI_2)
			retLat=retLat>0?(PI-retLat): (-PI-retLat);
		retLon=lon+dLon+PI;
		int k=(int)(retLon/(2.0*PI));
		retLon=retLon-(1+2*k)*PI;
	}
	void calKB(double& k,double& b,double x1,double y1,double x2,double y2)
	{
		double _y = y2-y1;
		double _x = x2-x1;
		k=_y/_x;
		b =(x2*y1-x1*y2)/_x;
	}//求y=kx+b形式中的k,b
	void calJoint(double& x,double& y,
		double x1,double y1,
		double x2,double y2,
		double x3,double y3,
		double x4,double y4)//求两个两点直线的交点
	{
		double k1,k2,b1,b2;

		calKB(k1,b1,x1,y1,x2,y2);
		calKB(k2,b2,x3,y3,x4,y4);

		if (x1==x2)
		{
			x = x1;
			y = k2*x+b2;
			return;
		}
		if (x3==x4)
		{
			x = x3;
			y = k1*x+b1;
			return;
		}

		x = (b1-b2)/(k2-k1);
		y = (b1*k2-b2*k1)/(k2-k1);
	}

	void calPoint3(double& x,double& y,
		double x1,double y1,double x2,double y2,double dis)//求一点向另一点固定距离上的一点
	{
		double angle = atan((y2-y1)/(x2-x1));
		x = x1+cos(angle)*dis;
		y = y1+sin(angle)*dis;
	}
	void calPoint4(double& x,double& y,
		double x1,double y1,
		double x2,double y2,
		double x3,double y3,
		double x4,double y4)//求相切的另外一个点
	{
		double xx,yy;
		calJoint(xx,yy,x1,y1,x2,y2,x3,y3,x4,y4);

		double _x = xx-x2;
		double _y = yy-y2;
		double dis = sqrt(_x*_x+_y*_y);
		calPoint3(x,y,xx,yy,x3,y3,dis);
	}
	void BiasPt3ToArc(std::vector<DPoint3>& ptlist
		,double lon1,double lat1
		,double lon2, double lat2
		,double lon3,double lat3)
	{
		double radlon1 = Deg2Rad(lon1);
		double radlat1 = Deg2Rad(lat1);
		double radlon2 = Deg2Rad(lon2);
		double radlat2 = Deg2Rad(lat2);
		double radlon3 = Deg2Rad(lon3);
		double radlat3 = Deg2Rad(lat3);

		ptlist.push_back(DPoint3(lon3,lat3,0));//push the far point

		double dis1 = distanceLatLon(radlon1,radlat1,radlon2,radlat2);//pt1-pt2
		double dis2 = distanceLatLon(radlon1,radlat1,radlon3,radlat3);//pt2-pt3
		while (dis1<dis2)
		{
			double radlon,radlat;
			mirrorLatLon(radlon,radlat,radlon2,radlat2
				,radlon1,radlat1,radlon3,radlat3);// get point pt which is pt2 mirrored by pt1 & pt3,near pt3
			ptlist.push_back(DPoint3(Rad2Deg(radlon),Rad2Deg(radlat),0));//push the point
			radlon3 = radlon;
			radlat3 = radlat;//take near pt3
			dis2 = distanceLatLon(radlon1,radlat1,radlon3,radlat3);
		}
		ptlist.push_back(DPoint3(lon2,lat2,0));
		ptlist.push_back(DPoint3(lon1,lat1,0));
	}
	void CenterPt3ToArc(vector<DPoint3>& ptlist
		,double lon1,double lat1
		,double lon2, double lat2
		,double lon3,double lat3)
	{
		double radlon1 = Deg2Rad(lon1);
		double radlat1 = Deg2Rad(lat1);
		double radlon2 = Deg2Rad(lon2);
		double radlat2 = Deg2Rad(lat2);
		double radlon3 = Deg2Rad(lon3);
		double radlat3 = Deg2Rad(lat3);

		//定义拓展点A、B，并对中点进行水平拓展
		double radlonA,radlatA,radlonB,radlatB;
		extendLatLon(radlonA,radlatA,radlonB,radlatB,radlon1,radlat1,radlon2,radlat2,radlon3,radlat3);

		//前半部分对称求点列
		vector<DPoint3> listTemp;
		listTemp.push_back(DPoint3(lon1,lat1,0));//push the far point

		double dis1 = distanceLatLon(radlon2,radlat2,radlonA,radlatA);//pt1-pt2
		double dis2 = distanceLatLon(radlon2,radlat2,radlon1,radlat1);//pt2-pt3
		while (dis1<dis2)
		{
			double radlon,radlat;
			mirrorLatLon(radlon,radlat,radlonA,radlatA
				,radlon2,radlat2,radlon1,radlat1);// get point pt which is pt2 mirrored by pt1 & pt3,near pt3
			listTemp.push_back(DPoint3(Rad2Deg(radlon),Rad2Deg(radlat),0));//push the point
			radlon1 = radlon;
			radlat1 = radlat;//take near pt3
			dis2 = distanceLatLon(radlon2,radlat2,radlon1,radlat1);
		}

		//后半部分对称求点列
		ptlist.push_back(DPoint3(lon3,lat3,0));//push the far point

		dis1 = distanceLatLon(radlon2,radlat2,radlonB,radlatB);//pt1-pt2
		dis2 = distanceLatLon(radlon2,radlat2,radlon3,radlat3);//pt2-pt3
		while (dis1<dis2)
		{
			double radlon,radlat;
			mirrorLatLon(radlon,radlat,radlonB,radlatB
				,radlon2,radlat2,radlon3,radlat3);// get point pt which is pt2 mirrored by pt1 & pt3,near pt3
			ptlist.push_back(DPoint3(Rad2Deg(radlon),Rad2Deg(radlat),0));//push the point
			radlon3 = radlon;
			radlat3 = radlat;//take near pt3
			dis2 = distanceLatLon(radlon2,radlat2,radlon3,radlat3);
		}

		//组合点列
		for (int i = listTemp.size()-1;i>=0;i--)
		{
			ptlist.push_back(listTemp[i]);
		}

	}//--------------------------------------------------------------------------------
	void Pt3ToArc(std::vector<DPoint3>& ptlist
		,double lon1,double lat1
		,double lon2, double lat2 
		,double lon3,double lat3)
	{
		double radlon1 = Deg2Rad(lon1);
		double radlat1 = Deg2Rad(lat1);
		double radlon2 = Deg2Rad(lon2);
		double radlat2 = Deg2Rad(lat2);
		double radlon3 = Deg2Rad(lon3);
		double radlat3 = Deg2Rad(lat3);
		double radlon,radlat;

		mirrorLatLon(radlon,radlat,radlon2,radlat2
			,radlon1,radlat1,radlon3,radlat3);
		double dis1 = distanceLatLon(radlon,radlat,radlon2,radlat2);
		double dis2 = distanceLatLon(radlon1,radlat1,radlon,radlat);
		if (dis1>dis2)
		{
			BiasPt3ToArc(ptlist,lon1,lat1,lon2,lat2,lon3,lat3);
		}
		else
		{
			BiasPt3ToArc(ptlist,lon1,lat1,Rad2Deg(radlon),Rad2Deg(radlat),lon2,lat2);
		}

	}

}
