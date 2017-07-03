#ifndef POINTSFRAMEWORK_H
#define POINTSFRAMEWORK_H

#include <QtWidgets/QMainWindow>

#include "myglwidget.h"
#include "pointscontrolwidget.h"


class PointsFramework : public QMainWindow
{
	Q_OBJECT

public:
	PointsFramework(QWidget *parent = 0);
	~PointsFramework();

private:
	MyGLWidget* pWidget;
	PointsControlWidget* _pControlWidget;
protected:
	void createDockWidgets();
	void createConnections();
public slots:
	void onGenerateNormalClicked();
	void onGenerateBlueNoiseClicked();
	void onGenerateBlueNoiseNormalClicked();
	void onGenerateMulticlassBlueNoiseClicked();
	void onGenerateRandomClicked();
	void onGenerateMultiNormalClicked();
	void onGenerateSequenceClicked();
	void onGenerateSequence2DClicked();
	void onCalculateClicked();
	void onCalculateDCClicked();
	void onPickingClicked(bool bChecked);
};

#endif // POINTSFRAMEWORK_H
