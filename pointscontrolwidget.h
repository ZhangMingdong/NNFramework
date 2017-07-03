#ifndef POINTSCONTROLWIDGET_H
#define POINTSCONTROLWIDGET_H

#include <QWidget>
#include "ui_pointscontrolwidget.h"

class PointsControlWidget : public QWidget
{
	Q_OBJECT

public:
	PointsControlWidget(QWidget *parent = 0);
	~PointsControlWidget();

public:
	Ui::PointsControlWidget ui;
};

#endif // POINTSCONTROLWIDGET_H
