#include "pointscontrolwidget.h"

PointsControlWidget::PointsControlWidget(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	ui.spinBoxNum->setRange(0, 1000000);
	ui.spinBoxNum->setValue(1000);


	connect(ui.horizontalSliderLen, SIGNAL(valueChanged(int)), ui.spinBoxLen, SLOT(setValue(int)));
	connect(ui.spinBoxLen, SIGNAL(valueChanged(int)), ui.horizontalSliderLen, SLOT(setValue(int)));

	connect(ui.horizontalSliderPeriod, SIGNAL(valueChanged(int)), ui.spinBoxPeriod, SLOT(setValue(int)));
	connect(ui.spinBoxPeriod, SIGNAL(valueChanged(int)), ui.horizontalSliderPeriod, SLOT(setValue(int)));
}

PointsControlWidget::~PointsControlWidget()
{

}
