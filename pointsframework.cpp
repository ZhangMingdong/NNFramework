#include "pointsframework.h"

#include <QDockWidget>



PointsFramework::PointsFramework(QWidget *parent)
	: QMainWindow(parent)
{

	pWidget = new MyGLWidget(this);
	this->setCentralWidget(pWidget);


	createDockWidgets();
	createConnections();
	this->showMaximized();
}

PointsFramework::~PointsFramework()
{

}

void PointsFramework::createDockWidgets(){
	setDockOptions(QMainWindow::AnimatedDocks);
	QDockWidget::DockWidgetFeatures features =
		QDockWidget::DockWidgetMovable |
		QDockWidget::DockWidgetFloatable;

	_pControlWidget = new PointsControlWidget();


	QDockWidget *controlDockWidget = new QDockWidget(
		tr("Control"), this);
	controlDockWidget->setFeatures(features);
	controlDockWidget->setWidget(_pControlWidget);
	addDockWidget(Qt::RightDockWidgetArea, controlDockWidget);

}

void PointsFramework::createConnections(){
	connect(_pControlWidget->ui.pushButtonCalc, SIGNAL(pressed()), this, SLOT(onCalculateClicked()));
	connect(_pControlWidget->ui.pushButtonCalcDC, SIGNAL(pressed()), this, SLOT(onCalculateDCClicked()));
	connect(_pControlWidget->ui.pushButtonGenerateRandom, SIGNAL(pressed()), this, SLOT(onGenerateRandomClicked()));
	connect(_pControlWidget->ui.pushButtonGenerateBlueNoise, SIGNAL(pressed()), this, SLOT(onGenerateBlueNoiseClicked()));
	connect(_pControlWidget->ui.pushButtonGenerateBlueNoiseNormal, SIGNAL(pressed()), this, SLOT(onGenerateBlueNoiseNormalClicked()));
	connect(_pControlWidget->ui.pushButtonGenerateMulticlassBlueNoise, SIGNAL(pressed()), this, SLOT(onGenerateMulticlassBlueNoiseClicked()));
	connect(_pControlWidget->ui.pushButtonGenerateNormal, SIGNAL(pressed()), this, SLOT(onGenerateNormalClicked()));
	connect(_pControlWidget->ui.pushButtonGenerateMultiNormal, SIGNAL(pressed()), this, SLOT(onGenerateMultiNormalClicked()));
	connect(_pControlWidget->ui.pushButtonGenerateSequence, SIGNAL(pressed()), this, SLOT(onGenerateSequenceClicked()));
	connect(_pControlWidget->ui.pushButtonGenerateSequence2D, SIGNAL(pressed()), this, SLOT(onGenerateSequence2DClicked()));
	connect(_pControlWidget->ui.checkBoxPicking, SIGNAL(clicked(bool)), this, SLOT(onPickingClicked(bool)));


	connect(_pControlWidget->ui.checkBoxShowBackground, SIGNAL(clicked(bool)), pWidget, SLOT(onShowBackground(bool)));
	connect(_pControlWidget->ui.pushButtonUpdateLayer, SIGNAL(pressed()), pWidget, SLOT(onUpdateLayer()));

	connect(_pControlWidget->ui.horizontalSliderLen, SIGNAL(valueChanged(int)), pWidget, SLOT(onSetSampleLen(int)));
	connect(_pControlWidget->ui.horizontalSliderPeriod, SIGNAL(valueChanged(int)), pWidget, SLOT(onSetSamplePeriod(int)));
}

void PointsFramework::onGenerateRandomClicked(){
	// generate random points
	/*
 	pWidget->GenerateRandomPoints(_pControlWidget->ui.spinBoxNum->value());

	pWidget->updateGL();*/

}

void PointsFramework::onGenerateBlueNoiseClicked() {
	// generate normal points

	long t1 = GetTickCount();
	pWidget->GenerateBlueNoise(_pControlWidget->ui.spinBoxNum->value());
	int t = GetTickCount() - t1; 
	QString s1 = QStringLiteral("生成蓝噪声。\n计算时间：") + QString::number(t) + QStringLiteral("dms。\n");
	_pControlWidget->ui.textEditResult->setPlainText(_pControlWidget->ui.textEditResult->toPlainText() + s1);

	pWidget->updateGL();
}void PointsFramework::onGenerateBlueNoiseNormalClicked() {
	// generate normal points

	long t1 = GetTickCount();
	pWidget->GenerateBlueNoiseNormal(_pControlWidget->ui.spinBoxNum->value(), 0, 0, .4, .2);
	int t = GetTickCount() - t1;
	QString s1 = QStringLiteral("生成正态蓝噪声。\n计算时间：") + QString::number(t) + QStringLiteral("dms。\n");
	_pControlWidget->ui.textEditResult->setPlainText(_pControlWidget->ui.textEditResult->toPlainText() + s1);

	pWidget->updateGL();
}

void PointsFramework::onGenerateMulticlassBlueNoiseClicked() {
	// generate normal points

	long t1 = GetTickCount();
	pWidget->GenerateMulticlassBlueNoise(_pControlWidget->ui.spinBoxNum->value());
	int t = GetTickCount() - t1; 
	QString s1 = QStringLiteral("生成多组蓝噪声。\n计算时间：") + QString::number(t) + QStringLiteral("dms。\n");
	_pControlWidget->ui.textEditResult->setPlainText(_pControlWidget->ui.textEditResult->toPlainText() + s1);

	pWidget->updateGL();
}

void PointsFramework::onGenerateNormalClicked() {
	// generate normal points
	pWidget->GenerateNormalPoints(_pControlWidget->ui.spinBoxNum->value(), 0, 0, .4, .2);
	pWidget->updateGL();
}

void PointsFramework::onGenerateMultiNormalClicked() {
	// generate normal points
	pWidget->GenerateMVNPoints(_pControlWidget->ui.spinBoxNum->value(), 0, 0, .4, .2);
	pWidget->updateGL();

}

void PointsFramework::onGenerateSequenceClicked() {
	// generate sequence
	pWidget->GenerateSequence();
	pWidget->updateGL();

}

void PointsFramework::onGenerateSequence2DClicked() {
	// generate sequence
	pWidget->GenerateSequence2D();
	pWidget->updateGL();

}

void PointsFramework::onCalculateClicked(){
	long t1 = GetTickCount();
	DistanceAndIndices result = pWidget->CalcNearestPairOfPoints();
	pWidget->updateGL();
	int t = GetTickCount() - t1;
	QString s1 = QStringLiteral("找到最近点对。\n计算时间：") + QString::number(t) + QStringLiteral("dms。\n");
	QString s2 = QStringLiteral("最近点对的索引：") + QString::number(result._nIndex1) + ","
		+ QString::number(result._nIndex2) + QStringLiteral("。\n");
	QString s3 = QStringLiteral("距离：") + QString::number(result._dbDis) + QStringLiteral("。\n");
	QString s4 = "==============\n";
	_pControlWidget->ui.textEditResult->setPlainText(_pControlWidget->ui.textEditResult->toPlainText() + s1 + s2 + s3 + s4);


}

void PointsFramework::onCalculateDCClicked(){
	long t1 = GetTickCount();
	DistanceAndIndices result = pWidget->CalcNearestPairOfPoints_DC();
	pWidget->updateGL();
	int t = GetTickCount() - t1;
	QString s1 = QStringLiteral("分治找到最近点对。\n计算时间：") + QString::number(t) + QStringLiteral("dms。\n");
	QString s2 = QStringLiteral("最近点对的索引：") + QString::number(result._nIndex1) + ","
		+ QString::number(result._nIndex2) + QStringLiteral("。\n");
	QString s3= QStringLiteral("距离：") + QString::number(result._dbDis) + QStringLiteral("。\n");
	QString s4 = "==============\n";
	_pControlWidget->ui.textEditResult->setPlainText(_pControlWidget->ui.textEditResult->toPlainText()+s1+s2+s3+s4);

}

void PointsFramework::onPickingClicked(bool bChecked){
	pWidget->SetHandPoint();

}

