#include <QApplication>
#include <QPainter>
#include "canvas.h"

Canvas::Canvas(QWidget* parent) :
    QWidget(parent)
{
}

Canvas::~Canvas()
{
}

void Canvas::paintEvent(QPaintEvent *e)
{
    Q_UNUSED(e);
    doPainting();
}

void Canvas::doPainting()
{
    QPainter painter(this);
    QPen pen(Qt::black, 2);

    pen.setStyle(Qt::SolidLine);
    pen.setBrush(Qt::red);
    painter.setPen(pen);

    // TODO Detectar tamaños respecto a model (que loadGrid emita señal de la grilla, y aquí el slot receiveGrid lee la información)
    // TODO Generar tamaños de células
    // TODO Dibujarlas

    painter.drawLine(QPoint(10, 10), QPoint(200, 30));
}
