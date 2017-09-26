#include <QApplication>
#include <QPainter>
#include "canvas.h"
#include <iostream>

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

    QVector<QString> data = m_gridReader.getData();
    for (int j = 0; j < data.size(); j++)
    {
        for (int i = 0; i < data.at(i).size(); i++)
        {
            painter.drawRect(i * m_cellWidth, j * m_cellHeight, m_cellWidth, m_cellHeight);
        }
    }
}

void Canvas::receiveGridReader(GridReader gridReader)
{
    m_gridReader = gridReader;

    m_gridWidth = gridReader.getDetectedWidth();
    m_gridHeight = gridReader.getDetectedHeight();

    m_cellWidth = this->width() / m_gridWidth;
    m_cellHeight = this->height() / m_gridHeight;
}
