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

    if (m_data.size() > 0)
    {
        for (int j = 0; j < m_data.size(); j++)
        {
            for (int i = 0; i < m_data.at(j).size(); i++)
            {
                Qt::GlobalColor color = (m_data.at(j).at(i) == QChar('1')) ? Qt::red : Qt::black;
                painter.fillRect(i * m_cellWidth, j * m_cellHeight, m_cellWidth, m_cellHeight, color);
            }
        }
    }
}

void Canvas::receiveGridReader(GridReader *gridReader)
{
    m_data = gridReader->getData();

    m_gridWidth = gridReader->getDetectedWidth();
    m_gridHeight = gridReader->getDetectedHeight();

    m_cellWidth = this->width() / m_gridWidth;
    m_cellHeight = this->height() / m_gridHeight;
}
