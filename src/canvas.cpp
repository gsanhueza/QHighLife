#include <QApplication>
#include <QPainter>
#include "canvas.h"

Canvas::Canvas(QWidget* parent) :
    QWidget(parent),
    m_grid(nullptr)
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

    // Initial (GridReader)
    if (m_grid == nullptr)
    {
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
    // Updated (Grid)
    else
    {
        for (unsigned int j = 0; j < m_grid->getHeight(); j++)
        {
            for (unsigned int i = 0; i < m_grid->getWidth(); i++)
            {
                Qt::GlobalColor color = (m_grid->getAt(i, j)) ? Qt::blue : Qt::green;
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

void Canvas::receiveGrid(Grid *grid)
{
    m_grid = grid;

    update();
}