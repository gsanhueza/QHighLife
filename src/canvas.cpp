#include <QApplication>
#include <QPainter>
#include "canvas.h"

Canvas::Canvas(QWidget* parent) :
    QWidget(parent),
    m_grid(nullptr),
    out(stdout)
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
                    painter.fillRect(i * m_cellWidth, j * m_cellHeight, m_cellWidth - 0.1, m_cellHeight - 0.1, color);
                }
            }
        }
    }
    // Updated (Grid)
    else
    {
        for (int j = 0; j < m_grid->getHeight(); j++)
        {
            for (int i = 0; i < m_grid->getWidth(); i++)
            {
                Qt::GlobalColor color = (m_grid->getAt(i, j)) ? Qt::green : Qt::darkGray;
                painter.fillRect(i * m_cellWidth, j * m_cellHeight, m_cellWidth - 0.1, m_cellHeight - 0.1, color);
            }
        }
    }
}

void Canvas::resizeEvent(QResizeEvent* event)
{
    Q_UNUSED(event);
    m_cellWidth = this->width() / m_gridWidth;
    m_cellHeight = this->height() / m_gridHeight;
}

void Canvas::receiveGridReader(GridReader *gridReader)
{
    m_grid = nullptr;
    m_data = gridReader->getData();

    m_gridWidth = gridReader->getDetectedWidth();
    m_gridHeight = gridReader->getDetectedHeight();

    m_cellWidth = this->width() / m_gridWidth;
    m_cellHeight = this->height() / m_gridHeight;

    out << "Received grid has (width, height) = (" << m_gridWidth << ", " << m_gridHeight << ")" << endl;

    update();
}

void Canvas::receiveGrid(Grid *grid)
{
    m_grid = grid;

    update();
}
