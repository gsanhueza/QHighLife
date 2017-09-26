#ifndef _CANVAS_H_
#define _CANVAS_H_

#include <QWidget>
#include <QVector>
#include "grid.h"
#include "gridreader.h"

class Canvas : public QWidget
{
    Q_OBJECT;

public:
    Canvas(QWidget* parent = nullptr);
    ~Canvas();

public slots:
    void receiveGridReader(GridReader *gridReader);
    void receiveGrid(Grid *grid);

protected:
    void paintEvent(QPaintEvent *e) override;

private:
    void doPainting();

    QVector<QString> m_data;
    Grid* m_grid;
    unsigned int m_gridWidth;
    unsigned int m_gridHeight;
    unsigned int m_cellWidth;
    unsigned int m_cellHeight;
};

#endif
