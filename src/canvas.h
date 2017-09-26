#ifndef _CANVAS_H_
#define _CANVAS_H_

#include <QWidget>
#include <QVector>
#include <QStringRef>
#include "gridreader.h"

class Canvas : public QWidget
{
    Q_OBJECT;

public:
    Canvas(QWidget* parent = nullptr);
    ~Canvas();

public slots:
    void receiveGridReader(GridReader gridReader);

protected:
    void paintEvent(QPaintEvent *e) override;

private:
    void doPainting();
};

#endif
