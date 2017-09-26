#ifndef _CANVAS_H_
#define _CANVAS_H_

#include <QWidget>

class Canvas : public QWidget
{
    Q_OBJECT;

public:
    Canvas(QWidget* parent = nullptr);
    ~Canvas();

protected:
    void paintEvent(QPaintEvent *e) override;

private:
    void doPainting();
};

#endif
