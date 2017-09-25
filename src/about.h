#ifndef _ABOUT_H_
#define _ABOUT_H_

#include <QWidget>
#include <QDesktopWidget>

namespace Ui
{
    class About;
}

class About : public QWidget
{
    Q_OBJECT

public:
    explicit About(QWidget *parent = nullptr);
    ~About();

private:
    Ui::About* ui;
};

#endif // _ABOUT_H_
