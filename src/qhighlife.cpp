#include "qhighlife.h"
#include "ui_qhighlife.h"

QHighLife::QHighLife(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::QHighLife),
    m_tutorial(new Tutorial),
    m_about(new About)
{
    ui->setupUi(this);
    ui->statusbar->showMessage("Select your desired implementation in the File menu.");

    int screenWidth = QApplication::desktop()->width();
    int screenHeight = QApplication::desktop()->height();

    int x = (screenWidth - this->width()) / 2;
    int y = (screenHeight - this->height()) / 2;
    this->move(x, y);
}

QHighLife::~QHighLife()
{
    delete m_tutorial;
    delete m_about;
    delete ui;
}

void QHighLife::loadTutorialClicked()
{
    m_tutorial->show();
}

void QHighLife::loadAboutClicked()
{
    m_about->show();
}
