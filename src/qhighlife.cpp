#include "qhighlife.h"
#include "ui_qhighlife.h"

QHighLife::QHighLife(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::QHighLife),
    m_tutorial(new Tutorial),
    m_about(new About),
    m_model(nullptr)
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

void QHighLife::loadCPUModelClicked()
{
    ui->statusbar->showMessage("TODO: CPU implementation loaded.");
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    m_model = new CPUModel(5, 3); // FIXME Necesito inicializar el modelo con algun tamaño de grilla

    m_model->run(); // FIXME Sacar de aqui
}

void QHighLife::loadCUDAModelClicked()
{
    ui->statusbar->showMessage("TODO: CUDA implementation loaded.");
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    m_model = new CUDAModel(5, 3);

    m_model->run();
}

void QHighLife::loadOpenCLModelClicked()
{
    ui->statusbar->showMessage("TODO: OpenCL implementation loaded.");
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    m_model = new OpenCLModel(5, 3);

    m_model->run();
}

void QHighLife::loadTutorialClicked()
{
    m_tutorial->show();
}

void QHighLife::loadAboutClicked()
{
    m_about->show();
}
