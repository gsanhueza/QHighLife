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
    ui->statusbar->showMessage("Load your initial grid in the File menu, select your desired implementation in Model, and run it.");

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

void QHighLife::loadGridClicked()
{
    QString filepath = QFileDialog::getOpenFileName(this, tr("Grid files"), ".", tr("Grid Files (.grid) (*.grid)"));

    if (m_gridreader.loadFile(filepath))
    {
        ui->statusbar->showMessage("File loaded.");
        ui->actionLoadCPUModel->setEnabled(true);
        ui->actionLoadCUDAModel->setEnabled(true);
        ui->actionLoadOpenCLModel->setEnabled(true);

        emit sendGridReader(&m_gridreader);
    }
    else
    {
        ui->statusbar->showMessage("Cannot load file.");
        ui->actionLoadCPUModel->setDisabled(true);
        ui->actionLoadCUDAModel->setDisabled(true);
        ui->actionLoadOpenCLModel->setDisabled(true);
    }
}

void QHighLife::loadCPUModelClicked()
{
    ui->statusbar->showMessage("TODO: CPU implementation loaded.");
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    m_model = new CPUModel(m_gridreader.getDetectedWidth(), m_gridreader.getDetectedHeight());
    m_model->setLoadedGrid(m_gridreader.getData());
}

void QHighLife::loadCUDAModelClicked()
{
    ui->statusbar->showMessage("TODO: CUDA implementation loaded.");
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    m_model = new CUDAModel(m_gridreader.getDetectedWidth(), m_gridreader.getDetectedHeight());
    m_model->setLoadedGrid(m_gridreader.getData());
}

void QHighLife::loadOpenCLModelClicked()
{
    ui->statusbar->showMessage("TODO: OpenCL implementation loaded.");
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    m_model = new OpenCLModel(m_gridreader.getDetectedWidth(), m_gridreader.getDetectedHeight());
    m_model->setLoadedGrid(m_gridreader.getData());
}

void QHighLife::loadTutorialClicked()
{
    m_tutorial->show();
}

void QHighLife::loadAboutClicked()
{
    m_about->show();
}

void QHighLife::loadRunClicked()
{
    ui->statusbar->showMessage("TODO: Run implementation");
    m_model->run();
}
