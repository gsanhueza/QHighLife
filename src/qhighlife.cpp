#include "qhighlife.h"
#include "ui_qhighlife.h"

#include <chrono>

QHighLife::QHighLife(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::QHighLife),
    m_tutorial(new Tutorial),
    m_about(new About),
    m_model(nullptr),
    out(stdout)
{
    ui->setupUi(this);
    ui->statusbar->showMessage("Load your initial grid in File, select your desired implementation in Model, and Run it.");

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
        ui->actionRunImplementation->setDisabled(true);
        ui->actionRunStressTest->setDisabled(true);
    }
}

void QHighLife::loadCPUModelClicked()
{
    ui->statusbar->showMessage("CPU implementation loaded.");
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    emit sendGridReader(&m_gridreader);
    m_model = new CPUModel(m_gridreader.getDetectedWidth(), m_gridreader.getDetectedHeight());
    m_model->setLoadedGrid(m_gridreader.getData());

    ui->actionRunImplementation->setEnabled(true);
    ui->actionRunStressTest->setEnabled(true);
}

void QHighLife::loadCUDAModelClicked()
{
    ui->statusbar->showMessage("CUDA implementation loaded.");
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    emit sendGridReader(&m_gridreader);
    m_model = new CUDAModel(m_gridreader.getDetectedWidth(), m_gridreader.getDetectedHeight());
    m_model->setLoadedGrid(m_gridreader.getData());

    ui->actionRunImplementation->setEnabled(true);
    ui->actionRunStressTest->setEnabled(true);
}

void QHighLife::loadOpenCLModelClicked()
{
    ui->statusbar->showMessage("OpenCL implementation loaded.");
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    emit sendGridReader(&m_gridreader);
    m_model = new OpenCLModel(m_gridreader.getDetectedWidth(), m_gridreader.getDetectedHeight());
    m_model->setLoadedGrid(m_gridreader.getData());

    ui->actionRunImplementation->setEnabled(true);
    ui->actionRunStressTest->setEnabled(true);
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
    ui->statusbar->showMessage("Implementation has run.");
    m_model->run();

    emit sendGrid(m_model->getGrid());
}

void QHighLife::loadRunStressTestClicked()
{
    ui->statusbar->showMessage("Stress implementation is running.");

    std::chrono::time_point<std::chrono::high_resolution_clock> m_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end = m_start + std::chrono::seconds(10);
    int iterations = 0;

    // FIXME Quizá haya que cambiar esto a un m_model->runStress(), donde allí esté el timer...
    while (std::chrono::high_resolution_clock::now() < m_end)
    {
        m_model->run();
        ++iterations;
    }

    emit sendGrid(m_model->getGrid());
    out << "# of iterations: " << iterations << endl;

    ui->statusbar->showMessage("Stress implementation has run.");
}
