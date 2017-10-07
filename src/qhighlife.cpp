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
    ui->statusbar->showMessage(QString("Load your initial grid in File, select your desired implementation in Model, and Run it."));

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
        ui->statusbar->showMessage(QString("File loaded."));
        ui->actionLoadCPUModel->setEnabled(true);
        ui->actionLoadCUDAModel->setEnabled(true);
        ui->actionLoadOpenCLModel->setEnabled(true);
        ui->actionRunImplementation->setDisabled(true);

        ui->menuRunStressTest->setDisabled(true);
        ui->actionOriginalImplementation->setDisabled(true);
        ui->actionVariantIf->setDisabled(true);
        ui->actionVariantNonIf->setDisabled(true);
        ui->actionVariant32->setDisabled(true);
        ui->actionVariantNon32->setDisabled(true);

        emit sendGridReader(&m_gridreader);
    }
    else
    {
        ui->statusbar->showMessage(QString("Cannot load file."));
        ui->actionLoadCPUModel->setDisabled(true);
        ui->actionLoadCUDAModel->setDisabled(true);
        ui->actionLoadOpenCLModel->setDisabled(true);
        ui->actionRunImplementation->setDisabled(true);

        ui->menuRunStressTest->setDisabled(true);
        ui->actionOriginalImplementation->setDisabled(true);
        ui->actionVariantIf->setDisabled(true);
        ui->actionVariantNonIf->setDisabled(true);
        ui->actionVariant32->setDisabled(true);
        ui->actionVariantNon32->setDisabled(true);
    }
}

void QHighLife::loadCPUModelClicked()
{
    ui->statusbar->showMessage(QString("CPU implementation loaded."));
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    m_model = new CPUModel(m_gridreader.getDetectedWidth(), m_gridreader.getDetectedHeight());
    m_model->setLoadedGrid(m_gridreader.getData());
    m_model->setup();

    emit sendGrid(m_model->getGrid());

    ui->menuRunStressTest->setEnabled(true);
    ui->actionRunImplementation->setEnabled(true);
    ui->actionOriginalImplementation->setEnabled(true);
    ui->actionVariantIf->setDisabled(true);
    ui->actionVariantNonIf->setDisabled(true);
    ui->actionVariant32->setDisabled(true);
    ui->actionVariantNon32->setDisabled(true);
}

void QHighLife::loadCUDAModelClicked()
{
    ui->statusbar->showMessage(QString("CUDA implementation loaded."));
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    m_model = new CUDAModel(m_gridreader.getDetectedWidth(), m_gridreader.getDetectedHeight());
    m_model->setLoadedGrid(m_gridreader.getData());
    m_model->setup();

    emit sendGrid(m_model->getGrid());

    ui->menuRunStressTest->setEnabled(true);
    ui->actionRunImplementation->setEnabled(true);
    ui->actionOriginalImplementation->setEnabled(true);
    ui->actionVariantIf->setEnabled(true);
    ui->actionVariantNonIf->setEnabled(true);
    ui->actionVariant32->setEnabled(true);
    ui->actionVariantNon32->setEnabled(true);
}

void QHighLife::loadOpenCLModelClicked()
{
    ui->statusbar->showMessage(QString("OpenCL implementation loaded."));
    if (m_model != nullptr)
    {
        delete m_model;
        m_model = nullptr;
    }
    m_model = new OpenCLModel(m_gridreader.getDetectedWidth(), m_gridreader.getDetectedHeight());
    m_model->setLoadedGrid(m_gridreader.getData());
    m_model->setup();

    emit sendGrid(m_model->getGrid());

    ui->menuRunStressTest->setEnabled(true);
    ui->actionRunImplementation->setEnabled(true);
    ui->actionOriginalImplementation->setEnabled(true);
    ui->actionVariantIf->setEnabled(true);
    ui->actionVariantNonIf->setEnabled(true);
    ui->actionVariant32->setEnabled(true);
    ui->actionVariantNon32->setEnabled(true);
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
    ui->statusbar->showMessage(QString("Implementation has run."));
    m_model->run();

    emit sendGrid(m_model->getGrid());
}

void QHighLife::loadRunStressTestClicked()
{
    ui->statusbar->showMessage(QString("Stress implementation is running..."));

    int seconds = 10;
    int iterations = m_model->runStressTest(seconds);
    out << iterations << " iterations in " << seconds << " seconds." << endl;

    emit sendGrid(m_model->getGrid());

    ui->statusbar->showMessage(QString("Stress implementation has run: %1 iterations in %2 seconds.").arg(iterations).arg(seconds));
}

void QHighLife::loadRunStressTestClickedVariantIf()
{
    ui->statusbar->showMessage(QString("Stress implementation (variant \"if\") is running..."));

    int seconds = 10;
    int iterations = m_model->runStressTestVariantIf(seconds);
    out << iterations << " iterations in " << seconds << " seconds." << endl;

    emit sendGrid(m_model->getGrid());

    ui->statusbar->showMessage(QString("Stress implementation (variant \"if\") has run: %1 iterations in %2 seconds.").arg(iterations).arg(seconds));
}

void QHighLife::loadRunStressTestClickedVariantNonIf()
{
    ui->statusbar->showMessage(QString("Stress implementation (variant \"non if\") is running..."));

    int seconds = 10;
    int iterations = m_model->runStressTestVariantNonIf(seconds);
    out << iterations << " iterations in " << seconds << " seconds." << endl;

    emit sendGrid(m_model->getGrid());

    ui->statusbar->showMessage(QString("Stress implementation (variant \"non if\") has run: %1 iterations in %2 seconds.").arg(iterations).arg(seconds));
}

void QHighLife::loadRunStressTestClickedVariant32()
{
    ui->statusbar->showMessage(QString("Stress implementation (variant \"32\") is running..."));

    int seconds = 10;
    int iterations = m_model->runStressTestVariant32(seconds);
    out << iterations << " iterations in " << seconds << " seconds." << endl;

    emit sendGrid(m_model->getGrid());

    ui->statusbar->showMessage(QString("Stress implementation (variant \"32\") has run: %1 iterations in %2 seconds.").arg(iterations).arg(seconds));
}

void QHighLife::loadRunStressTestClickedVariantNon32()
{
    ui->statusbar->showMessage(QString("Stress implementation (variant \"non 32\") is running..."));

    int seconds = 10;
    int iterations = m_model->runStressTestVariantNon32(seconds);
    out << iterations << " iterations in " << seconds << " seconds." << endl;

    emit sendGrid(m_model->getGrid());

    ui->statusbar->showMessage(QString("Stress implementation (variant \"non 32\") has run: %1 iterations in %2 seconds.").arg(iterations).arg(seconds));
}
