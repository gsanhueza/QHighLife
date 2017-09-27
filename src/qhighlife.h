#ifndef _QHIGHLIFE_H_
#define _QHIGHLIFE_H_

#include <QDesktopWidget>
#include <QMainWindow>
#include <QFileDialog>
#include "about.h"
#include "tutorial.h"
#include "model.h"
#include "cpumodel.h"
#include "cudamodel.h"
#include "openclmodel.h"
#include "gridreader.h"

/**
* @brief Namespace used by qhighlife.ui
*
*/
namespace Ui
{
    class QHighLife;
}

/**
* @brief QHighLife class. Contains the whole window, menu bar, inner widget and status bar.
*
*/
class QHighLife : public QMainWindow
{
    Q_OBJECT;

public:
    /**
    * @brief QHighLife class constructor.
    *
    * @param parent p_parent: Parent of the class. Used by Qt.
    */
    explicit QHighLife(QWidget *parent = nullptr);

    /**
    * @brief QHighLife class destructor.
    *
    */
    ~QHighLife();

signals:
    void sendGridReader(GridReader *gridReader);
    void sendGrid(Grid *grid);

public slots:

    /**
     * @brief Receiver of a Qt signal when the Help -> Tutorial action is clicked in the window.
     *
     */
    void loadTutorialClicked();

    /**
     * @brief Receiver of a Qt signal when the Help -> About action is clicked in the window.
     *
     */
    void loadAboutClicked();

    /**
     * @brief Receiver of a Qt signal when the File -> Load Grid action is clicked in the window.
     *
     */
    void loadGridClicked();

    /**
     * @brief Receiver of a Qt signal when the Model -> Load CPU Model action is clicked in the window.
     *
     */
    void loadCPUModelClicked();

    /**
     * @brief Receiver of a Qt signal when the Model -> Load CUDA Model action is clicked in the window.
     *
     */
    void loadCUDAModelClicked();

    /**
     * @brief Receiver of a Qt signal when the Model -> Load OpenCL Model action is clicked in the window.
     *
     */
    void loadOpenCLModelClicked();

    /**
     * @brief Receiver of a Qt signal when the Run -> Run implementation action is clicked in the window.
     *
     */
    void loadRunClicked();

    /**
     * @brief Receiver of a Qt signal when the Run -> Run stress test action is clicked in the window.
     *
     */
    void loadRunStressTestClicked();

private:
    Ui::QHighLife *ui;
    Tutorial *m_tutorial;
    About *m_about;
    Model *m_model;
    GridReader m_gridreader;
};

#endif
