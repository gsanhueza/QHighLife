/*
 * QHighLife is a High Life cellular-automata computing and visualization application using CPU and GPU.
 * Copyright (C) 2017  Gabriel Sanhueza <gabriel_8032@hotmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _QHIGHLIFE_H_
#define _QHIGHLIFE_H_

#include <QDesktopWidget>
#include <QMainWindow>
#include <QFileDialog>
#include <QTextStream>
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

    void keyPressEvent(QKeyEvent *event) override;

signals:
    /**
    * @brief Signal that emits a GridReader object when loading a .grid file.
    *
    * @param gridReader p_gridReader: GridReader.
    */
    void sendGridReader(GridReader *gridReader);

    /**
    * @brief Signal that emits a Grid object when an implementation has been
    * selected, or the grid has been updated.
    *
    * @param grid p_grid: Grid.
    */
    void sendGrid(Grid *grid);
    void keyPressed(QKeyEvent *event);

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
    * @brief Receiver of a Qt signal when the Run -> Run stress test -> Original Implementation action is clicked in the window.
    *
    */
    void loadRunStressTestClicked();

    /**
    * @brief Receiver of a Qt signal when the Run -> Run stress test -> Variant If action is clicked in the window.
    *
    */
    void loadRunStressTestClickedVariantIf();

    /**
    * @brief Receiver of a Qt signal when the Run -> Run stress test -> Variant Non If action is clicked in the window.
    *
    */
    void loadRunStressTestClickedVariantNonIf();

    /**
    * @brief Receiver of a Qt signal when the Run -> Run stress test -> Variant 32 action is clicked in the window.
    *
    */
    void loadRunStressTestClickedVariant32();

    /**
    * @brief Receiver of a Qt signal when the Run -> Run stress test -> Variant Non 32 action is clicked in the window.
    *
    */
    void loadRunStressTestClickedVariantNon32();

private:
    Ui::QHighLife *ui;
    Tutorial *m_tutorial;
    About *m_about;
    Model *m_model;
    GridReader m_gridreader;
    QTextStream out;
};

#endif
