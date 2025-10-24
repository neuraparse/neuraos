/*
 * NeuralOS AI Dashboard
 * Qt5-based visual interface for AI inference monitoring
 */

#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QListWidget>
#include <QProgressBar>
#include <QTimer>
#include <QChart>
#include <QChartView>
#include <QLineSeries>
#include <QValueAxis>
#include <QDateTime>
#include <QFont>
#include <QPixmap>
#include <QPainter>
#include <QGradient>

extern "C" {
#include "../npie/include/npie.h"
}

using namespace QtCharts;

class NeuralOSDashboard : public QMainWindow {
    Q_OBJECT

public:
    NeuralOSDashboard(QWidget *parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("NeuralOS AI Dashboard v1.0.0-alpha");
        setMinimumSize(1024, 600);
        
        // Initialize NPIE
        npie_init_options_t options = {0};
        options.log_level = NPIE_LOG_INFO;
        npie_init(&options);
        
        setupUI();
        setupCharts();
        setupTimers();
        
        // Apply dark theme
        applyDarkTheme();
    }
    
    ~NeuralOSDashboard() {
        npie_shutdown();
    }

private:
    void setupUI() {
        QWidget *centralWidget = new QWidget(this);
        QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
        
        // Header
        QLabel *header = new QLabel("🧠 NeuralOS AI Dashboard");
        QFont headerFont;
        headerFont.setPointSize(24);
        headerFont.setBold(true);
        header->setFont(headerFont);
        header->setAlignment(Qt::AlignCenter);
        header->setStyleSheet("color: #00D9FF; padding: 20px;");
        mainLayout->addWidget(header);
        
        // Status bar
        QHBoxLayout *statusLayout = new QHBoxLayout();
        
        systemStatusLabel = new QLabel("System: Online");
        systemStatusLabel->setStyleSheet("color: #00FF00; font-size: 14px; padding: 10px;");
        statusLayout->addWidget(systemStatusLabel);
        
        npieVersionLabel = new QLabel(QString("NPIE: %1").arg(npie_version_string()));
        npieVersionLabel->setStyleSheet("color: #FFFFFF; font-size: 14px; padding: 10px;");
        statusLayout->addWidget(npieVersionLabel);
        
        statusLayout->addStretch();
        mainLayout->addLayout(statusLayout);
        
        // Main content area
        QHBoxLayout *contentLayout = new QHBoxLayout();
        
        // Left panel - Model list
        QVBoxLayout *leftPanel = new QVBoxLayout();
        QLabel *modelsLabel = new QLabel("AI Models");
        modelsLabel->setStyleSheet("font-size: 16px; font-weight: bold; color: #00D9FF;");
        leftPanel->addWidget(modelsLabel);
        
        modelList = new QListWidget();
        modelList->setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #00D9FF;");
        modelList->addItem("📊 Image Classification");
        modelList->addItem("🎯 Object Detection");
        modelList->addItem("📝 Text Analysis");
        modelList->addItem("🔊 Audio Processing");
        leftPanel->addWidget(modelList);
        
        QPushButton *loadModelBtn = new QPushButton("Load Model");
        loadModelBtn->setStyleSheet("background-color: #00D9FF; color: #000000; font-weight: bold; padding: 10px;");
        connect(loadModelBtn, &QPushButton::clicked, this, &NeuralOSDashboard::onLoadModel);
        leftPanel->addWidget(loadModelBtn);
        
        contentLayout->addLayout(leftPanel, 1);
        
        // Right panel - Performance charts
        QVBoxLayout *rightPanel = new QVBoxLayout();
        
        // Inference time chart
        QLabel *perfLabel = new QLabel("Inference Performance");
        perfLabel->setStyleSheet("font-size: 16px; font-weight: bold; color: #00D9FF;");
        rightPanel->addWidget(perfLabel);
        
        inferenceChartView = new QChartView();
        inferenceChartView->setRenderHint(QPainter::Antialiasing);
        inferenceChartView->setMinimumHeight(200);
        rightPanel->addWidget(inferenceChartView);
        
        // Hardware utilization
        QLabel *hwLabel = new QLabel("Hardware Utilization");
        hwLabel->setStyleSheet("font-size: 16px; font-weight: bold; color: #00D9FF;");
        rightPanel->addWidget(hwLabel);
        
        cpuProgressBar = new QProgressBar();
        cpuProgressBar->setStyleSheet("QProgressBar { border: 1px solid #00D9FF; background-color: #1E1E1E; color: #FFFFFF; } QProgressBar::chunk { background-color: #00D9FF; }");
        cpuProgressBar->setFormat("CPU: %p%");
        rightPanel->addWidget(cpuProgressBar);
        
        memoryProgressBar = new QProgressBar();
        memoryProgressBar->setStyleSheet("QProgressBar { border: 1px solid #00FF00; background-color: #1E1E1E; color: #FFFFFF; } QProgressBar::chunk { background-color: #00FF00; }");
        memoryProgressBar->setFormat("Memory: %p%");
        rightPanel->addWidget(memoryProgressBar);
        
        npuProgressBar = new QProgressBar();
        npuProgressBar->setStyleSheet("QProgressBar { border: 1px solid #FF00FF; background-color: #1E1E1E; color: #FFFFFF; } QProgressBar::chunk { background-color: #FF00FF; }");
        npuProgressBar->setFormat("NPU: %p%");
        rightPanel->addWidget(npuProgressBar);
        
        // Accelerator info
        QLabel *accelLabel = new QLabel("Detected Accelerators");
        accelLabel->setStyleSheet("font-size: 16px; font-weight: bold; color: #00D9FF; margin-top: 20px;");
        rightPanel->addWidget(accelLabel);
        
        acceleratorList = new QListWidget();
        acceleratorList->setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #00D9FF;");
        updateAcceleratorList();
        rightPanel->addWidget(acceleratorList);
        
        contentLayout->addLayout(rightPanel, 2);
        
        mainLayout->addLayout(contentLayout);
        
        // Footer
        QLabel *footer = new QLabel("NeuralOS v1.0.0-alpha | AI-Native Embedded Operating System");
        footer->setAlignment(Qt::AlignCenter);
        footer->setStyleSheet("color: #888888; padding: 10px;");
        mainLayout->addWidget(footer);
        
        setCentralWidget(centralWidget);
    }
    
    void setupCharts() {
        // Inference time chart
        inferenceChart = new QChart();
        inferenceChart->setTitle("Inference Time (ms)");
        inferenceChart->setTheme(QChart::ChartThemeDark);
        inferenceChart->setBackgroundBrush(QBrush(QColor("#1E1E1E")));
        
        inferenceSeries = new QLineSeries();
        inferenceSeries->setName("Latency");
        QPen pen(QColor("#00D9FF"));
        pen.setWidth(2);
        inferenceSeries->setPen(pen);
        
        inferenceChart->addSeries(inferenceSeries);
        
        QValueAxis *axisX = new QValueAxis();
        axisX->setTitleText("Time (s)");
        axisX->setLabelFormat("%d");
        axisX->setRange(0, 60);
        
        QValueAxis *axisY = new QValueAxis();
        axisY->setTitleText("Latency (ms)");
        axisY->setLabelFormat("%d");
        axisY->setRange(0, 100);
        
        inferenceChart->addAxis(axisX, Qt::AlignBottom);
        inferenceChart->addAxis(axisY, Qt::AlignLeft);
        inferenceSeries->attachAxis(axisX);
        inferenceSeries->attachAxis(axisY);
        
        inferenceChartView->setChart(inferenceChart);
    }
    
    void setupTimers() {
        // Update performance metrics every second
        QTimer *updateTimer = new QTimer(this);
        connect(updateTimer, &QTimer::timeout, this, &NeuralOSDashboard::updateMetrics);
        updateTimer->start(1000);
        
        // Simulate inference data
        dataTimer = new QTimer(this);
        connect(dataTimer, &QTimer::timeout, this, &NeuralOSDashboard::addInferenceData);
        dataTimer->start(500);
    }
    
    void applyDarkTheme() {
        setStyleSheet(
            "QMainWindow { background-color: #0D0D0D; }"
            "QWidget { background-color: #0D0D0D; color: #FFFFFF; }"
            "QLabel { color: #FFFFFF; }"
        );
    }
    
    void updateAcceleratorList() {
        acceleratorList->clear();
        
        uint32_t count = npie_get_accelerator_count();
        if (count == 0) {
            acceleratorList->addItem("⚠️  No hardware accelerators detected");
            acceleratorList->addItem("✓  CPU fallback available");
        } else {
            for (uint32_t i = 0; i < count; i++) {
                npie_accelerator_info_t info;
                if (npie_get_accelerator_info(i, &info) == NPIE_SUCCESS) {
                    QString item = QString("✓  %1 (%2)")
                        .arg(info.name)
                        .arg(info.type == NPIE_ACCELERATOR_NPU ? "NPU" : "GPU");
                    acceleratorList->addItem(item);
                }
            }
        }
    }

private slots:
    void updateMetrics() {
        // Simulate CPU/Memory/NPU usage
        static int cpuUsage = 0;
        static int memUsage = 0;
        static int npuUsage = 0;
        
        cpuUsage = (cpuUsage + (rand() % 20 - 10)) % 100;
        if (cpuUsage < 0) cpuUsage = 0;
        
        memUsage = (memUsage + (rand() % 10 - 5)) % 100;
        if (memUsage < 0) memUsage = 0;
        
        npuUsage = (npuUsage + (rand() % 30 - 15)) % 100;
        if (npuUsage < 0) npuUsage = 0;
        
        cpuProgressBar->setValue(cpuUsage);
        memoryProgressBar->setValue(memUsage);
        npuProgressBar->setValue(npuUsage);
    }
    
    void addInferenceData() {
        static qreal time = 0;
        qreal latency = 10 + (rand() % 40); // Random latency 10-50ms
        
        inferenceSeries->append(time, latency);
        
        // Keep only last 60 seconds of data
        if (inferenceSeries->count() > 120) {
            inferenceSeries->remove(0);
        }
        
        time += 0.5;
        
        // Update X axis range
        if (time > 60) {
            QValueAxis *axisX = qobject_cast<QValueAxis*>(inferenceChart->axes(Qt::Horizontal).at(0));
            if (axisX) {
                axisX->setRange(time - 60, time);
            }
        }
    }
    
    void onLoadModel() {
        QListWidgetItem *item = modelList->currentItem();
        if (item) {
            systemStatusLabel->setText(QString("Loading: %1...").arg(item->text()));
            QTimer::singleShot(1000, this, [this, item]() {
                systemStatusLabel->setText(QString("Loaded: %1").arg(item->text()));
            });
        }
    }

private:
    QLabel *systemStatusLabel;
    QLabel *npieVersionLabel;
    QListWidget *modelList;
    QListWidget *acceleratorList;
    QProgressBar *cpuProgressBar;
    QProgressBar *memoryProgressBar;
    QProgressBar *npuProgressBar;
    QChartView *inferenceChartView;
    QChart *inferenceChart;
    QLineSeries *inferenceSeries;
    QTimer *dataTimer;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    NeuralOSDashboard dashboard;
    dashboard.showFullScreen();
    
    return app.exec();
}

#include "main.moc"

