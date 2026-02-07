#ifndef NPIEBRIDGE_H
#define NPIEBRIDGE_H

#include <QObject>
#include <QStringList>

class NPIEBridge : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString version READ version CONSTANT)
    Q_PROPERTY(QString currentBackend READ currentBackend NOTIFY backendChanged)
    Q_PROPERTY(QStringList backends READ backends CONSTANT)
    Q_PROPERTY(bool modelLoaded READ modelLoaded NOTIFY modelChanged)
    Q_PROPERTY(QString modelName READ modelName NOTIFY modelChanged)
    Q_PROPERTY(quint64 inferenceCount READ inferenceCount NOTIFY statsUpdated)
    Q_PROPERTY(double lastInferenceMs READ lastInferenceMs NOTIFY statsUpdated)
    Q_PROPERTY(double avgInferenceMs READ avgInferenceMs NOTIFY statsUpdated)

public:
    explicit NPIEBridge(QObject *parent = nullptr);
    ~NPIEBridge();

    QString version() const;
    QString currentBackend() const;
    QStringList backends() const;
    bool modelLoaded() const;
    QString modelName() const;
    quint64 inferenceCount() const;
    double lastInferenceMs() const;
    double avgInferenceMs() const;

    Q_INVOKABLE bool loadModel(const QString &path);
    Q_INVOKABLE void unloadModel();
    Q_INVOKABLE double runInference();
    Q_INVOKABLE bool setBackend(const QString &name);

signals:
    void backendChanged();
    void modelChanged();
    void statsUpdated();

private:
    void *m_ctx = nullptr;
    QString m_currentBackend = "auto";
    bool m_modelLoaded = false;
    QString m_modelName;
    quint64 m_inferenceCount = 0;
    double m_lastInferenceMs = 0;
    double m_totalInferenceMs = 0;
};

#endif
