#include "npumonitor.h"
#include <QVariantMap>

extern "C" {
#include "npu_driver.h"
}

NPUMonitor::NPUMonitor(QObject *parent) : QObject(parent)
{
    npu_driver_init();
    m_initialized = true;

    connect(&m_timer, &QTimer::timeout, this, &NPUMonitor::refresh);
    m_timer.start(2000);
    refresh();
}

int NPUMonitor::deviceCount() const { return m_deviceCount; }
QVariantList NPUMonitor::devices() const { return m_devices; }
quint64 NPUMonitor::totalInferences() const { return m_totalInferences; }
quint64 NPUMonitor::totalTimeUs() const { return m_totalTimeUs; }
quint64 NPUMonitor::powerMw() const { return m_powerMw; }
quint32 NPUMonitor::frequencyMhz() const { return m_frequencyMhz; }
bool NPUMonitor::powerEnabled() const { return m_powerEnabled; }

void NPUMonitor::refresh()
{
    if (!m_initialized) return;

    npu_device_t devs[8];
    m_deviceCount = npu_detect_devices(devs, 8);

    m_devices.clear();
    for (int i = 0; i < m_deviceCount; i++) {
        npu_capabilities_t caps;
        if (npu_get_capabilities(devs[i], &caps) == 0) {
            QVariantMap dev;
            dev["name"] = QString(caps.name);
            dev["version"] = QString(caps.version);
            dev["cores"] = (int)caps.num_cores;
            dev["maxFreqMhz"] = (int)caps.max_frequency_mhz;
            dev["memoryMB"] = (qint64)(caps.memory_size / (1024 * 1024));
            dev["supportsInt8"] = caps.supports_int8;
            dev["supportsFloat16"] = caps.supports_float16;
            dev["supportsFloat32"] = caps.supports_float32;

            const char* typeNames[] = {
                "Unknown", "Edge TPU", "Ethos-U", "Hailo",
                "Rockchip", "Amlogic", "Custom", "Simulated"
            };
            int typeIdx = (int)caps.type;
            dev["type"] = (typeIdx >= 0 && typeIdx <= 7) ? typeNames[typeIdx] : "Unknown";
            m_devices.append(dev);
        }

        /* Get stats from first device */
        if (i == 0) {
            uint64_t inf = 0, time_us = 0, power = 0;
            npu_get_stats(devs[i], &inf, &time_us, &power);
            m_totalInferences = inf;
            m_totalTimeUs = time_us;
            m_powerMw = power;
        }
    }

    emit updated();
}

bool NPUMonitor::setPower(bool enabled)
{
    npu_device_t devs[8];
    int count = npu_detect_devices(devs, 8);
    if (count > 0) {
        int ret = npu_set_power_state(devs[0], enabled);
        if (ret == 0) {
            m_powerEnabled = enabled;
            refresh();
            return true;
        }
    }
    return false;
}

bool NPUMonitor::setFrequency(quint32 mhz)
{
    npu_device_t devs[8];
    int count = npu_detect_devices(devs, 8);
    if (count > 0) {
        int ret = npu_set_frequency(devs[0], mhz);
        if (ret == 0) {
            m_frequencyMhz = mhz;
            refresh();
            return true;
        }
    }
    return false;
}
