/**
 * @file npi_init.c
 * @brief NeuralOS Init System (npi) - Fast boot init for embedded AI systems
 * @version 1.0.0-alpha
 * @date October 2025
 * 
 * NPI is a lightweight init system optimized for fast boot times
 * and AI workload management. It replaces traditional init systems
 * with a minimal, purpose-built solution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <time.h>

/* Platform-specific includes */
#ifdef __linux__
#include <sys/mount.h>
#include <sys/reboot.h>
#endif

#define NPI_VERSION "1.0.0-alpha"
#define NPI_CONFIG_FILE "/etc/npi/init.conf"
#define NPI_SERVICE_DIR "/etc/npi/services"
#define NPI_LOG_FILE "/var/log/npi.log"

/* Service states */
typedef enum {
    SERVICE_STOPPED = 0,
    SERVICE_STARTING,
    SERVICE_RUNNING,
    SERVICE_STOPPING,
    SERVICE_FAILED
} service_state_t;

/* Service structure */
typedef struct service {
    char name[64];
    char exec[256];
    char depends[256];
    int priority;
    pid_t pid;
    service_state_t state;
    int restart_count;
    time_t start_time;
    struct service* next;
} service_t;

/* Global variables */
static service_t* services = NULL;
static int running = 1;
static FILE* log_file = NULL;

/**
 * @brief Logging function
 */
static void npi_log(const char* level, const char* format, ...) {
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
    
    va_list args;
    va_start(args, format);
    
    /* Log to file */
    if (log_file) {
        fprintf(log_file, "[%s] [%s] ", timestamp, level);
        vfprintf(log_file, format, args);
        fprintf(log_file, "\n");
        fflush(log_file);
    }
    
    /* Also log to console */
    printf("[%s] [%s] ", timestamp, level);
    vprintf(format, args);
    printf("\n");
    
    va_end(args);
}

/**
 * @brief Mount essential filesystems
 */
static int mount_filesystems(void) {
#ifdef __linux__
    npi_log("INFO", "Mounting essential filesystems");

    /* Create mount points */
    mkdir("/proc", 0755);
    mkdir("/sys", 0755);
    mkdir("/dev", 0755);
    mkdir("/tmp", 0755);
    mkdir("/run", 0755);

    /* Mount proc */
    if (mount("proc", "/proc", "proc", 0, NULL) != 0) {
        npi_log("ERROR", "Failed to mount /proc: %s", strerror(errno));
        return -1;
    }

    /* Mount sysfs */
    if (mount("sysfs", "/sys", "sysfs", 0, NULL) != 0) {
        npi_log("ERROR", "Failed to mount /sys: %s", strerror(errno));
        return -1;
    }

    /* Mount devtmpfs */
    if (mount("devtmpfs", "/dev", "devtmpfs", 0, NULL) != 0) {
        npi_log("WARN", "Failed to mount /dev: %s", strerror(errno));
    }

    /* Mount tmpfs for /tmp */
    if (mount("tmpfs", "/tmp", "tmpfs", 0, "size=64M") != 0) {
        npi_log("WARN", "Failed to mount /tmp: %s", strerror(errno));
    }

    /* Mount tmpfs for /run */
    if (mount("tmpfs", "/run", "/tmpfs", 0, "size=32M") != 0) {
        npi_log("WARN", "Failed to mount /run: %s", strerror(errno));
    }

    npi_log("INFO", "Filesystems mounted successfully");
#else
    npi_log("INFO", "Filesystem mounting skipped (not on Linux)");
#endif
    return 0;
}

/**
 * @brief Set hostname
 */
static void set_hostname(void) {
    FILE* fp = fopen("/etc/hostname", "r");
    if (fp) {
        char hostname[256];
        if (fgets(hostname, sizeof(hostname), fp)) {
            hostname[strcspn(hostname, "\n")] = 0;
            if (sethostname(hostname, strlen(hostname)) == 0) {
                npi_log("INFO", "Hostname set to: %s", hostname);
            }
        }
        fclose(fp);
    }
}

/**
 * @brief Load service configuration
 */
static __attribute__((unused)) service_t* load_service(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        return NULL;
    }
    
    service_t* svc = (service_t*)calloc(1, sizeof(service_t));
    if (!svc) {
        fclose(fp);
        return NULL;
    }
    
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        /* Skip comments and empty lines */
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        
        /* Parse key=value */
        char* eq = strchr(line, '=');
        if (!eq) continue;
        
        *eq = '\0';
        char* key = line;
        char* value = eq + 1;
        
        /* Trim whitespace */
        while (*value == ' ' || *value == '\t') value++;
        value[strcspn(value, "\n")] = 0;
        
        if (strcmp(key, "name") == 0) {
            strncpy(svc->name, value, sizeof(svc->name) - 1);
        } else if (strcmp(key, "exec") == 0) {
            strncpy(svc->exec, value, sizeof(svc->exec) - 1);
        } else if (strcmp(key, "depends") == 0) {
            strncpy(svc->depends, value, sizeof(svc->depends) - 1);
        } else if (strcmp(key, "priority") == 0) {
            svc->priority = atoi(value);
        }
    }
    
    fclose(fp);
    
    svc->state = SERVICE_STOPPED;
    svc->pid = 0;
    svc->restart_count = 0;
    
    return svc;
}

/**
 * @brief Load all services
 */
static int load_services(void) {
    npi_log("INFO", "Loading services from %s", NPI_SERVICE_DIR);
    
    /* For simplicity, we'll define services inline */
    /* In production, these would be loaded from config files */
    
    /* Service 1: syslogd */
    service_t* syslog = (service_t*)calloc(1, sizeof(service_t));
    strcpy(syslog->name, "syslogd");
    strcpy(syslog->exec, "/sbin/syslogd -n");
    syslog->priority = 10;
    syslog->next = services;
    services = syslog;
    
    /* Service 2: klogd */
    service_t* klog = (service_t*)calloc(1, sizeof(service_t));
    strcpy(klog->name, "klogd");
    strcpy(klog->exec, "/sbin/klogd -n");
    strcpy(klog->depends, "syslogd");
    klog->priority = 20;
    klog->next = services;
    services = klog;
    
    /* Service 3: NPIE daemon */
    service_t* npie = (service_t*)calloc(1, sizeof(service_t));
    strcpy(npie->name, "npie-daemon");
    strcpy(npie->exec, "/usr/bin/npie-daemon");
    npie->priority = 50;
    npie->next = services;
    services = npie;
    
    npi_log("INFO", "Loaded 3 services");
    return 0;
}

/**
 * @brief Start a service
 */
static int start_service(service_t* svc) {
    if (svc->state == SERVICE_RUNNING) {
        return 0;
    }
    
    npi_log("INFO", "Starting service: %s", svc->name);
    svc->state = SERVICE_STARTING;
    
    pid_t pid = fork();
    if (pid < 0) {
        npi_log("ERROR", "Failed to fork for service %s: %s", 
                svc->name, strerror(errno));
        svc->state = SERVICE_FAILED;
        return -1;
    }
    
    if (pid == 0) {
        /* Child process */
        /* Parse command and arguments */
        char* argv[32];
        int argc = 0;
        char* cmd = strdup(svc->exec);
        char* token = strtok(cmd, " ");
        while (token && argc < 31) {
            argv[argc++] = token;
            token = strtok(NULL, " ");
        }
        argv[argc] = NULL;
        
        /* Execute service */
        execvp(argv[0], argv);
        
        /* If we get here, exec failed */
        fprintf(stderr, "Failed to execute %s: %s\n", 
                argv[0], strerror(errno));
        exit(1);
    }
    
    /* Parent process */
    svc->pid = pid;
    svc->state = SERVICE_RUNNING;
    svc->start_time = time(NULL);
    
    npi_log("INFO", "Service %s started with PID %d", svc->name, pid);
    return 0;
}

/**
 * @brief Start all services
 */
static void start_services(void) {
    npi_log("INFO", "Starting services");
    
    /* Start services in priority order */
    for (int priority = 0; priority <= 100; priority += 10) {
        service_t* svc = services;
        while (svc) {
            if (svc->priority == priority) {
                start_service(svc);
                /* Small delay between services */
                usleep(100000);
            }
            svc = svc->next;
        }
    }
}

/**
 * @brief Signal handler
 */
static void signal_handler(int sig) {
    if (sig == SIGCHLD) {
        /* Child process terminated */
        int status;
        pid_t pid;
        while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
            /* Find service by PID */
            service_t* svc = services;
            while (svc) {
                if (svc->pid == pid) {
                    if (WIFEXITED(status)) {
                        npi_log("INFO", "Service %s exited with code %d", 
                                svc->name, WEXITSTATUS(status));
                    } else if (WIFSIGNALED(status)) {
                        npi_log("WARN", "Service %s killed by signal %d", 
                                svc->name, WTERMSIG(status));
                    }
                    svc->state = SERVICE_STOPPED;
                    svc->pid = 0;
                    break;
                }
                svc = svc->next;
            }
        }
    } else if (sig == SIGTERM || sig == SIGINT) {
        npi_log("INFO", "Received shutdown signal");
        running = 0;
    }
}

/**
 * @brief Main init function
 */
int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    /* Check if we're PID 1 */
    if (getpid() != 1) {
        fprintf(stderr, "NPI must be run as PID 1 (init)\n");
        return 1;
    }
    
    /* Open log file */
    mkdir("/var/log", 0755);
    log_file = fopen(NPI_LOG_FILE, "a");
    
    npi_log("INFO", "NeuralOS Init System (npi) version %s", NPI_VERSION);
    npi_log("INFO", "Starting system initialization");
    
    /* Mount filesystems */
    if (mount_filesystems() != 0) {
        npi_log("ERROR", "Failed to mount filesystems");
        return 1;
    }
    
    /* Set hostname */
    set_hostname();
    
    /* Setup signal handlers */
    signal(SIGCHLD, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);
    
    /* Load services */
    load_services();
    
    /* Start services */
    start_services();
    
    npi_log("INFO", "System initialization complete");
    npi_log("INFO", "Boot time: %ld seconds", time(NULL));
    
    /* Main loop */
    while (running) {
        sleep(1);
        
        /* Monitor services and restart if needed */
        service_t* svc = services;
        while (svc) {
            if (svc->state == SERVICE_STOPPED && svc->restart_count < 3) {
                npi_log("WARN", "Restarting service: %s", svc->name);
                svc->restart_count++;
                start_service(svc);
            }
            svc = svc->next;
        }
    }
    
    /* Shutdown */
    npi_log("INFO", "Shutting down system");
    
    /* Stop all services */
    service_t* svc = services;
    while (svc) {
        if (svc->state == SERVICE_RUNNING) {
            npi_log("INFO", "Stopping service: %s", svc->name);
            kill(svc->pid, SIGTERM);
        }
        svc = svc->next;
    }
    
    /* Wait for services to stop */
    sleep(2);
    
    /* Unmount filesystems */
#ifdef __linux__
    umount("/tmp");
    umount("/run");
    umount("/dev");
    umount("/sys");
    umount("/proc");
#endif

    if (log_file) {
        fclose(log_file);
    }

    /* Reboot or halt */
    sync();
#ifdef __linux__
    reboot(RB_AUTOBOOT);
#endif
    
    return 0;
}

