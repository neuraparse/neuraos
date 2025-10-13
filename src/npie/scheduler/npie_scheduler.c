/**
 * @file npie_scheduler.c
 * @brief NPIE Inference Scheduler Implementation
 * @version 1.0.0-alpha
 * 
 * Real-time inference scheduler with priority-based task management
 */

#include "npie.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

#define MAX_QUEUE_SIZE 256
#define MAX_WORKERS 16

/**
 * @brief Inference task structure
 */
typedef struct inference_task {
    npie_model_t model;
    npie_tensor_t* inputs;
    uint32_t num_inputs;
    npie_tensor_t* outputs;
    uint32_t num_outputs;
    npie_callback_t callback;
    void* user_data;
    int priority;
    uint64_t submit_time;
    uint64_t deadline_us;
} inference_task_t;

/**
 * @brief Task queue structure
 */
typedef struct task_queue {
    inference_task_t tasks[MAX_QUEUE_SIZE];
    int head;
    int tail;
    int count;
    pthread_mutex_t mutex;
    sem_t sem;
} task_queue_t;

/**
 * @brief Scheduler structure
 */
typedef struct npie_scheduler {
    task_queue_t queue;
    pthread_t workers[MAX_WORKERS];
    int num_workers;
    bool running;
    npie_context_t context;
    
    /* Statistics */
    uint64_t tasks_submitted;
    uint64_t tasks_completed;
    uint64_t tasks_failed;
    uint64_t total_inference_time_us;
} npie_scheduler_t;

static npie_scheduler_t* g_scheduler = NULL;

/**
 * @brief Initialize task queue
 */
static void queue_init(task_queue_t* queue) {
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
    pthread_mutex_init(&queue->mutex, NULL);
    sem_init(&queue->sem, 0, 0);
}

/**
 * @brief Enqueue task (priority-based)
 */
static npie_status_t queue_enqueue(task_queue_t* queue, const inference_task_t* task) {
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->count >= MAX_QUEUE_SIZE) {
        pthread_mutex_unlock(&queue->mutex);
        return NPIE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Find insertion point based on priority */
    int insert_pos = queue->tail;
    
    /* Simple priority insertion (higher priority = lower number) */
    if (queue->count > 0) {
        int pos = queue->head;
        for (int i = 0; i < queue->count; i++) {
            if (task->priority < queue->tasks[pos].priority) {
                insert_pos = pos;
                break;
            }
            pos = (pos + 1) % MAX_QUEUE_SIZE;
        }
    }
    
    /* Insert task */
    memcpy(&queue->tasks[insert_pos], task, sizeof(inference_task_t));
    queue->tail = (queue->tail + 1) % MAX_QUEUE_SIZE;
    queue->count++;
    
    pthread_mutex_unlock(&queue->mutex);
    sem_post(&queue->sem);
    
    return NPIE_SUCCESS;
}

/**
 * @brief Dequeue task
 */
static npie_status_t queue_dequeue(task_queue_t* queue, inference_task_t* task) {
    sem_wait(&queue->sem);
    
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->count == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    memcpy(task, &queue->tasks[queue->head], sizeof(inference_task_t));
    queue->head = (queue->head + 1) % MAX_QUEUE_SIZE;
    queue->count--;
    
    pthread_mutex_unlock(&queue->mutex);
    
    return NPIE_SUCCESS;
}

/**
 * @brief Worker thread function
 */
static void* worker_thread(void* arg) {
    npie_scheduler_t* scheduler = (npie_scheduler_t*)arg;
    
    while (scheduler->running) {
        inference_task_t task;
        
        if (queue_dequeue(&scheduler->queue, &task) != NPIE_SUCCESS) {
            continue;
        }
        
        /* Check deadline */
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t current_time = ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
        
        if (task.deadline_us > 0 && current_time > task.deadline_us) {
            /* Task missed deadline */
            if (task.callback) {
                npie_metrics_t metrics = {0};
                task.callback(NPIE_ERROR_TIMEOUT, &metrics, task.user_data);
            }
            __sync_fetch_and_add(&scheduler->tasks_failed, 1);
            continue;
        }
        
        /* Run inference */
        npie_metrics_t metrics;
        npie_status_t status = npie_inference_run(
            task.model,
            task.inputs,
            task.num_inputs,
            task.outputs,
            task.num_outputs,
            &metrics
        );
        
        /* Update statistics */
        if (status == NPIE_SUCCESS) {
            __sync_fetch_and_add(&scheduler->tasks_completed, 1);
            __sync_fetch_and_add(&scheduler->total_inference_time_us, metrics.inference_time_us);
        } else {
            __sync_fetch_and_add(&scheduler->tasks_failed, 1);
        }
        
        /* Call callback */
        if (task.callback) {
            task.callback(status, &metrics, task.user_data);
        }
    }
    
    return NULL;
}

/**
 * @brief Initialize scheduler
 */
npie_status_t npie_scheduler_init(npie_context_t ctx, int num_workers) {
    if (g_scheduler) {
        return NPIE_ERROR_ALREADY_INITIALIZED;
    }
    
    if (num_workers <= 0 || num_workers > MAX_WORKERS) {
        num_workers = 4; /* Default */
    }
    
    g_scheduler = (npie_scheduler_t*)calloc(1, sizeof(npie_scheduler_t));
    if (!g_scheduler) {
        return NPIE_ERROR_OUT_OF_MEMORY;
    }
    
    g_scheduler->context = ctx;
    g_scheduler->num_workers = num_workers;
    g_scheduler->running = true;
    
    /* Initialize queue */
    queue_init(&g_scheduler->queue);
    
    /* Create worker threads */
    for (int i = 0; i < num_workers; i++) {
        pthread_create(&g_scheduler->workers[i], NULL, worker_thread, g_scheduler);
    }
    
    return NPIE_SUCCESS;
}

/**
 * @brief Shutdown scheduler
 */
npie_status_t npie_scheduler_shutdown(void) {
    if (!g_scheduler) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }
    
    /* Stop workers */
    g_scheduler->running = false;
    
    /* Wake up all workers */
    for (int i = 0; i < g_scheduler->num_workers; i++) {
        sem_post(&g_scheduler->queue.sem);
    }
    
    /* Wait for workers to finish */
    for (int i = 0; i < g_scheduler->num_workers; i++) {
        pthread_join(g_scheduler->workers[i], NULL);
    }
    
    /* Cleanup */
    pthread_mutex_destroy(&g_scheduler->queue.mutex);
    sem_destroy(&g_scheduler->queue.sem);
    
    free(g_scheduler);
    g_scheduler = NULL;
    
    return NPIE_SUCCESS;
}

/**
 * @brief Submit inference task to scheduler
 */
npie_status_t npie_scheduler_submit(npie_model_t model,
                                    const npie_tensor_t* inputs,
                                    uint32_t num_inputs,
                                    npie_tensor_t* outputs,
                                    uint32_t num_outputs,
                                    int priority,
                                    uint64_t deadline_us,
                                    npie_callback_t callback,
                                    void* user_data) {
    if (!g_scheduler) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }
    
    if (!model || !inputs || !outputs) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    /* Create task */
    inference_task_t task;
    task.model = model;
    task.inputs = (npie_tensor_t*)inputs;
    task.num_inputs = num_inputs;
    task.outputs = outputs;
    task.num_outputs = num_outputs;
    task.callback = callback;
    task.user_data = user_data;
    task.priority = priority;
    task.deadline_us = deadline_us;
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    task.submit_time = ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
    
    /* Enqueue task */
    npie_status_t status = queue_enqueue(&g_scheduler->queue, &task);
    if (status == NPIE_SUCCESS) {
        __sync_fetch_and_add(&g_scheduler->tasks_submitted, 1);
    }
    
    return status;
}

/**
 * @brief Get scheduler statistics
 */
npie_status_t npie_scheduler_get_stats(npie_scheduler_stats_t* stats) {
    if (!g_scheduler || !stats) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    stats->tasks_submitted = g_scheduler->tasks_submitted;
    stats->tasks_completed = g_scheduler->tasks_completed;
    stats->tasks_failed = g_scheduler->tasks_failed;
    stats->tasks_pending = g_scheduler->queue.count;
    stats->num_workers = g_scheduler->num_workers;
    
    if (g_scheduler->tasks_completed > 0) {
        stats->avg_inference_time_us = 
            g_scheduler->total_inference_time_us / g_scheduler->tasks_completed;
    } else {
        stats->avg_inference_time_us = 0;
    }
    
    return NPIE_SUCCESS;
}

