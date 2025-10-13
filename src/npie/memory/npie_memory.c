/**
 * @file npie_memory.c
 * @brief NPIE Memory Manager Implementation
 * @version 1.0.0-alpha
 * 
 * Optimized memory management for AI inference with pooling and zero-copy
 */

#include "npie.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <pthread.h>

#define MEMORY_POOL_SIZE (64 * 1024 * 1024)  /* 64 MB default pool */
#define ALIGNMENT 64  /* 64-byte alignment for SIMD */

/**
 * @brief Memory block structure
 */
typedef struct memory_block {
    void* ptr;
    size_t size;
    bool in_use;
    struct memory_block* next;
} memory_block_t;

/**
 * @brief Memory pool structure
 */
typedef struct memory_pool {
    void* base_ptr;
    size_t total_size;
    size_t used_size;
    memory_block_t* blocks;
    pthread_mutex_t mutex;
    bool use_hugepages;
} memory_pool_t;

static memory_pool_t* g_memory_pool = NULL;

/**
 * @brief Align size to specified alignment
 */
static size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Initialize memory pool
 */
npie_status_t npie_memory_init(size_t pool_size, bool use_hugepages) {
    if (g_memory_pool) {
        return NPIE_ERROR_ALREADY_INITIALIZED;
    }
    
    if (pool_size == 0) {
        pool_size = MEMORY_POOL_SIZE;
    }
    
    g_memory_pool = (memory_pool_t*)calloc(1, sizeof(memory_pool_t));
    if (!g_memory_pool) {
        return NPIE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Allocate memory pool */
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (use_hugepages) {
        flags |= MAP_HUGETLB;
    }
    
    g_memory_pool->base_ptr = mmap(NULL, pool_size, 
                                   PROT_READ | PROT_WRITE,
                                   flags, -1, 0);
    
    if (g_memory_pool->base_ptr == MAP_FAILED) {
        /* Fallback to regular allocation if hugepages fail */
        g_memory_pool->base_ptr = mmap(NULL, pool_size,
                                       PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        
        if (g_memory_pool->base_ptr == MAP_FAILED) {
            free(g_memory_pool);
            g_memory_pool = NULL;
            return NPIE_ERROR_OUT_OF_MEMORY;
        }
        use_hugepages = false;
    }
    
    g_memory_pool->total_size = pool_size;
    g_memory_pool->used_size = 0;
    g_memory_pool->use_hugepages = use_hugepages;
    g_memory_pool->blocks = NULL;
    
    pthread_mutex_init(&g_memory_pool->mutex, NULL);
    
    return NPIE_SUCCESS;
}

/**
 * @brief Shutdown memory pool
 */
npie_status_t npie_memory_shutdown(void) {
    if (!g_memory_pool) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }
    
    pthread_mutex_lock(&g_memory_pool->mutex);
    
    /* Free all blocks */
    memory_block_t* block = g_memory_pool->blocks;
    while (block) {
        memory_block_t* next = block->next;
        free(block);
        block = next;
    }
    
    /* Unmap memory pool */
    munmap(g_memory_pool->base_ptr, g_memory_pool->total_size);
    
    pthread_mutex_unlock(&g_memory_pool->mutex);
    pthread_mutex_destroy(&g_memory_pool->mutex);
    
    free(g_memory_pool);
    g_memory_pool = NULL;
    
    return NPIE_SUCCESS;
}

/**
 * @brief Allocate memory from pool
 */
void* npie_memory_alloc(size_t size) {
    if (!g_memory_pool || size == 0) {
        return NULL;
    }
    
    /* Align size */
    size = align_size(size, ALIGNMENT);
    
    pthread_mutex_lock(&g_memory_pool->mutex);
    
    /* Try to find free block */
    memory_block_t* block = g_memory_pool->blocks;
    while (block) {
        if (!block->in_use && block->size >= size) {
            block->in_use = true;
            pthread_mutex_unlock(&g_memory_pool->mutex);
            return block->ptr;
        }
        block = block->next;
    }
    
    /* Allocate new block */
    if (g_memory_pool->used_size + size > g_memory_pool->total_size) {
        pthread_mutex_unlock(&g_memory_pool->mutex);
        return NULL; /* Pool exhausted */
    }
    
    block = (memory_block_t*)malloc(sizeof(memory_block_t));
    if (!block) {
        pthread_mutex_unlock(&g_memory_pool->mutex);
        return NULL;
    }
    
    block->ptr = (char*)g_memory_pool->base_ptr + g_memory_pool->used_size;
    block->size = size;
    block->in_use = true;
    block->next = g_memory_pool->blocks;
    g_memory_pool->blocks = block;
    
    g_memory_pool->used_size += size;
    
    pthread_mutex_unlock(&g_memory_pool->mutex);
    
    return block->ptr;
}

/**
 * @brief Free memory back to pool
 */
void npie_memory_free(void* ptr) {
    if (!g_memory_pool || !ptr) {
        return;
    }
    
    pthread_mutex_lock(&g_memory_pool->mutex);
    
    /* Find block */
    memory_block_t* block = g_memory_pool->blocks;
    while (block) {
        if (block->ptr == ptr) {
            block->in_use = false;
            break;
        }
        block = block->next;
    }
    
    pthread_mutex_unlock(&g_memory_pool->mutex);
}

/**
 * @brief Allocate aligned memory
 */
void* npie_memory_alloc_aligned(size_t size, size_t alignment) {
    if (!g_memory_pool || size == 0) {
        return NULL;
    }
    
    size = align_size(size, alignment);
    return npie_memory_alloc(size);
}

/**
 * @brief Get memory pool statistics
 */
npie_status_t npie_memory_get_stats(npie_memory_stats_t* stats) {
    if (!g_memory_pool || !stats) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&g_memory_pool->mutex);
    
    stats->total_size = g_memory_pool->total_size;
    stats->used_size = g_memory_pool->used_size;
    stats->free_size = g_memory_pool->total_size - g_memory_pool->used_size;
    stats->use_hugepages = g_memory_pool->use_hugepages;
    
    /* Count blocks */
    stats->num_blocks = 0;
    stats->num_free_blocks = 0;
    
    memory_block_t* block = g_memory_pool->blocks;
    while (block) {
        stats->num_blocks++;
        if (!block->in_use) {
            stats->num_free_blocks++;
        }
        block = block->next;
    }
    
    pthread_mutex_unlock(&g_memory_pool->mutex);
    
    return NPIE_SUCCESS;
}

/**
 * @brief Reset memory pool (mark all blocks as free)
 */
npie_status_t npie_memory_reset(void) {
    if (!g_memory_pool) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }
    
    pthread_mutex_lock(&g_memory_pool->mutex);
    
    memory_block_t* block = g_memory_pool->blocks;
    while (block) {
        block->in_use = false;
        block = block->next;
    }
    
    pthread_mutex_unlock(&g_memory_pool->mutex);
    
    return NPIE_SUCCESS;
}

/**
 * @brief Copy memory with optimization
 */
void npie_memory_copy(void* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) {
        return;
    }
    
    /* Use optimized memcpy */
    memcpy(dst, src, size);
}

/**
 * @brief Zero-copy tensor data sharing
 */
npie_status_t npie_memory_share_tensor(npie_tensor_t* dst, const npie_tensor_t* src) {
    if (!dst || !src) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    /* Share data pointer (zero-copy) */
    memcpy(dst, src, sizeof(npie_tensor_t));
    
    /* Mark as shared (would need additional flag in real implementation) */
    
    return NPIE_SUCCESS;
}

