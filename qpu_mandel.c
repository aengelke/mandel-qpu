/*
 * Copyright (c) 2016, Alexis Engelke
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/time.h>

#include "mailbox.h"
#include "lqpu.h"

#include <netcdf.h>


// Start Stop Timer.
#define _POSIX_C_SOURCE 200809L

#include <time.h>

struct JTimer {
  struct timespec timetaken_time;
  struct timespec current_time;
};
typedef struct JTimer JTimer;

void JTimerInit(JTimer*);
void JTimerCont(JTimer*);
void JTimerStop(JTimer*);
double JTimerRead(JTimer*);

void JTimerInit(JTimer* timer) {
  timer->timetaken_time.tv_sec = 0;
  timer->timetaken_time.tv_nsec = 0;
}

void JTimerCont(JTimer* timer) {
  clock_gettime(CLOCK_REALTIME, &(timer->current_time));
}

void JTimerStop(JTimer* timer) {
  struct timespec comp;
  clock_gettime(CLOCK_REALTIME, &comp);

  timer->timetaken_time.tv_sec += comp.tv_sec - timer->current_time.tv_sec;
  timer->timetaken_time.tv_nsec += comp.tv_nsec - timer->current_time.tv_nsec;
  if (timer->timetaken_time.tv_nsec < 0) {
    timer->timetaken_time.tv_nsec += 1000000000;
    timer->timetaken_time.tv_sec--;
  }
  else if (timer->timetaken_time.tv_nsec > 1000000000) {
    timer->timetaken_time.tv_nsec -= 1000000000;
    timer->timetaken_time.tv_sec++;
  }
}

double JTimerRead(JTimer* timer) {
  return timer->timetaken_time.tv_sec + timer->timetaken_time.tv_nsec * 1e-9;
}
// End timer.



static const unsigned gpu_code[] = {
#include "gpu_code.hex"
};

static const unsigned gpu_code_size = sizeof(gpu_code);

#ifndef NUM_QPUS
#error Number of QPUs is not defined!
#define NUM_QPUS 12
#endif

#define NC(inner) do { \
            int retval = (inner); \
            if (retval != 0) { \
                printf("NC Error (line %d): %d %s\n", __LINE__, retval, nc_strerror(retval)); \
                abort(); \
            } \
        } while (0)

struct Uniforms {
    unsigned qpuId;
    unsigned debugAddr;
    unsigned outputAddr;
    float stepSize;
    float re;
    float im;
    int maxIters;
    float maxValue;
    int columnBlocks;
} __attribute__((packed));

typedef struct Uniforms Uniforms;

struct MandelQPUData {
    int mb;
    LQPUBase* base;
    LQPUMsg* msg;
    unsigned vc_msg;
    unsigned* results;
    unsigned* debug;
    Uniforms* uniforms;

    int ncId;
    int ncDimX;
    int ncDimY;
    int ncVar;
};

typedef struct MandelQPUData MandelQPUData;

struct MandelArgs {
    float reMin;
    float reMax;
    float imMin;
    size_t width;
    size_t height;
    unsigned maxIter;
    float maxValue;
    const char* outFile;
    size_t rank;
    size_t nprocs;
};

typedef struct MandelArgs MandelArgs;


static
void
mandel_qpu_init(MandelQPUData* data, const MandelArgs* margs)
{
    if (margs->width % 16 != 0)
    {
        printf("Invalid width!");
        exit(-1);
    }
    if (margs->height % NUM_QPUS != 0)
    {
        printf("Invalid height!");
        exit(-1);
    }
    unsigned columns = margs->width / 16;

    int mb = mbox_open();

    LQPUPtr ptr;
    LQPUBase* base;

    size_t requiredSize = (((size_t) gpu_code_size + 0xf) & ~0xf) +
        NUM_QPUS * (sizeof(Uniforms) + sizeof(LQPUMsg) + sizeof(unsigned) * 16 * columns);

#ifdef GPU_DEBUG
    requiredSize += sizeof(unsigned) * NUM_QPUS * 16;
#endif

    unsigned ret = lqpu_alloc(mb, requiredSize, &base, &ptr);
    if (ret)
    {
        printf("%s", lqpu_status_name(ret));
        mbox_close(mb);
        exit(-1);
    }

    memcpy(ptr.arm.uptr, gpu_code, (size_t) gpu_code_size);
    unsigned vc_code = lqpu_ptr_add(&ptr, ((size_t) gpu_code_size + 0xf) & ~0xf);

    // Allocate memory for variables
    Uniforms* uniforms = ptr.arm.vptr;
    unsigned vc_uniforms = lqpu_ptr_add(&ptr, sizeof(Uniforms) * NUM_QPUS);

    LQPUMsg* msg = ptr.arm.vptr;
    unsigned vc_msg = lqpu_ptr_add(&ptr, sizeof(LQPUMsg) * NUM_QPUS);

    data->results = ptr.arm.uptr;
    unsigned vc_results = lqpu_ptr_add(&ptr, sizeof(unsigned) * NUM_QPUS * 16 * columns);

#ifdef GPU_DEBUG
    data->debug = ptr.arm.uptr;
    unsigned vc_debug = lqpu_ptr_add(&ptr, sizeof(unsigned) * NUM_QPUS * 16);
#endif

    for (int i = 0; i < NUM_QPUS; i++)
    {
        uniforms[i].qpuId = i;
#ifdef GPU_DEBUG
        uniforms[i].debugAddr = vc_debug + sizeof(unsigned) * i * 16;
#endif
        uniforms[i].outputAddr = vc_results + sizeof(unsigned) * i * 16 * columns;
        uniforms[i].stepSize = (margs->reMax - margs->reMin) / margs->width;
        uniforms[i].maxIters = margs->maxIter;
        uniforms[i].maxValue = margs->maxValue;
        uniforms[i].columnBlocks = columns;
        msg[i].vc_uniforms = vc_uniforms + sizeof(Uniforms) * i;
        msg[i].vc_code = vc_code;
    }

    data->uniforms = uniforms;
    data->vc_msg = vc_msg;
    data->base = base;
    data->mb = mb;
    data->msg = msg;

    if (margs->outFile != NULL)
    {
        NC(nc_create(margs->outFile, NC_NETCDF4, &(data->ncId)));
        NC(nc_def_dim(data->ncId, "x", margs->width, &(data->ncDimX)));
        NC(nc_def_dim(data->ncId, "y", margs->height, &(data->ncDimY)));

        int dimensions[] = { data->ncDimY, data->ncDimX };

        NC(nc_def_var(data->ncId, "mandelData", NC_UINT, 2, dimensions, &(data->ncVar)));
    }
    else
    {
        data->ncId = -1;
    }
}

static
void
mandel_qpu_fini(MandelQPUData* data)
{
    lqpu_release(data->base);
    mbox_close(data->mb);
    data->mb = -1;

    if (data->ncId != -1)
    {
        NC(nc_close(data->ncId));
    }
}

static
unsigned
mandel_qpu_lines(int yOffset, MandelQPUData* data, const MandelArgs* margs)
{
    Uniforms* uniforms = data->uniforms;
    for (int i = 0; i < NUM_QPUS; i++)
    {
        uniforms[i].re = margs->reMin;
        uniforms[i].im = margs->imMin + uniforms[i].stepSize * (i + NUM_QPUS * yOffset);
    }

#ifdef GPU_DEBUG
    memset(data->results, 0, sizeof(unsigned) * NUM_QPUS * margs->width);
    memset(data->debug, 0, sizeof(unsigned) * NUM_QPUS * 16);
#endif

    unsigned ret = lqpu_execute(data->base, data->vc_msg, NUM_QPUS);

#ifdef GPU_DEBUG
    for (int i = 0; i < NUM_QPUS; i++)
    {
        printf("QPU %03d %2d: ", yOffset, i);
        for (int j = 0; j < 16; j++)
        {
            printf(" %08x", data->debug[16 * i + j]);
            // printf(" %2.6f", ((float*) data->debug)[16 * i + j]);
        }
        printf("\n");
    }
#endif

    if (data->ncId != -1)
    {
        size_t start[] = { NUM_QPUS * yOffset, 0 };
        size_t count[] = { NUM_QPUS, margs->width };
        NC(nc_put_vara(data->ncId, data->ncVar, start, count, &(data->results[0])));
    }

    return ret;
}

static
void
parse_params(int argc, char** argv, MandelArgs* margs)
{
    if (argc < 8 || strcmp(argv[1], "-h") == 0) goto usage;
    if (sscanf(argv[1], "%f", &(margs->reMin)) != 1) goto usage;
    if (sscanf(argv[2], "%f", &(margs->reMax)) != 1) goto usage;
    if (sscanf(argv[3], "%f", &(margs->imMin)) != 1) goto usage;
    if (sscanf(argv[4], "%lu", &(margs->width)) != 1) goto usage;
    if (sscanf(argv[5], "%lu", &(margs->height)) != 1) goto usage;
    if (sscanf(argv[6], "%u", &(margs->maxIter)) != 1) goto usage;
    if (sscanf(argv[7], "%f", &(margs->maxValue)) != 1) goto usage;
    margs->outFile = argc >= 9 ? argv[8] : NULL;

    return;

usage:
    printf("usage: %s [reMin] [reMax] [imMin] [width] [height] [maxIter] [maxValue] [[out]]\n", argv[0]);
    exit(1);
}

int main(int argc, char **argv)
{
    unsigned error = 0;

    MandelArgs margs;
    MandelQPUData qpuData;

    JTimer compTimer;
    JTimerInit(&compTimer);

    parse_params(argc, argv, &margs);

    mandel_qpu_init(&qpuData, &margs);

    JTimerCont(&compTimer);
    for (unsigned i = 0; i < margs.height / NUM_QPUS && !error; i++)
    {
        error = mandel_qpu_lines(i, &qpuData, &margs);
    }
    JTimerStop(&compTimer);

    if (error)
    {
        printf("Error: %s\n", lqpu_status_name(error));
    }

    printf("Time: %f secs\n", JTimerRead(&compTimer));

    mandel_qpu_fini((&qpuData));
}
