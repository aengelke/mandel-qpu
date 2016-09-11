// Code partly taken from the GPU_FFT release 3.0 from
// http://www.aholme.co.uk/GPU_FFT/Main.htm

/*
BCM2835 "GPU_FFT" release 3.0
Copyright (c) 2015, Andrew Holme.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/



#include <dlfcn.h>

#include "lqpu.h"
#include "mailbox.h"


struct LQPUHostInfo {
    unsigned mem_flg;
    unsigned mem_map;
    unsigned peri_addr;
    unsigned peri_size;
};

typedef struct LQPUHostInfo LQPUHostInfo;

struct LQPUBase {
    int mb;

    unsigned handle;
    unsigned size;
    unsigned vcBase;

    volatile unsigned* peri;
    unsigned peri_size;
};


// Setting this define to zero on Pi 1 allows GPU_FFT and Open GL
// to co-exist and also improves performance of longer transforms:
#define LQPU_USE_VC4_L2_CACHE 1 // Pi 1 only: cached=1; direct=0
#define BUS_TO_PHYS(x) ((x)&~0xC0000000)

static
int
lqpu_get_host_info(LQPUHostInfo* info)
{
    // Pi 1 defaults
    info->peri_addr = 0x20000000;
    info->peri_size = 0x01000000;
    info->mem_flg = LQPU_USE_VC4_L2_CACHE ? 0xC : 0x4;
    info->mem_map = LQPU_USE_VC4_L2_CACHE ? 0x0 : 0x20000000; // Pi 1 only

    void* handle = dlopen("libbcm_host.so", RTLD_LAZY);
    if (!handle)
        return -1;

    unsigned (*bcm_host_get_sdram_address)(void) = dlsym(handle, "bcm_host_get_sdram_address");
    unsigned (*bcm_host_get_peripheral_address)(void) = dlsym(handle, "bcm_host_get_peripheral_address");
    unsigned (*bcm_host_get_peripheral_size)(void) = dlsym(handle, "bcm_host_get_peripheral_size");

    if (bcm_host_get_sdram_address && bcm_host_get_sdram_address() != 0x40000000) // Pi 2?
    {
        info->mem_flg = 0x4; // ARM cannot see VC4 L2 on Pi 2
        info->mem_map = 0x0;
    }

    if (bcm_host_get_peripheral_address)
        info->peri_addr = bcm_host_get_peripheral_address();
    if (bcm_host_get_peripheral_size)
        info->peri_size = bcm_host_get_peripheral_size();

    dlclose(handle);
    return 0;
}

unsigned
lqpu_ptr_add(LQPUPtr* ptr, int bytes)
{
    unsigned vc = ptr->vc;
    ptr->vc += bytes;
    ptr->arm.bptr += bytes;
    return vc;
}

unsigned
lqpu_alloc(int mb, unsigned size, LQPUBase** basePtr, LQPUPtr* ptr)
{
    LQPUHostInfo host;

    if (lqpu_get_host_info(&host))
        return 2;

    if (qpu_enable(mb, 1))
        return 1;

    size += sizeof(LQPUBase);

    // Allocate memory with 4k alignment
    unsigned handle = mem_alloc(mb, size, 0x1000, host.mem_flg);
    if (!handle)
    {
        qpu_enable(mb, 0);
        return 3;
    }

    volatile unsigned* peri = (volatile unsigned*) mapmem(host.peri_addr, host.peri_size);
    if (!peri)
    {
        mem_free(mb, handle);
        qpu_enable(mb, 0);
        return 4;
    }

    ptr->vc = mem_lock(mb, handle);
    ptr->arm.vptr = mapmem(BUS_TO_PHYS(ptr->vc + host.mem_map), size);

    LQPUBase* base = (LQPUBase*) ptr->arm.vptr;
    base->peri = peri;
    base->peri_size = host.peri_size;
    base->mb = mb;
    base->handle = handle;
    base->size = size;
    base->vcBase = ptr->vc;

    *basePtr = base;

    // Align this to 16 bytes as code is required to have this alignment.
    lqpu_ptr_add(ptr, (sizeof(LQPUBase) + 0xf) & ~0xf);

    return 0;
}


#define V3D_L2CACTL (0xC00020>>2)
#define V3D_SLCACTL (0xC00024>>2)
#define V3D_SRQPC   (0xC00430>>2)
#define V3D_SRQUA   (0xC00434>>2)
#define V3D_SRQCS   (0xC0043c>>2)
#define V3D_PCTRC   (0xC00670>>2)
#define V3D_PCTRE   (0xC00674>>2)
#define V3D_PCTR(n) ((0xC00680+8*n)>>2)
#define V3D_PCTRS(n) ((0xC00684+8*n)>>2)
#define V3D_DBCFG   (0xC00e00>>2)
#define V3D_DBQITE  (0xC00e2c>>2)
#define V3D_DBQITC  (0xC00e30>>2)


unsigned
lqpu_execute(LQPUBase* base, unsigned vc_msg, unsigned num_qpus)
{
    // Using the registers directly reduces the overhead by 0.1ms.
#if defined(LQPU_DIRECT_EXECUTION)
    LQPUMsg* msgs = (LQPUMsg*) ((unsigned) base + (vc_msg - base->vcBase));

    base->peri[V3D_DBCFG] = 0;
    base->peri[V3D_DBQITE] = 0;
    base->peri[V3D_DBQITC] = -1;
    base->peri[V3D_L2CACTL] = 1 << 2;
    base->peri[V3D_SLCACTL] = -1;
    base->peri[V3D_SRQCS] = (1 << 7) | (1 << 8) | (1 << 16);

    for (unsigned i = 0; i < num_qpus; i++)
    {
        base->peri[V3D_SRQUA] = msgs[i].vc_uniforms;
        base->peri[V3D_SRQPC] = msgs[i].vc_code;
    }

    for (;;)
    {
        if (((base->peri[V3D_SRQCS] >> 16) & 0xff) == num_qpus)
        {
            break;
        }
    }

    return 0;
#else
    // No flush, timeout after 2000 ms.
    return execute_qpu(base->mb, num_qpus, vc_msg, 1, 2000);
#endif
}

void
lqpu_release(LQPUBase* base)
{
    int mb = base->mb;
    unsigned handle = base->handle;
    unsigned size = base->size;
    unmapmem((void*) base->peri, base->peri_size);
    unmapmem(base, size);
    mem_unlock(mb, handle);
    mem_free(mb, handle);
    qpu_enable(mb, 0);
}

const char*
lqpu_status_name(unsigned status)
{
    switch (status)
    {
        case 0: return "Operation successful.\n";
        case 1: return "Unable to enable V3D. Please check your firmware is up to date.\n";
        case 2: return "Can't open libbcm_host.\n";
        case 3: return "Out of memory. Try a smaller batch or increase GPU memory.\n";
        case 4: return "Unable to map Videocore peripherals into ARM memory space.\n";
        case 0x80000000: return "Execution timed out.\n";
    }

    return "Unknown status.\n";
}
