
#ifndef LQPU_H
#define LQPQ_H

struct LQPUPtr {
    unsigned vc;
    union {
        void* vptr;
        char* bptr;
        float* fptr;
        unsigned* uptr;
    } arm;
};

typedef struct LQPUPtr LQPUPtr;

struct LQPUBase;

typedef struct LQPUBase LQPUBase;

struct LQPUMsg {
    unsigned vc_uniforms;
    unsigned vc_code;
} __attribute__((packed));

typedef struct LQPUMsg LQPUMsg;


unsigned lqpu_ptr_add(LQPUPtr*, int bytes);
unsigned lqpu_alloc(int mb, unsigned size, LQPUBase** baseOut, LQPUPtr* ptrOut);
unsigned lqpu_execute(LQPUBase* base, unsigned vc_msg, unsigned num_qpus);
void lqpu_release(LQPUBase* base);

void lqpu_stats_enable(LQPUBase* base);
void lqpu_stats_print(LQPUBase* base);

const char* lqpu_status_name(unsigned status);

#endif // LQPU_H
