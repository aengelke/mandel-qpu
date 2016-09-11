CFLAGS=-g -O2 -DNUM_QPUS=12
LDLIBS=-ldl -lnetcdf
OBJ=qpu_mandel.o mailbox.o lqpu.o

.PHONY: all
all: qpu_mandel

qpu_mandel.o: qpu_mandel.c gpu_code.hex
qpu_mandel: $(OBJ)

.PHONY: clean
clean:
	$(RM) $(OBJ)
	$(RM) qpu_mandel
