# Copyright (c) 2016, Alexis Engelke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


#
# Possible optimizations:
# - Buffer some blocks in VPM to reduce number of DMA calls
# - Support computation of multiple rows per kernel execution
#
# It should be noted that these optimizations have only little effect on the
# overall performance, as they are only useful once per element/line and as the
# resolution can be assumed to be reasonably small. (< 2k blocks per call)
#

#
# Naming convention.
#
#  line: | o o o o ... o o | o o o ... o o | ...
#          /\ element      \_______________/
#                         block = 16 elements
#
# A block is computed via the 16 SIMD lanes of one QPU. One QPU always processes
# one line. Multiple QPUs process *adjacent* lines, i.e. if QPU 0 computes line
# zero, QPU 1 must compute line 1, etc.. As there are 12 QPUs in total, at most
# 12 adjacent lines can be computed in parallel.
#

#
# Uniforms (in order).
#
# - qpuid (int) -- value in the range of 0, ..., NUM_QPUS - 1
# - debugAddr (ptr) -- currently not used
# - outputAddr (ptr)
# - stepSize (float)
# - re (float) - X coord of the first element of a line
# - im (float) - Y coord of the line of the QPU
# - maxIters (int)
# - maxValue (float)
# - blockCount (int) -- must be equal for all QPUs
#

.include <vc4.qinc>

# Must be equal to the number of QPUs started by the host processor and is used
# for synchronization. Alternatively, we *could* make this a uniform, but...
.set NUM_QPUS, 12

# Preparation for blocking rows in the VPM. Only value 1 makes currently sense.
# If we ever change this to a higher value, e.g. 4, we require the number of
# blocks to be a multiple of 4.
.set VPM_ROWS, 1

# Assignment of semaphores used for synchronization.
.set SEMA_COUNTER, 2
.set SEMA_SYNC_PRE, 3
.set SEMA_SYNC_POST, 4


#
# Register allocation.
#
# Registers suffixed with [] are restored for every block.
#      Register File A           Register File B
# 0     Re[=ReC]                  .
# 1     .                         Im[=ImC]
# 2     Iters[=MaxIters]          MaxIters
# 3     .                         .
# 4     Addend[=1]                .
# 5     MaxValue                  .
# 6     StepSize*16               StepSize
# 7     .                         .
# 8     .                         .
# 9     .                         .
# 10    ReC                       ImC
# 11    OutputAddr                .
# 12    .                         .
# 13    .                         BlocksLeft
# 14    .                         OutputShift
# 15    .                         .
# 16    .                         .
# 17    .                         .
# 18    .                         .
# 19    .                         .
# 20    .                         .
# 21    .                         .
# 22    .                         .
# 23    .                         .
# 24    .                         .
# 25    .                         .
# 26    PrcVpmWriteConfig         .
# 27    .                         .
# 28    PrcDmaStrideConfig        .
# 29    .                         .
# 30    .                         QpuId*VPM_ROWS
# 31    DebugAddr                 QpuId
#
# Registers 32-63 are not general-purpose registers and represent other hardware
# functions. The accumulators r0-r3 are used for different purposes.
#

mov r2, unif
mov ra31, unif


itof r1, elem_num; mov ra11, unif
# *r1 = elementIndex, *ra11 = output
# r2 is temporarily the QpuId, which is moved to rb31 in the next instruction

mov rb31, r2; mov r0, unif
# *r0 = stepSize, r1 = elementIndex@f

mov ra0, unif; mul24 rb30, r2, VPM_ROWS
# r0 = stepSize, r1 = elementIndex@f, *ra0 = re
# r3 is now QpuID*VPM_Rows, which is later moved to rb30

mov rb1, unif; fmul r1, r0, r1
# r0 = stepSize, r1 = elementIndex * stepSize, ra0 = re, *rb1 = im

mov rb2, unif; fmul ra6, r0, 16.0
# ra6 = 16.0 * stepSize
# r0 = stepSize, r1 = elementIndex * stepSize, ra0 = re, rb1 = im, *rb2 = maxIters

mov ra5, unif
fadd ra0, ra0, r1
# DROP r1 = elementIndex * stepSize
# r0 = stepSize, *ra0 = reStep, rb1 = im, *ra5 = maxValue, rb2 = maxIters

mov rb6, r0; mov ra2, rb2
# DROP r0 = stepSize
# ra0 = reStep, rb1 = im, *rb6 = stepSize, ra5 = maxValue, rb2 = maxIters, *ra2 = iters

mov ra4, 1
# ra0 = reStep, rb1 = im, rb6 = stepSize, ra5 = maxValue, rb2 = maxIters, ra2 = iters, *ra4 = addend

# Store reC and imC
mov ra10, ra0; mov rb10, rb1
# ra0 = reStep, rb1 = im, rb6 = stepSize, ra5 = maxValue, rb2 = maxIters, ra2 = iters, ra4 = addend, *ra10 = reC, *rb10 = imC

mov rb14, 64
# rb14 = outputShift

# Compute VPM configuration.  We always write one horizontal 32-bit line at the
# offset indicated by rb30=QpuId*VPM_ROWS.
mov r1, vpm_setup(1, 1, h32(0, 0))
or ra26, rb30, r1; mov r3, unif
# r3 = blockCount, later moved to rb13

# The DMA "basic configuration" is written where the DMA is actually used.

# Compute the DMA "stride configuration". The distance between two lines in
# the VPM is  (blockCount-VPM_ROWS)*64 bytes.
sub r0, r3, VPM_ROWS; mov rb13, r3
mov r1, vdw_setup_1(0)

# In fact, the loop begins here. The first three instructions are duplicated:
# here, and after the branch since three instructions after the branch are
# always executed.
#                            We start the computation loop on the MUL side
shl r2, r0, 6               ; fmul r0, ra0, ra0
or ra28, r2, r1             ; fmul r1, rb1, rb1

# r3 = abs, r0 = re2, r1 = im2, r2 = reim * 2
# r0 = re2, r1 = im2

fadd r3, r0, r1; fmul r2, ra0, rb1
# r0 = re2, r1 = im2, r2 = reim, r3 = re2 + im2



:loop

fsub.setf -, r3, ra5; fmul r2, r2, 2.0
# r0 = re2, r1 = im2, r2 = r * reim, flags = (re2 + im2) - maxValue

and ra4.nn, ra4, 0
# ra4 = addend & continue

brr.allnn -, r:loopExit
# No need for NOPs here, since we don't care about the next two instructions
# anyway. The third instruction is also not relevant since ra4 is known to be
# zero if it would be relevant.

# r3 is now re2 - im2
fsub r3, r0, r1
# r0 = re2, r1 = im2, r2 = 2 * reim, r3 = re2 - im2

# IM update: reim * 2 + imC, load addend to an accumulator
fadd rb1, r2, rb10; mov r0, ra4
# rb1 = 2 * reim + imC, r0 = addend

# Subtract iters = iters - addend
sub.setf ra2, ra2, r0

brr.allnz -, r:loop
# No NOPs here, we use the cycles. In the last iteration we don't care about
# the values anyway.

# We temporarily use r2 and write it back to the correct register ra0 in the
# next instruction for better interleaving of instructions as values written
# to an accumulator can be used in the next instruction while writes to the
# register file take one more instruction cycle.
fadd r2, r3, ra10; fmul r1, rb1, rb1
mov ra0, r2; fmul r0, r2, r2
fadd r3, r0, r1; fmul r2, r2, rb1
# ra0 = re2 - im2 + reC, r0 = re2, r1 = im2, r2 = reim, r3 = re2 + im2

:loopExit
###### END OF LOOP OVER ITERATIONS



mov.setf -, rb31; mov vw_setup, ra26
sub vpm, rb2, ra2
mov -, vw_wait


brr.allnz -, r:nowrite
# No NOPs here, we use the cycles which are executed by all QPUs
# Restore original re to r1, restore im from imC
mov r1, ra10; mov rb1, rb10
# Reset addend to 1 and move output address temporarily to r2
mov r2, ra11; mov ra4, 1
# Increment re = reC + stepSize * 16; move maxIters temporarily to r3
fadd ra0, r1, ra6; mov r3, rb2


# QPU 0 triggers DMA when all are done with the current block
srel -, SEMA_SYNC_PRE
.rep i, NUM_QPUS - 1
sacq -, SEMA_COUNTER
.endr
sacq -, SEMA_SYNC_PRE


# Write DMA configuration.  We first compute the "basic configuration", which
# indicates that we write  NUM_QPUS lines of length VPM_ROWS * 16  elements,
# starting at line 0. This split will allow to buffer a few lines in the VPM.
mov vw_setup, vdw_setup_0(NUM_QPUS, VPM_ROWS * 16, dma_h32(0, 0))

# Write "stride configuration" computed above.
mov vw_setup, ra28
mov vw_addr, ra11
mov -, vw_wait

# Allow other QPUs to continue working.
srel -, SEMA_SYNC_POST
.rep i, NUM_QPUS - 1
sacq -, SEMA_COUNTER
.endr
sacq -, SEMA_SYNC_POST

brr -, r:merge
nop
nop
nop


:nowrite
# If we are non-master, we await to barriers: the first to ensure that all
# QPUs are finished with computation and the second to ensure that the IO is
# finished. This *could* be relaxed to move the second barrier just before the
# write to the VPM, but...
sacq -, SEMA_SYNC_PRE
srel -, SEMA_SYNC_PRE
srel -, SEMA_COUNTER

sacq -, SEMA_SYNC_POST
srel -, SEMA_SYNC_POST
srel -, SEMA_COUNTER


:merge


# Reduce number of blocks to do by one (the addend ra4 is known to be 1 at this point)
# and restore iters
sub.setf rb13, rb13, ra4; mov ra2, r3
brr.anynz -, r:loop
# Also no NOPs here; we can use these cycles as well. The MUL part and the third
# instruction actually are the beginning of the next loop iteration.

# Update reC = re, r1 = im2
mov ra10, ra0; fmul r1, rb1, rb1
# Increment output address, r0 = re2
add ra11, r2, rb14; fmul r0, ra0, ra0
fadd r3, r0, r1; fmul r2, ra0, rb1

###### END OF LOOP OVER BLOCKS

# We don't need synchronization here since we already had a barrier after
# writing the data to memory.

# Send interrupt to finish
mov interrupt, 1
thrend
nop
nop
