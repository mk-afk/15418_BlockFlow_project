# BlockFlow build. The defaults suit the GHC machines: one GPU each, MPI but no NCCL.
#   make gpu        train, sample, test_dit, test_comm with nvcc (gradient all-reduce through MPI)
#   make test-gpu   GPU mini-tests (needs python3 with torch for the test data); 4 ranks share the GPU
#   make emu        the same programs for CPU emulation of the CUDA kernels (no GPU needed)
#   make test       CPU-emulation mini-tests
# Overrides: USE_NCCL=1 (NCCL all-reduce, for multi-GPU nodes), USE_MPI=0 (single process only),
#            GPU_ARCH=sm_80|sm_86|sm_90|native (default sm_75, which also runs on newer GPUs),
#            CUDA_INC/CUDA_LIB/NCCL_INC/NCCL_LIB for non-standard install paths.

NVCC     ?= nvcc
CXX      ?= g++
MPICXX   ?= mpicxx
PYTHON   ?= python3
MPIRUN   ?= mpirun
GPU_ARCH ?= sm_75
USE_MPI  ?= 1
USE_NCCL ?= 0

HDRS  := $(wildcard csrc/*.h csrc/*.cuh)
NVFLAGS := -O3 -std=c++17 -arch=$(GPU_ARCH) -lineinfo $(if $(CUDA_INC),-I$(CUDA_INC)) $(if $(NCCL_INC),-I$(NCCL_INC))
NVLIBS  := $(if $(CUDA_LIB),-L$(CUDA_LIB)) $(if $(NCCL_LIB),-L$(NCCL_LIB)) -lcublas
ifeq ($(USE_MPI),1)
  NVPAR := -ccbin $(MPICXX) -DUSE_MPI $(if $(filter 1,$(USE_NCCL)),-DUSE_NCCL)
  NVPARLIBS := $(if $(filter 1,$(USE_NCCL)),-lnccl)
endif
EMUFLAGS := -std=c++17 -O2 -x c++ -DBLOCKFLOW_EMU -Wall -Wno-unused-function

gpu: build/gpu/train build/gpu/sample build/gpu/test_dit build/gpu/test_comm
emu: build/emu/train build/emu/sample build/emu/test_dit build/emu/test_comm

build/gpu/train build/gpu/test_comm: build/gpu/%: csrc/%.cu $(HDRS)
	@mkdir -p build/gpu
	$(NVCC) $(NVFLAGS) $(NVPAR) $< -o $@ $(NVLIBS) $(NVPARLIBS)
build/gpu/sample build/gpu/test_dit: build/gpu/%: csrc/%.cu $(HDRS)
	@mkdir -p build/gpu
	$(NVCC) $(NVFLAGS) $< -o $@ $(NVLIBS)
build/emu/train build/emu/test_comm: build/emu/%: csrc/%.cu $(HDRS)
	@mkdir -p build/emu
	$(MPICXX) $(EMUFLAGS) -DUSE_MPI $< -o $@ -lm
build/emu/sample build/emu/test_dit: build/emu/%: csrc/%.cu $(HDRS)
	@mkdir -p build/emu
	$(CXX) $(EMUFLAGS) $< -o $@ -lm

tests/data/%.bin: python/make_test_case.py python/blockflow/*.py
	@mkdir -p tests/data
	$(PYTHON) python/make_test_case.py --config $* --out $@
testdata: tests/data/tiny.bin tests/data/dits.bin

test: emu tests/data/tiny.bin
	./build/emu/test_dit tests/data/tiny.bin
	$(MPIRUN) --oversubscribe -np 4 ./build/emu/test_comm

test-gpu: gpu testdata
	./build/gpu/test_dit tests/data/tiny.bin
	./build/gpu/test_dit tests/data/dits.bin
	$(MPIRUN) --oversubscribe -np 4 ./build/gpu/test_comm

clean:
	rm -rf build

.PHONY: gpu emu test test-gpu testdata clean
