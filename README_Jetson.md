# Qwen3 AWQ INT4 Inference on Jetson Orin Nano

이 저장소는 packed AWQ INT4 체크포인트를 이용하여 **Jetson Orin Nano에서 Qwen3 모델을 추론하기 위한 코드**입니다.

현재 검증한 대상 환경은 다음과 같습니다.

```text
Device: Jetson Orin Nano
Jetson Linux: R36.4.3
JetPack: 6.2
CUDA: 12.6
Python: 3.10
Architecture: aarch64

PyTorch: 2.8.0
PyTorch CUDA build: 12.6
CUDA available: True
GPU: Orin
```

지원하는 실행 경로는 두 가지입니다.

* `kernel`

  * `awq_inference_engine` CUDA extension을 사용하는 AWQ INT4 실행 경로
  * 실제 Jetson GPU 추론 및 성능 측정에 사용할 기본 경로

* `torch_fallback`

  * AWQ CUDA 커널을 사용하지 않는 디버깅용 실행 경로
  * packed INT4 weight를 dense weight로 복원한 후 `F.linear`를 수행
  * 동작 검증 및 병목 분석 용도로만 사용 권장

일반적인 Jetson 추론 실험에서는 `kernel` backend를 사용해야 합니다.

---

# 1. Repository Structure

```text
.
├── infer_awq.py
├── infer_awq_no_kernel.py
├── requirements.txt
│
├── awq/
│   └── kernels/
│       ├── setup.py
│       └── ...
│
├── model/
│   └── qwen3-4b-w4-g128-awq-v2.pt
│
└── qwen3-4b-awq-runtime/
    ├── config.json
    ├── generation_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    └── ...
```

주요 파일의 역할은 다음과 같습니다.

```text
infer_awq.py
    메인 추론 스크립트
    kernel / torch_fallback backend 선택 가능
    모델 로딩 및 생성 단계별 시간 측정 지원

infer_awq_no_kernel.py
    AWQ CUDA kernel을 사용하지 않는 fallback 전용 스크립트

awq/kernels/setup.py
    awq_inference_engine CUDA extension 빌드 설정

requirements.txt
    PyTorch를 제외한 Python dependency 목록
build_rag_index.py
    already chunked JSONL에서 embeddings / FAISS index 생성
rag/
    chat_awq.py와 다른 실행 경로에서 재사용 가능한 RAG 모듈

model/qwen3-4b-w4-g128-awq-v2.pt
    packed AWQ W4A16 checkpoint

qwen3-4b-awq-runtime/
    Qwen3 모델 config 및 tokenizer 관련 런타임 파일
```

---

# 2. Jetson Environment Requirements

본 프로젝트는 다음 환경을 기준으로 합니다.

```text
Jetson Linux R36.4.3
JetPack 6.2
CUDA 12.6
Python 3.10
aarch64
Jetson Orin Nano GPU
```

필요한 주요 구성 요소:

```text
CUDA 지원 PyTorch
CUDA Toolkit
nvcc
C++ compiler
ninja
setuptools
wheel
transformers
accelerate
sentencepiece
protobuf
numpy
tqdm
sentence-transformers
```

PyTorch는 `requirements.txt`에 포함하지 않습니다.

Jetson에서는 대상 JetPack 및 CUDA 환경에 맞는 PyTorch wheel을 먼저 설치해야 합니다.

---

# 3. PyTorch Installation

## 3.1 기존 잘못된 PyTorch 제거

Jetson 환경과 맞지 않는 PyTorch가 이미 설치된 경우 먼저 제거합니다.

```bash
python -m pip uninstall -y torch torchvision torchaudio
```

캐시를 제거하려면:

```bash
python -m pip cache purge
```

---

## 3.2 NumPy 설치

현재 검증 환경에서는 다음 버전을 사용합니다.

```bash
python -m pip install --no-cache-dir numpy==1.26.4
```

---

## 3.3 Jetson CUDA 12.6용 PyTorch 설치

현재 검증한 환경에서는 다음 PyTorch를 사용합니다.

```text
torch 2.8.0
CUDA build 12.6
```

설치:

```bash
python -m pip install --no-cache-dir \
    torch==2.8.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

설치 이후 반드시 CUDA 인식을 확인합니다.

```bash
python - <<'PY'
import torch

print("torch version:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

현재 검증된 정상 출력:

```text
torch version: 2.8.0
torch CUDA: 12.6
CUDA available: True
device count: 1
GPU: Orin
```

이 결과가 출력되면 PyTorch가 Jetson Orin GPU를 정상적으로 인식하는 상태입니다.

---

# 4. Python Dependencies

먼저 빌드 관련 패키지를 설치합니다.

```bash
python -m pip install -U pip setuptools wheel ninja
```

그다음:

```bash
python -m pip install -r requirements.txt
```

Jetson에서 RAG를 사용할 경우 `sentence-transformers` 외에 FAISS도 별도로 준비해야 합니다. `requirements.txt`에는 포함하지 않았고, JetPack/CUDA 환경에 맞는 wheel 또는 소스 빌드를 사용해야 합니다. FAISS가 없으면 인덱스 생성과 retrieval에서 즉시 에러가 발생하도록 구현되어 있습니다.

권장 `requirements.txt`:

```text
# PyTorch is intentionally not listed here.
# Install the CUDA/JetPack-compatible PyTorch build first.

# Inference
accelerate>=0.34.2,<2.0
transformers>=4.51.0,<5.0
sentencepiece
protobuf
numpy<2
tqdm

# AWQ CUDA extension build helper
ninja
```

현재 환경을 정확하게 재현해야 하는 경우에는 실제 테스트가 끝난 이후 다음과 같이 버전을 고정하는 것을 권장합니다.

```text
accelerate==1.14.0
transformers==4.51.3
numpy==1.26.4
sentencepiece
protobuf
tqdm
ninja
```

---

# 5. CUDA Build Environment Check

`awq_inference_engine`은 PyTorch CUDA extension입니다.

따라서 다음 세 가지를 모두 확인해야 합니다.

```text
1. PyTorch에서 CUDA 사용 가능
2. nvcc 사용 가능
3. C++ compiler 사용 가능
```

확인:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

```bash
nvcc --version
```

```bash
which nvcc
```

```bash
g++ --version
```

```bash
which g++
```

CUDA 경로도 확인합니다.

```bash
ls -l /usr/local/cuda
```

```bash
echo $CUDA_HOME
```

필요한 경우:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

주의:

기존 `LD_LIBRARY_PATH`를 완전히 덮어쓰지 말고 기존 값을 유지한 상태에서 CUDA 경로를 추가하는 것이 안전합니다.

---

# 6. Verify Real CUDA Computation

`torch.cuda.is_available()` 확인 후 실제 CUDA tensor 연산을 수행합니다.

```bash
python - <<'PY'
import torch
import time

assert torch.cuda.is_available(), "CUDA is not available"

device = torch.device("cuda")

x = torch.randn(2048, 2048, device=device)
y = torch.randn(2048, 2048, device=device)

torch.cuda.synchronize()
start = time.time()

z = x @ y

torch.cuda.synchronize()
elapsed = time.time() - start

print("GPU:", torch.cuda.get_device_name(0))
print("Result device:", z.device)
print("Elapsed:", elapsed)
print(
    "Allocated memory:",
    torch.cuda.memory_allocated() / 1024**2,
    "MB",
)
PY
```

정상적인 경우:

```text
GPU: Orin
Result device: cuda:0
Elapsed: ...
Allocated memory: ... MB
```

형태로 출력됩니다.

다른 터미널에서는 다음을 이용해 GPU 사용량을 관찰할 수 있습니다.

```bash
tegrastats
```

추론 또는 행렬 연산 중 `GR3D_FREQ`의 변화를 확인할 수 있습니다.

---

# 7. Download Runtime Files and AWQ Checkpoint

GitHub 저장소에는 코드만 유지하고, 대용량 AWQ checkpoint와 runtime 파일은 Hugging Face Hub에서 받습니다.

Repository:

```text
HMHMlee/qwen3-4b-awq-runtime
```

먼저 Hugging Face Hub client를 설치합니다.

```bash
python -m pip install -U huggingface_hub
```

다운로드:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HMHMlee/qwen3-4b-awq-runtime",
    local_dir="qwen3-4b-awq-runtime",
)
PY
```

AWQ checkpoint를 `model/` 디렉토리로 이동합니다.

```bash
mkdir -p model
```

```bash
mv \
    qwen3-4b-awq-runtime/qwen3-4b-w4-g128-awq-v2.pt \
    model/
```

최종 구조:

```text
.
├── infer_awq.py
├── infer_awq_no_kernel.py
│
├── model/
│   └── qwen3-4b-w4-g128-awq-v2.pt
│
└── qwen3-4b-awq-runtime/
    ├── config.json
    ├── generation_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    └── ...
```

---

# 8. Build AWQ CUDA Kernel on Jetson

## 8.1 Important

`awq_inference_engine`은 Jetson 보드의 실제 Python, PyTorch, CUDA 환경에서 직접 빌드해야 합니다.

현재 Conda 환경이 `hyunmin`이라면:

```bash
conda activate hyunmin
```

확인:

```bash
which python
```

```bash
python -m pip --version
```

둘 다 현재 `hyunmin` 환경을 가리키는지 확인합니다.

예:

```text
/home/isp/miniconda3/envs/hyunmin/bin/python
```

---

## 8.2 Architecture Check

Jetson Orin 계열 CUDA architecture를 확인합니다.

PyTorch에서:

```bash
python - <<'PY'
import torch

print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

Orin에서는 다음 형태가 예상됩니다.

```text
device: Orin
capability: (8, 7)
```

커널 빌드 시 architecture를 명시하려면:

```bash
export TORCH_CUDA_ARCH_LIST="8.7"
```

확인:

```bash
echo $TORCH_CUDA_ARCH_LIST
```

---

## 8.3 Build

권장 빌드 방법:

```bash
cd awq/kernels
```

기존 빌드 결과가 있다면 제거합니다.

```bash
rm -rf build
rm -rf *.egg-info
```

그다음 현재 Python 환경에 설치합니다.

```bash
python setup.py install
```

또는 프로젝트의 패키징 구성이 지원한다면 다음 형태도 사용할 수 있습니다.

```bash
python -m pip install -v .
```

빌드 후 프로젝트 루트로 돌아갑니다.

```bash
cd ../..
```

---

## 8.4 Import Check

```bash
python - <<'PY'
import awq_inference_engine

print("awq_inference_engine OK")
print(awq_inference_engine.__file__)
PY
```

정상이라면 `.so` 모듈의 경로가 출력됩니다.

예:

```text
awq_inference_engine OK
.../awq_inference_engine.cpython-310-aarch64-linux-gnu.so
```

---

# 9. If Kernel Build Succeeds but Import Fails

`.so` 파일을 찾습니다.

```bash
find awq/kernels \
    -name "awq_inference_engine*.so" \
    -print
```

Jetson에서는 build directory 이름이 일반적으로 x86_64가 아니라 aarch64를 포함합니다.

예:

```text
build/lib.linux-aarch64-cpython-310/
```

필요한 경우 임시로 Python path를 추가합니다.

```bash
export PYTHONPATH=\
$PWD/awq/kernels/build/lib.linux-aarch64-cpython-310:\
$PWD:\
$PYTHONPATH
```

그다음 다시 확인합니다.

```bash
python -c \
"import awq_inference_engine; print(awq_inference_engine.__file__)"
```

단, 가장 권장되는 방식은 `.so`를 임시 `PYTHONPATH`로 참조하는 것보다 현재 Python 환경에 extension을 정상적으로 설치하는 것입니다.

---

# 10. Run AWQ Kernel Inference

CUDA와 AWQ extension이 모두 정상적인 경우 다음 명령으로 실행합니다.

```bash
python infer_awq.py \
    --model_path qwen3-4b-awq-runtime \
    --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
    --prompt "please talk about harry potter" \
    --max_new_tokens 64 \
    --awq_backend kernel
```

kernel backend 사용 조건:

```text
torch.cuda.is_available() == True

and

import awq_inference_engine 성공

and

AWQ checkpoint가 CUDA device에 정상적으로 로드됨

and

CUDA extension이 Jetson Orin architecture에서 정상 실행됨
```

성공적으로 실행되면 다음 흐름을 사용합니다.

```text
Packed INT4 checkpoint
        │
        ▼
AWQ quantized linear layer
        │
        ▼
awq_inference_engine
        │
        ▼
CUDA kernel
        │
        ▼
Jetson Orin GPU
```

---

# 11. Run Without AWQ CUDA Kernel

커널 빌드 실패 또는 동작 검증이 필요한 경우 fallback backend를 사용합니다.

## Dedicated fallback script

```bash
python infer_awq_no_kernel.py \
    --model_path qwen3-4b-awq-runtime \
    --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
    --prompt "please talk about harry potter" \
    --max_new_tokens 64
```

또는:

```bash
python infer_awq.py \
    --model_path qwen3-4b-awq-runtime \
    --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
    --prompt "please talk about harry potter" \
    --max_new_tokens 64 \
    --awq_backend torch_fallback
```

fallback 경로는:

```text
Packed INT4 weight
        │
        ▼
Dequantization
        │
        ▼
Dense FP16/BF16 weight
        │
        ▼
torch.nn.functional.linear
        │
        ▼
CUDA or CPU execution
```

형태로 작동합니다.

주의:

* `torch_fallback`은 AWQ INT4 CUDA kernel을 사용하지 않습니다.
* packed INT4 weight를 dense 형태로 복원합니다.
* 실제 INT4 kernel 성능 평가용으로 사용하면 안 됩니다.
* 반복적인 dequantization이 발생할 경우 매우 느릴 수 있습니다.
* dequantized weight caching은 메모리 사용량을 크게 증가시킬 수 있습니다.

Jetson Orin Nano처럼 메모리가 제한적인 환경에서는 `--cache_dequantized_weights` 사용에 특히 주의해야 합니다.

---

# 12. Timing Measurement

`infer_awq.py`는 단계별 시간을 출력합니다.

측정 항목:

```text
load_tokenizer
load_config
init_empty_model
replace_linear_modules
load_checkpoint_to_device
model_to_eval
build_inputs
generate
decode
tokens_per_second
```

예:

```text
load_tokenizer:              ...
load_config:                 ...
init_empty_model:            ...
replace_linear_modules:      ...
load_checkpoint_to_device:   ...
model_to_eval:               ...
build_inputs:                ...
generate:                    ...
decode:                      ...

tokens_per_second:           ...
```

시간 출력을 비활성화하려면:

```bash
python infer_awq.py \
    ... \
    --no_timing
```

---

# 13. Profile torch_fallback Operations

fallback 내부 병목을 분석하려면:

```bash
python infer_awq_no_kernel.py \
    --model_path qwen3-4b-awq-runtime \
    --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
    --prompt "please talk about harry potter" \
    --max_new_tokens 64 \
    --profile_fallback_ops
```

추가 측정 항목:

```text
fallback_dequantize_weight
fallback_linear
fallback_forward
average time per call
```

주의:

`--profile_fallback_ops`는 CUDA operation timing 정확도를 높이기 위해 synchronization을 삽입할 수 있으므로 일반 추론보다 느려질 수 있습니다.

따라서:

```text
일반 throughput 측정:
    --profile_fallback_ops 사용하지 않음

병목 분석:
    --profile_fallback_ops 사용
```

으로 구분합니다.

---

# 14. Monitoring on Jetson

Jetson에서는 추론 실행과 동시에 다른 터미널에서 다음을 실행합니다.

```bash
tegrastats
```

주요 확인 항목:

```text
RAM
SWAP
CPU
GR3D_FREQ
EMC_FREQ
GPU temperature
VDD_IN
VDD_CPU_GPU_CV
```

특히 다음을 관찰합니다.

```text
GR3D_FREQ
    GPU activity 확인

RAM
    unified memory 사용량 확인

SWAP
    메모리 부족 여부 확인

EMC_FREQ
    memory bandwidth pressure 간접 확인

VDD_IN
    전체 보드 소비전력 확인
```

PyTorch의 GPU 메모리 정보도 함께 확인할 수 있습니다.

```python
print(
    torch.cuda.memory_allocated() / 1024**3
)

print(
    torch.cuda.memory_reserved() / 1024**3
)

print(
    torch.cuda.max_memory_allocated() / 1024**3
)
```

Jetson은 CPU와 GPU가 시스템 메모리를 공유하므로 `tegrastats`의 RAM 사용량과 PyTorch CUDA allocator 수치를 함께 관찰하는 것이 좋습니다.

---

# 15. Recommended Validation Order

Jetson에서는 한 번에 전체 시스템을 실행하기보다 다음 순서로 검증하는 것을 권장합니다.

```text
Step 1
PyTorch import 확인

        ↓

Step 2
torch.cuda.is_available() 확인

        ↓

Step 3
CUDA tensor 연산 확인

        ↓

Step 4
nvcc 및 compiler 확인

        ↓

Step 5
awq_inference_engine 빌드

        ↓

Step 6
awq_inference_engine import 확인

        ↓

Step 7
torch_fallback으로 checkpoint correctness 확인

        ↓

Step 8
kernel backend 실행

        ↓

Step 9
출력 품질 비교

        ↓

Step 10
latency / TPS / RAM / GPU utilization 측정
```

이 순서를 사용하면:

```text
PyTorch 문제
CUDA 문제
CUDA extension 문제
AWQ checkpoint 문제
모델 로딩 문제
kernel correctness 문제
performance 문제
```

를 서로 분리하여 분석할 수 있습니다.

---

# 16. Troubleshooting

## 16.1 torch.cuda.is_available() == False

확인:

```bash
python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

현재 Jetson 환경에서는 다음 결과가 정상입니다.

```text
torch version: 2.8.0
torch CUDA: 12.6
CUDA available: True
device count: 1
GPU: Orin
```

현재 환경에서 다시 `False`가 된다면 다음을 확인합니다.

```text
1. 다른 Conda environment가 활성화되었는가?
2. torch가 일반 PyPI 버전으로 재설치되었는가?
3. CUDA 13용 torch로 교체되었는가?
4. LD_LIBRARY_PATH가 잘못 설정되었는가?
5. 실행 중인 python과 pip가 서로 다른 환경인가?
```

확인:

```bash
which python
```

```bash
python -m pip --version
```

```bash
python -c \
"import torch; print(torch.__file__)"
```

```bash
python -c \
"import torch; print(torch.__version__, torch.version.cuda)"
```

---

## 16.2 `ModuleNotFoundError: No module named 'awq_inference_engine'`

extension이 빌드되지 않았거나 현재 Python 환경에 설치되지 않은 상태입니다.

확인:

```bash
find awq/kernels \
    -name "awq_inference_engine*.so" \
    -print
```

재빌드:

```bash
export TORCH_CUDA_ARCH_LIST="8.7"

cd awq/kernels

rm -rf build
rm -rf *.egg-info

python setup.py install

cd ../..
```

확인:

```bash
python -c \
"import awq_inference_engine; print('OK'); print(awq_inference_engine.__file__)"
```

---

## 16.3 Kernel Build Fails

먼저 다음 정보를 기록합니다.

```bash
python -c \
"import torch; print(torch.__version__); print(torch.version.cuda)"
```

```bash
nvcc --version
```

```bash
g++ --version
```

```bash
echo $CUDA_HOME
```

```bash
echo $TORCH_CUDA_ARCH_LIST
```

권장 환경:

```text
torch: 2.8.0
torch CUDA: 12.6
nvcc: CUDA 12.6
CUDA_HOME: /usr/local/cuda
TORCH_CUDA_ARCH_LIST: 8.7
```

그다음 완전 재빌드합니다.

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.7"

cd awq/kernels

rm -rf build
rm -rf *.egg-info

python setup.py install

cd ../..
```

빌드 실패 시 가장 먼저 확인해야 할 것은 단순한 PyTorch CUDA 인식 여부가 아니라 **AWQ kernel source 자체가 aarch64 및 sm_87에서 컴파일 가능한 구현인지**입니다.

x86_64 전용 assembly, architecture-specific optimization 또는 특정 CUDA API에 의존하는 코드가 포함되어 있다면 source 수정이 필요할 수 있습니다.

---

## 16.4 Build Succeeds but Kernel Execution Fails

예를 들어 다음과 같은 runtime 문제가 발생할 수 있습니다.

```text
invalid device function
```

또는:

```text
no kernel image is available for execution on the device
```

이 경우 extension을 대상 GPU architecture로 다시 빌드합니다.

```bash
export TORCH_CUDA_ARCH_LIST="8.7"
```

```bash
cd awq/kernels

rm -rf build
rm -rf *.egg-info

python setup.py install

cd ../..
```

그리고 다시 확인합니다.

```bash
python -c \
"import awq_inference_engine; print(awq_inference_engine.__file__)"
```

---

## 16.5 CPU RAM Increases but GPU Activity Is Low

이 경우 다음을 구분해야 합니다.

### Case A: CUDA를 사용하지 않는 경우

```python
torch.cuda.is_available() == False
```

이 경우 먼저 CUDA 환경을 해결해야 합니다.

### Case B: torch_fallback을 사용하는 경우

CUDA device에 tensor를 올렸더라도 다음 경로라면:

```text
INT4 unpack
→ dequantization
→ dense weight 생성
→ F.linear
```

AWQ INT4 전용 kernel의 메모리 및 연산 효율성을 얻을 수 없습니다.

따라서 backend를 확인해야 합니다.

```text
--awq_backend kernel
```

을 명시적으로 사용합니다.

### Case C: model loading 과정

Jetson은 CPU와 GPU가 시스템 메모리를 공유하므로 단순히 RAM 사용량 증가만으로 CPU-only 실행이라고 판단하면 안 됩니다.

다음 항목을 함께 확인해야 합니다.

```text
torch.cuda.is_available()
parameter.device
input.device
GR3D_FREQ
PyTorch CUDA allocator statistics
```

---

# 17. Recommended First Run on Jetson

처음부터 긴 생성을 수행하지 말고 짧은 generation으로 테스트합니다.

## Step 1. CUDA 확인

```bash
python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
PY
```

---

## Step 2. AWQ extension 확인

```bash
python - <<'PY'
import awq_inference_engine

print("AWQ kernel import OK")
print(awq_inference_engine.__file__)
PY
```

---

## Step 3. Fallback correctness test

```bash
python infer_awq.py \
    --model_path qwen3-4b-awq-runtime \
    --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
    --prompt "What is artificial intelligence?" \
    --max_new_tokens 16 \
    --awq_backend torch_fallback
```

---

## Step 4. Kernel test

```bash
python infer_awq.py \
    --model_path qwen3-4b-awq-runtime \
    --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
    --prompt "What is artificial intelligence?" \
    --max_new_tokens 16 \
    --awq_backend kernel
```

---

## Step 5. Compare Outputs

다음 항목을 비교합니다.

```text
1. 모델이 정상적인 문장을 생성하는가?
2. kernel과 fallback의 출력이 지나치게 다르지 않은가?
3. NaN 또는 Inf가 발생하지 않는가?
4. kernel backend가 실제로 더 빠른가?
5. RAM 사용량이 허용 범위 내인가?
6. swap이 과도하게 증가하지 않는가?
7. GR3D_FREQ가 실제 추론 중 증가하는가?
```

---

# 18. Performance Evaluation

Jetson 실험에서는 최소한 다음 결과를 기록하는 것을 권장합니다.

```text
Model:
    Qwen3-4B

Quantization:
    AWQ W4A16
    group size = 128

Backend:
    kernel
    torch_fallback

Metrics:
    model load time
    first token latency
    total generation latency
    tokens per second
    peak RAM
    peak CUDA allocated memory
    average power
    peak power
```

비교 구조:

```text
Qwen3-4B BF16
        vs
Qwen3-4B AWQ torch_fallback
        vs
Qwen3-4B AWQ kernel
```

이 비교를 통해 다음을 분리할 수 있습니다.

```text
Quantization 자체의 메모리 감소 효과

vs

AWQ INT4 kernel을 적용했을 때의 실제 latency 감소 효과
```

---

# 19. Important Limitation

PyTorch에서 다음이 성공했다고 해서:

```text
torch.cuda.is_available() == True
```

AWQ CUDA kernel도 자동으로 정상 동작하는 것은 아닙니다.

전체 검증 단계는 다음과 같이 구분해야 합니다.

```text
PyTorch CUDA
    ✅ 현재 확인 완료

        ↓

CUDA extension build
    별도 확인 필요

        ↓

awq_inference_engine import
    별도 확인 필요

        ↓

sm_87 kernel execution
    별도 확인 필요

        ↓

AWQ numerical correctness
    별도 확인 필요

        ↓

Jetson performance benefit
    benchmark 필요
```

특히 `ninja`는 CUDA extension 빌드를 빠르게 수행하기 위한 빌드 도구이며, 설치만으로 AWQ CUDA kernel의 Jetson 호환성을 보장하지 않습니다.

실제 핵심 검증 대상은:

```text
awq_inference_engine
    +
aarch64
    +
CUDA 12.6
    +
sm_87
    +
PyTorch 2.8.0
```

조합의 빌드 및 runtime compatibility입니다.

---

# 20. Summary

현재 검증된 Jetson 실행 환경:

```text
Jetson Orin Nano
Jetson Linux R36.4.3
JetPack 6.2
CUDA 12.6
Python 3.10
aarch64

torch 2.8.0
torch CUDA 12.6
CUDA available True
GPU Orin
```

권장 실행 과정:

```text
1. Jetson용 PyTorch 설치
2. CUDA tensor 연산 확인
3. Python dependencies 설치
4. nvcc 및 compiler 확인
5. TORCH_CUDA_ARCH_LIST=8.7 설정
6. awq_inference_engine 빌드
7. extension import 확인
8. torch_fallback correctness test
9. kernel backend test
10. tegrastats와 timing 결과를 이용한 성능 측정
```

최종 목표는 단순히 AWQ checkpoint를 Jetson에서 로드하는 것이 아니라:

```text
Packed AWQ INT4 checkpoint
        │
        ▼
AWQ quantized linear layer
        │
        ▼
awq_inference_engine
        │
        ▼
Jetson Orin CUDA kernel
        │
        ▼
Low-memory, GPU-accelerated inference
```

경로가 실제로 동작하는 것을 검증하는 것입니다.
