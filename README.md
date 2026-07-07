# Qwen3 AWQ INT4 추론 실행 가이드

이 저장소는 packed AWQ INT4 체크포인트를 이용해 Qwen3 모델을 추론하기 위한 코드입니다.

지원하는 실행 경로는 두 가지입니다.

- `kernel`: `awq_inference_engine` CUDA 커널을 사용하는 빠른 AWQ INT4 경로
- `torch_fallback`: CUDA 커널을 사용하지 않고 packed INT4 weight를 dense weight로 풀어서 `F.linear`로 계산하는 디버깅용 경로

일반적인 추론에는 `kernel` 경로를 사용해야 합니다. `torch_fallback` 경로는 커널이 없는 환경에서 동작 확인이나 병목 분석을 할 때만 사용하는 것을 권장합니다.

## 파일 구성

- `infer_awq.py`: 메인 추론 스크립트. backend 선택과 단계별 시간 출력 지원
- `infer_awq_no_kernel.py`: 커널을 쓰지 않는 `torch_fallback` 전용 실행 스크립트
- `awq/kernels/setup.py`: `awq_inference_engine` CUDA 확장 모듈 빌드 스크립트
- `requirements.txt`: PyTorch를 제외한 Python 의존성 목록
- `model/qwen3-4b-w4-g128-awq-v2.pt`: packed AWQ 체크포인트
- `qwen3-4b-awq-runtime/`: tokenizer, config 등 런타임 파일

## Requirements

PyTorch는 대상 환경에 맞게 별도로 설치해야 합니다.

특히 Jetson에서는 일반 PyPI용 PyTorch wheel을 무작정 설치하면 CUDA가 잡히지 않을 수 있습니다. JetPack 버전에 맞는 NVIDIA 제공 PyTorch wheel을 사용해야 합니다.

공통 Python 패키지 설치:

```bash
python -m pip install -U pip setuptools wheel ninja
python -m pip install -r requirements.txt
```

필요한 구성 요소:

- Python 3.10 이상
- GPU 추론용 CUDA 지원 PyTorch
- `awq_inference_engine` 빌드용 CUDA toolkit 및 `nvcc`
- CUDA toolkit과 호환되는 C++ compiler
- `ninja`, `setuptools`, `wheel`
- `transformers>=4.51.0`
- `accelerate>=0.34.2`

CUDA 인식 여부 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.device_count() else 'no cuda device')"
```

`torch.version.cuda`는 값이 있는데 `torch.cuda.is_available()`이 `False`라면, PyTorch는 CUDA 지원 버전이지만 현재 실행 환경에서 CUDA device를 못 보고 있는 상태입니다. driver, runtime, container GPU passthrough, 환경 변수 등을 확인해야 합니다.

## 환경 설정

### Colab / A100

먼저 Colab 런타임을 GPU로 설정해야 합니다.

확인:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

라이브러리 경로를 추가해야 하는 경우 기존 `LD_LIBRARY_PATH`를 덮어쓰면 안 됩니다.

아래 방식은 잘못된 예입니다.

```python
%env LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64
```

이 명령은 기존 NVIDIA driver 경로를 지워서 `torch.cuda.is_available()`이 `False`가 될 수 있습니다.

대신 기존 값을 보존해서 추가해야 합니다.

```python
import os

extra = "/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64"
old = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{extra}:{old}" if old else extra
print(os.environ["LD_LIBRARY_PATH"])
```

또는 Colab에서 필요한 NVIDIA 경로를 명시적으로 포함합니다.

```python
%env LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib64-nvidia
```

이미 잘못된 `LD_LIBRARY_PATH` 상태에서 `torch`를 import했다면, 같은 런타임에서 복구가 안 될 수 있습니다. 이 경우 런타임을 재시작한 뒤 `torch` import 전에 환경 변수를 다시 설정하세요.

### Jetson Board

Jetson에서는 JetPack 버전에 맞는 PyTorch를 설치해야 합니다.

확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
tegrastats
nvcc --version
```

`torch.cuda.is_available()`이 `False`이면 AWQ 커널 경로를 사용할 수 없습니다. 먼저 JetPack, CUDA, PyTorch 설치를 맞춰야 합니다.

## Kernel Setting

`awq_inference_engine`은 Python에서 import하는 CUDA 확장 모듈입니다. 추론에 사용할 Python, PyTorch, CUDA 환경과 같은 환경에서 빌드해야 합니다.

### A100

A100은 compute capability `sm_80`입니다. Colab A100에서는 CUDA가 정상적으로 보이면 PyTorch extension 빌드가 보통 자동으로 아키텍처를 감지합니다.

빌드 및 설치:

```bash
cd awq/kernels
python setup.py install
cd ../..
```

확인:

```bash
python -c "import awq_inference_engine; print('awq_inference_engine OK'); print(awq_inference_engine.__file__)"
```

`.so` 파일은 있는데 import가 안 된다면 `PYTHONPATH`에 빌드 결과 경로가 없을 수 있습니다.

확인:

```bash
find awq/kernels -name "awq_inference_engine*.so" -print
```

임시로 `PYTHONPATH` 추가:

```bash
export PYTHONPATH=$PWD/awq/kernels/build/lib.linux-x86_64-cpython-312:$PWD:$PYTHONPATH
```

Python 버전이 다르면 `cpython-312` 부분이 달라질 수 있습니다.

### Jetson

Jetson 보드 위에서 Jetson용 PyTorch 환경을 활성화한 뒤 직접 빌드합니다.

```bash
cd awq/kernels
python setup.py install
cd ../..
```

CUDA architecture 자동 감지가 실패하면 보드에 맞는 `TORCH_CUDA_ARCH_LIST`를 지정한 뒤 다시 빌드합니다.

예시:

```bash
export TORCH_CUDA_ARCH_LIST="8.7"
```

변경 후 재빌드:

```bash
cd awq/kernels
python setup.py clean
python setup.py install
cd ../..
```

확인:

```bash
python -c "import awq_inference_engine; print('awq_inference_engine OK'); print(awq_inference_engine.__file__)"
```

## 추론 실행 방법

### Hugging Face Hub에서 모델 파일 받기

GitHub에는 코드만 올리고, 대용량 AWQ checkpoint와 runtime 파일은 Hugging Face Hub model repository에서 가져옵니다.

모델/runtime repository:

```text
https://huggingface.co/HMHMlee/qwen3-4b-awq-runtime
```

이 repository를 로컬의 `qwen3-4b-awq-runtime/` 디렉토리로 내려받고, checkpoint 파일은 `model/` 디렉토리에 배치해야 합니다.

권장 Hub repository 구조:

```text
HMHMlee/qwen3-4b-awq-runtime
├── README.md
├── LICENSE
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── model.safetensors.index.json
└── qwen3-4b-w4-g128-awq-v2.pt
```

다운로드 예시:

```bash
python -m pip install -U huggingface_hub

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HMHMlee/qwen3-4b-awq-runtime",
    local_dir="qwen3-4b-awq-runtime",
    local_dir_use_symlinks=False,
)
PY

mkdir -p model
mv qwen3-4b-awq-runtime/qwen3-4b-w4-g128-awq-v2.pt model/
```

이후 실행:

```bash
python infer_awq.py \
  --model_path qwen3-4b-awq-runtime \
  --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
  --prompt "please talk about harry potter"
```

### AWQ CUDA Kernel 경로

CUDA가 사용 가능하면 `infer_awq.py`는 기본적으로 `kernel` backend를 선택합니다.

```bash
python infer_awq.py \
  --model_path qwen3-4b-awq-runtime \
  --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
  --prompt "please talk about harry potter" \
  --max_new_tokens 64
```

backend를 명시하려면:

```bash
python infer_awq.py \
  --model_path qwen3-4b-awq-runtime \
  --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
  --prompt "please talk about harry potter" \
  --awq_backend kernel
```

`kernel` backend 조건:

- `torch.cuda.is_available() == True`
- `awq_inference_engine` import 성공
- checkpoint가 CUDA device로 로드됨

### No-Kernel Fallback 경로

커널 없이 동작을 확인하거나 병목을 분석할 때 사용합니다.

```bash
python infer_awq_no_kernel.py \
  --model_path qwen3-4b-awq-runtime \
  --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
  --prompt "please talk about harry potter" \
  --max_new_tokens 64
```

동일한 명령을 `infer_awq.py`에서 직접 실행하려면:

```bash
python infer_awq.py \
  --model_path qwen3-4b-awq-runtime \
  --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
  --prompt "please talk about harry potter" \
  --awq_backend torch_fallback
```

주의:

- `torch_fallback`은 AWQ CUDA kernel을 사용하지 않습니다.
- packed INT4 weight를 dense weight로 풀어서 `F.linear`를 수행합니다.
- `--cache_dequantized_weights`를 사용하지 않으면 forward 중 반복적으로 dense weight를 생성합니다.
- `--cache_dequantized_weights`는 속도는 개선할 수 있지만 GPU 메모리를 크게 요구합니다.

## 시간 측정

기본적으로 추론이 끝난 뒤 단계별 시간이 출력됩니다.

출력 항목:

- `load_tokenizer`
- `load_config`
- `init_empty_model`
- `replace_linear_modules`
- `load_checkpoint_to_device`
- `model_to_eval`
- `build_inputs`
- `generate`
- `decode`
- `tokens_per_second`

시간 출력을 끄려면:

```bash
python infer_awq.py ... --no_timing
```

no-kernel fallback 내부 연산 시간을 보고 싶으면:

```bash
python infer_awq_no_kernel.py \
  --model_path qwen3-4b-awq-runtime \
  --load_quant model/qwen3-4b-w4-g128-awq-v2.pt \
  --prompt "please talk about harry potter" \
  --profile_fallback_ops
```

추가 출력 항목:

- `fallback_dequantize_weight`
- `fallback_linear`
- `fallback_forward`
- per-call 평균 시간

`--profile_fallback_ops`는 정확한 측정을 위해 CUDA synchronization을 추가하므로 실제 추론 속도를 더 느리게 만들 수 있습니다.

## 문제 해결

### `torch.cuda.is_available()`이 `False`

확인:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
echo $LD_LIBRARY_PATH
```

주요 원인:

- CPU-only runtime 사용
- NVIDIA driver가 보이지 않음
- Docker/container 실행 시 GPU passthrough가 안 됨
- `LD_LIBRARY_PATH`를 덮어써서 NVIDIA driver 경로가 사라짐
- Jetson에서 JetPack과 맞지 않는 PyTorch wheel 사용

### `ModuleNotFoundError: No module named 'awq_inference_engine'`

CUDA 확장 모듈이 설치되지 않았거나 Python import 경로에 없습니다.

확인 및 재설치:

```bash
find awq/kernels -name "awq_inference_engine*.so" -print
cd awq/kernels
python setup.py install
cd ../..
python -c "import awq_inference_engine; print('OK')"
```

### GPU 메모리는 0인데 CPU RAM만 증가함

대부분 PyTorch가 CUDA를 못 보고 CPU로 실행 중인 경우입니다.

```python
torch.cuda.is_available() == False
```

이 상태에서는 코드가 `device="cpu"`를 선택하고, `device_map={"": "cpu"}`로 모델을 CPU RAM에 로드합니다. 먼저 CUDA 인식 문제를 해결해야 합니다.

### Colab Drive에서 로딩이 느림

Google Drive에서 `.pt` checkpoint를 직접 읽으면 매우 느릴 수 있습니다. 시간 측정이나 반복 실험을 할 때는 `/content` 로컬 디스크로 복사한 뒤 실행하는 것이 좋습니다.

```bash
cp -r /content/drive/MyDrive/Global_capstone/jetson_test/qwen3-4b-awq-runtime /content/
mkdir -p /content/model
cp /content/drive/MyDrive/Global_capstone/jetson_test/model/qwen3-4b-w4-g128-awq-v2.pt /content/model/
```

실행:

```bash
python infer_awq.py \
  --model_path /content/qwen3-4b-awq-runtime \
  --load_quant /content/model/qwen3-4b-w4-g128-awq-v2.pt \
  --prompt "please talk about harry potter"
```
