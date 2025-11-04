# pyBGVAR GitHub 설치 및 사용 가이드

> GitHub에 퍼블리시된 pyBGVAR 패키지를 설치하고 사용하는 완벽 가이드

## 목차
1. [GitHub에서 설치하기](#1-github에서-설치하기)
2. [설치 방법 비교](#2-설치-방법-비교)
3. [가상 환경 설정 (권장)](#3-가상-환경-설정-권장)
4. [설치 확인](#4-설치-확인)
5. [업데이트 및 제거](#5-업데이트-및-제거)
6. [문제 해결](#6-문제-해결)
7. [GitHub 저장소 활용](#7-github-저장소-활용)

---

## 1. GitHub에서 설치하기

### 방법 A: 최신 버전 직접 설치 (권장)

GitHub의 main 브랜치에서 최신 버전을 직접 설치합니다.

```bash
pip install git+https://github.com/[사용자명]/pyBGVAR.git
```

**예시:**
```bash
# GitHub 사용자명이 'your-username'인 경우
pip install git+https://github.com/your-username/pyBGVAR.git
```

### 방법 B: 특정 브랜치 설치

개발 중인 기능이나 특정 브랜치를 설치하고 싶을 때:

```bash
pip install git+https://github.com/[사용자명]/pyBGVAR.git@[브랜치명]
```

**예시:**
```bash
# development 브랜치 설치
pip install git+https://github.com/your-username/pyBGVAR.git@development

# feature 브랜치 설치
pip install git+https://github.com/your-username/pyBGVAR.git@feature/new-functionality
```

### 방법 C: 특정 태그/릴리즈 설치

안정적인 버전(태그)을 설치:

```bash
pip install git+https://github.com/[사용자명]/pyBGVAR.git@v0.1.0
```

**예시:**
```bash
# v0.1.0 태그 설치
pip install git+https://github.com/your-username/pyBGVAR.git@v0.1.0

# v1.0.0 릴리즈 설치
pip install git+https://github.com/your-username/pyBGVAR.git@v1.0.0
```

### 방법 D: 특정 커밋 설치

특정 커밋을 설치하고 싶을 때:

```bash
pip install git+https://github.com/[사용자명]/pyBGVAR.git@[커밋해시]
```

**예시:**
```bash
# 특정 커밋 해시로 설치
pip install git+https://github.com/your-username/pyBGVAR.git@a1b2c3d4
```

### 방법 E: 개발 모드 설치 (개발자용)

코드를 수정하면서 사용하고 싶을 때:

```bash
# 1. 저장소 클론
git clone https://github.com/[사용자명]/pyBGVAR.git
cd pyBGVAR

# 2. 개발 모드로 설치
pip install -e .
```

이 방법을 사용하면 코드를 수정해도 재설치 없이 변경사항이 즉시 반영됩니다.

---

## 2. 설치 방법 비교

| 방법 | 장점 | 단점 | 추천 대상 |
|------|------|------|-----------|
| **방법 A: 최신 버전** | 항상 최신 기능 사용 | 불안정할 수 있음 | 일반 사용자 |
| **방법 B: 특정 브랜치** | 개발 중인 기능 사용 | 더 불안정 | 얼리어답터 |
| **방법 C: 태그/릴리즈** | 가장 안정적 | 최신 기능 없음 | 프로덕션 환경 |
| **방법 D: 특정 커밋** | 정확한 버전 고정 | 관리 어려움 | 재현 연구 |
| **방법 E: 개발 모드** | 코드 수정 가능 | 저장소 필요 | 개발자/기여자 |

**권장 설치 방법:**
- **연구/분석용**: 방법 C (안정된 태그)
- **일반 사용**: 방법 A (최신 버전)
- **개발/기여**: 방법 E (개발 모드)

---

## 3. 가상 환경 설정 (권장)

### 왜 가상 환경을 사용해야 하나요?

- ✅ 패키지 충돌 방지
- ✅ 프로젝트별 독립적인 환경
- ✅ 재현 가능한 연구 환경
- ✅ 시스템 Python 오염 방지

### 3-1. venv 사용 (Python 내장)

#### Windows:
```bash
# 1. 가상 환경 생성
python -m venv bgvar_env

# 2. 가상 환경 활성화
bgvar_env\Scripts\activate

# 3. pyBGVAR 설치
pip install git+https://github.com/your-username/pyBGVAR.git

# 4. 작업 완료 후 비활성화
deactivate
```

#### macOS/Linux:
```bash
# 1. 가상 환경 생성
python3 -m venv bgvar_env

# 2. 가상 환경 활성화
source bgvar_env/bin/activate

# 3. pyBGVAR 설치
pip install git+https://github.com/your-username/pyBGVAR.git

# 4. 작업 완료 후 비활성화
deactivate
```

### 3-2. conda 사용

```bash
# 1. 새 환경 생성 (Python 3.10)
conda create -n bgvar_env python=3.10

# 2. 환경 활성화
conda activate bgvar_env

# 3. pyBGVAR 설치
pip install git+https://github.com/your-username/pyBGVAR.git

# 4. 작업 완료 후 비활성화
conda deactivate
```

### 3-3. requirements.txt로 환경 재현

프로젝트의 의존성을 저장하고 공유:

```bash
# 의존성 저장
pip freeze > requirements.txt

# 다른 환경에서 동일하게 설치
pip install -r requirements.txt
```

---

## 4. 설치 확인

설치가 제대로 되었는지 확인합니다.

### 4-1. Python에서 임포트 테스트

```python
# Python 인터프리터 실행
python

# 패키지 임포트 테스트
>>> import pyBGVAR
>>> print(pyBGVAR.__version__)
0.1.0

>>> from pyBGVAR import BGVAR, get_shockinfo
>>> print("설치 성공!")
```

### 4-2. 설치된 버전 확인

```bash
pip show pyBGVAR
```

**출력 예시:**
```
Name: pyBGVAR
Version: 0.1.0
Summary: Python implementation of Bayesian Global Vector Autoregressions
Home-page: https://github.com/your-username/pyBGVAR
Author: Python BGVAR Team
Location: /path/to/site-packages
Requires: numpy, scipy, pandas, matplotlib, seaborn, numba, joblib, openpyxl, mpmath
```

### 4-3. 간단한 실행 테스트

```python
import numpy as np
import pandas as pd
from pyBGVAR import BGVAR

# 간단한 데이터 생성
np.random.seed(42)
data_dict = {
    'US': pd.DataFrame({
        'y': np.random.randn(100).cumsum(),
        'Dp': np.random.randn(100) * 0.5,
    }),
    'EA': pd.DataFrame({
        'y': np.random.randn(100).cumsum(),
        'Dp': np.random.randn(100) * 0.5,
    })
}

W = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], 
                 index=['US', 'EA'], 
                 columns=['US', 'EA'])

# 빠른 테스트 (작은 draws/burnin)
model = BGVAR(Data=data_dict, W=W, plag=1, draws=50, burnin=50)
print("✅ pyBGVAR 설치 및 실행 성공!")
```

---

## 5. 업데이트 및 제거

### 5-1. 패키지 업데이트

#### 최신 버전으로 업데이트:
```bash
pip install --upgrade git+https://github.com/your-username/pyBGVAR.git
```

#### 또는 강제 재설치:
```bash
pip install --force-reinstall git+https://github.com/your-username/pyBGVAR.git
```

#### 개발 모드에서 업데이트:
```bash
cd pyBGVAR
git pull origin main
# 이미 -e로 설치했으므로 재설치 불필요
```

### 5-2. 패키지 제거

```bash
pip uninstall pyBGVAR
```

### 5-3. 의존성까지 완전 제거

```bash
# 1. pyBGVAR 제거
pip uninstall pyBGVAR

# 2. 사용하지 않는 의존성 확인
pip list

# 3. 필요시 의존성 제거
pip uninstall numpy scipy pandas matplotlib seaborn numba joblib openpyxl mpmath
```

---

## 6. 문제 해결

### 문제 1: Git이 설치되지 않음

**증상:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**해결:**
```bash
# Windows: Git 설치
# https://git-scm.com/download/win 에서 다운로드

# macOS:
xcode-select --install

# Linux (Ubuntu/Debian):
sudo apt-get install git

# 확인
git --version
```

### 문제 2: 권한 오류 (Permission Denied)

**증상:**
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**해결:**
```bash
# 방법 1: --user 플래그 사용
pip install --user git+https://github.com/your-username/pyBGVAR.git

# 방법 2: 가상 환경 사용 (권장)
python -m venv bgvar_env
# (가상환경 활성화 후 설치)
```

### 문제 3: 의존성 충돌

**증상:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**해결:**
```bash
# 방법 1: 새로운 가상 환경에서 설치 (권장)
python -m venv fresh_env
source fresh_env/bin/activate  # Windows: fresh_env\Scripts\activate
pip install git+https://github.com/your-username/pyBGVAR.git

# 방법 2: pip 업그레이드
pip install --upgrade pip setuptools wheel
pip install git+https://github.com/your-username/pyBGVAR.git
```

### 문제 4: NumPy/SciPy 컴파일 오류

**증상:**
```
ERROR: Failed building wheel for numpy
```

**해결:**
```bash
# 방법 1: 미리 컴파일된 바이너리 사용
pip install --only-binary :all: numpy scipy
pip install git+https://github.com/your-username/pyBGVAR.git

# 방법 2: Anaconda 사용 (Windows 권장)
conda install numpy scipy pandas
pip install git+https://github.com/your-username/pyBGVAR.git
```

### 문제 5: SSL 인증서 오류

**증상:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**해결:**
```bash
# 임시 해결 (권장하지 않음, 보안 위험)
pip install --trusted-host github.com git+https://github.com/your-username/pyBGVAR.git

# 근본 해결: 인증서 업데이트
# Windows: certifi 재설치
pip install --upgrade certifi

# macOS:
/Applications/Python\ 3.x/Install\ Certificates.command
```

### 문제 6: 메모리 부족 (Large Model)

**증상:**
```
MemoryError
```

**해결:**
```python
# 1. 더 작은 draws/burnin 사용
model = BGVAR(Data=data, W=W, draws=1000, burnin=1000)

# 2. thinning 사용
model = BGVAR(Data=data, W=W, draws=10000, burnin=5000, thin=10)

# 3. 메모리 효율적 옵션
model = BGVAR(Data=data, W=W, draws=5000, burnin=5000, SV=False)

# 4. running mean 사용 (예측 시)
fcast = model.predict(n_ahead=8, save_store=False)
```

### 문제 7: Git 인증 오류 (Private 저장소)

**증상:**
```
fatal: could not read Username for 'https://github.com': No such device or address
exit code: 128
```

**원인:** 
- Private 저장소이거나 Git credential helper 설정 문제

**해결:**
```bash
# 방법 1: 저장소를 클론한 후 로컬 설치 (권장)
git clone https://github.com/nachoidot/pyBGVAR.git
cd pyBGVAR
pip install -e .

# 방법 2: SSH 사용 (SSH 키 설정된 경우)
pip install git+ssh://git@github.com/nachoidot/pyBGVAR.git

# 방법 3: Personal Access Token 사용
# GitHub에서 PAT 생성 후:
pip install git+https://[YOUR_TOKEN]@github.com/nachoidot/pyBGVAR.git

# 방법 4: Credential Helper 설정
git config --global credential.helper manager  # Windows
git config --global credential.helper osxkeychain  # macOS
git config --global credential.helper store  # Linux
```

**저장소가 public이라면:**
- 네트워크/방화벽 설정 확인
- Git 버전 업데이트: `git --version` (2.17 이상 권장)

### 문제 8: Colab에서 경로 오류

**증상:**
```
[Errno 2] No such file or directory: 'pyBGVAR'
ERROR: file:///content does not appear to be a Python project
```

**원인:**
- 저장소 구조가 `BayesianGVAR/pyBGVAR/` 형태인데 잘못된 경로로 이동
- Colab은 `/content`에서 시작하므로 상대 경로 주의 필요

**해결 (Colab):**
```python
# 방법 1: 올바른 경로로 이동
!git clone https://github.com/nachoidot/BayesianGVAR.git
%cd BayesianGVAR/pyBGVAR  # 저장소 안의 pyBGVAR 폴더로 이동
!pip install -e .

# 방법 2: 절대 경로 사용
!git clone https://github.com/nachoidot/BayesianGVAR.git
import os
os.chdir('/content/BayesianGVAR/pyBGVAR')
!pip install -e .

# 방법 3: 현재 위치 확인 후 이동
!pwd  # 현재 경로 확인
!ls -la  # 폴더 구조 확인
%cd BayesianGVAR/pyBGVAR  # 올바른 경로로 이동
!pip install -e .
```

**Colab 완전한 설치 스크립트:**
```python
# Colab 셀에서 실행
!git clone https://github.com/nachoidot/BayesianGVAR.git

# 경로 확인
!ls -la BayesianGVAR/

# pyBGVAR 폴더로 이동
%cd BayesianGVAR/pyBGVAR

# 설치
!pip install -e .

# 확인
import pyBGVAR
print("✅ 설치 성공!")
```

### 문제 9: Numba 경고 메시지

**증상:**
```
NumbaDeprecationWarning: ...
```

**해결:**
```bash
# Numba 업데이트
pip install --upgrade numba

# 또는 경고 무시 (코드 실행에는 문제 없음)
import warnings
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)
```

---

## 7. GitHub 저장소 활용

### 7-1. 버전 확인

#### GitHub에서 최신 릴리즈 확인:
1. https://github.com/your-username/pyBGVAR 방문
2. 우측 "Releases" 섹션 확인
3. 최신 버전 번호 및 변경사항 확인

#### 설치된 버전과 비교:
```bash
pip show pyBGVAR
```

### 7-2. 이슈 리포팅

버그를 발견하거나 기능 요청이 있을 때:

1. https://github.com/your-username/pyBGVAR/issues 방문
2. "New Issue" 클릭
3. 템플릿 작성:

```markdown
**문제 설명**
간단명료하게 문제를 설명합니다.

**재현 방법**
1. 다음 코드 실행
2. 오류 발생

**예상 동작**
원래 어떻게 작동해야 하는지 설명

**실제 동작**
실제로 어떻게 작동하는지 설명

**환경 정보**
- OS: Windows 10
- Python 버전: 3.10
- pyBGVAR 버전: 0.1.0

**재현 코드**
```python
import pyBGVAR
# 오류를 재현하는 최소 코드
```

**에러 메시지**
```
전체 에러 메시지 및 traceback
```
```

### 7-3. 기여하기 (Contributing)

코드 개선이나 버그 수정에 기여하고 싶을 때:

#### Step 1: Fork 및 Clone
```bash
# 1. GitHub에서 Fork 버튼 클릭
# 2. 자신의 저장소로 클론
git clone https://github.com/[당신의-사용자명]/pyBGVAR.git
cd pyBGVAR

# 3. 원본 저장소를 upstream으로 추가
git remote add upstream https://github.com/[원본-사용자명]/pyBGVAR.git
```

#### Step 2: 브랜치 생성 및 수정
```bash
# 1. 새 브랜치 생성
git checkout -b feature/my-improvement

# 2. 코드 수정

# 3. 개발 모드로 설치하여 테스트
pip install -e .

# 4. 변경사항 커밋
git add .
git commit -m "Add: 새로운 기능 추가"
```

#### Step 3: Pull Request
```bash
# 1. 자신의 저장소에 푸시
git push origin feature/my-improvement

# 2. GitHub에서 Pull Request 생성
# 3. 변경사항 설명 작성
```

### 7-4. 최신 코드 동기화

```bash
# 1. 원본 저장소의 변경사항 가져오기
git fetch upstream

# 2. main 브랜치로 병합
git checkout main
git merge upstream/main

# 3. 자신의 GitHub에 푸시
git push origin main
```

### 7-5. 문서 및 예제

#### 저장소에서 찾을 수 있는 자료:
- **README.md**: 패키지 개요 및 기본 사용법
- **QUICKSTART.md**: 5분 빠른 시작 가이드
- **example_usage.py**: 상세한 사용 예제 (모든 기능 포함)
- **pyBGVAR/**: 소스 코드 (각 함수에 docstring 포함)

#### 예제 다운로드 및 실행:
```bash
# 1. 예제 파일 다운로드
curl -O https://raw.githubusercontent.com/your-username/pyBGVAR/main/example_usage.py

# 2. 실행
python example_usage.py
```

---

## 빠른 참조 가이드

### 일반 사용자 (연구/분석)

```bash
# 1. 가상 환경 생성 및 활성화
python -m venv bgvar_env
source bgvar_env/bin/activate  # Windows: bgvar_env\Scripts\activate

# 2. pyBGVAR 설치 (안정 버전)
pip install git+https://github.com/your-username/pyBGVAR.git@v0.1.0

# 3. 설치 확인
python -c "import pyBGVAR; print('Success!')"

# 4. 예제 실행
python example_usage.py
```

### 개발자/기여자

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/pyBGVAR.git
cd pyBGVAR

# 2. 가상 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 개발 모드 설치
pip install -e .
pip install -e ".[dev]"  # 개발 도구 포함

# 4. 테스트 실행 (있는 경우)
pytest
```

---

## 다음 단계

✅ 설치 완료 후:
1. **QUICKSTART.md** 읽기 - 5분 만에 시작
2. **example_usage.py** 실행 - 모든 기능 체험
3. **README.md** 정독 - API 전체 이해
4. **자신의 데이터로 분석 시작**

## 도움이 필요하신가요?

- 🐛 **버그 리포트**: [GitHub Issues](https://github.com/your-username/pyBGVAR/issues)
- 💬 **질문 및 토론**: [GitHub Discussions](https://github.com/your-username/pyBGVAR/discussions)
- 📧 **이메일**: your.email@example.com
- 📚 **원본 R 패키지 문서**: [BGVAR on CRAN](https://cran.r-project.org/package=BGVAR)

---

**행운을 빕니다!** 🚀

_마지막 업데이트: 2025년 11월_

