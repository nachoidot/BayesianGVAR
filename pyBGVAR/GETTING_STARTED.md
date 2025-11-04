# pyBGVAR 시작 가이드

> GitHub에서 pyBGVAR를 설치하고 사용하는 모든 것

## 🚀 한눈에 보는 설치 및 사용

### 1단계: 설치 (2분)

```bash
# 가상 환경 생성 (권장)
python -m venv bgvar_env

# 활성화
source bgvar_env/bin/activate  # Windows: bgvar_env\Scripts\activate

# pyBGVAR 설치
pip install git+https://github.com/[사용자명]/pyBGVAR.git
```

### 2단계: 첫 실행 (3분)

```python
import numpy as np
import pandas as pd
from pyBGVAR import BGVAR, get_shockinfo

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

# 가중치 행렬
W = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], 
                 index=['US', 'EA'], columns=['US', 'EA'])

# 모델 추정
model = BGVAR(Data=data_dict, W=W, plag=1, draws=100, burnin=100)

# IRF 계산
shockinfo = get_shockinfo(ident="chol", nr_rows=1)
shockinfo.loc[0, 'shock'] = 'US.y'
irf_result = model.irf(n_ahead=24, shockinfo=shockinfo)

print("✅ 성공!")
```

### 3단계: 더 알아보기

실행이 성공했다면, 이제 본격적으로 시작할 준비가 되었습니다!

---

## 📚 문서 가이드

### 당신의 목적에 맞는 가이드를 선택하세요

#### 🎯 **처음 시작하는 분**
→ **[QUICKSTART.md](QUICKSTART.md)** (5분)
- 최소한의 코드로 빠르게 시작
- 주요 기능 간단 예제
- FAQ 포함

#### 🔧 **설치 관련 문제가 있는 분**
→ **[GITHUB_INSTALLATION_GUIDE.md](GITHUB_INSTALLATION_GUIDE.md)** (10분)
- 다양한 설치 방법 상세 설명
- 문제 해결 가이드
- GitHub 저장소 활용법

#### 📊 **실전 분석을 하려는 분**
→ **[TUTORIAL.md](TUTORIAL.md)** (30-60분)
- 완전한 프로젝트 예제
- 실제 데이터 작업 방법
- 고급 기능 활용
- 결과 해석 및 보고

#### 📖 **전체 기능을 알고 싶은 분**
→ **[README.md](README.md)** (15분)
- 모든 기능 리스트
- API 개요
- 기본 사용 예제

#### 💻 **코드로 배우고 싶은 분**
→ **[example_usage.py](example_usage.py)** (실행)
- 모든 기능을 포함한 실행 가능한 예제
- 주석으로 상세 설명
```bash
python example_usage.py
```

---

## 🗺️ 학습 경로 추천

### 경로 A: 빠른 시작 (초보자)

```
1. QUICKSTART.md 읽기 (5분)
   ↓
2. 간단한 예제 실행 (5분)
   ↓
3. example_usage.py 실행 (10분)
   ↓
4. 자신의 데이터로 시도
```

### 경로 B: 체계적 학습 (중급자)

```
1. README.md 전체 읽기 (15분)
   ↓
2. GITHUB_INSTALLATION_GUIDE.md로 환경 완벽 설정 (10분)
   ↓
3. TUTORIAL.md 따라하기 (60분)
   ↓
4. 실제 프로젝트 시작
```

### 경로 C: 실전 바로 시작 (고급자)

```
1. GitHub에서 설치
   ↓
2. 자신의 데이터 준비
   ↓
3. TUTORIAL.md의 관련 섹션만 참고
   ↓
4. 문제 발생 시 GITHUB_INSTALLATION_GUIDE.md 참고
```

---

## 🎓 학습 체크리스트

### 기본 (필수)
- [ ] pyBGVAR 설치 완료
- [ ] 간단한 예제 실행 성공
- [ ] BGVAR 모델 추정 이해
- [ ] IRF 계산 및 해석

### 중급
- [ ] 자신의 데이터로 모델 추정
- [ ] 다양한 prior 비교
- [ ] FEVD 분석
- [ ] 예측 및 평가

### 고급
- [ ] 부호 제약 IRF 구현
- [ ] 조건부 예측 활용
- [ ] 모델 비교 및 선택
- [ ] 출판용 그래프 생성

---

## 🔍 주요 기능 빠른 참조

### 모델 추정

```python
from pyBGVAR import BGVAR

model = BGVAR(
    Data=data_dict,      # 국가별 데이터
    W=weight_matrix,     # 가중치 행렬
    plag=2,              # 시차
    draws=5000,          # MCMC 추출
    burnin=5000,         # Burn-in
    prior="NG",          # Prior 선택
    SV=True              # Stochastic Volatility
)
```

### IRF 분석

```python
from pyBGVAR import get_shockinfo

# Cholesky
shockinfo = get_shockinfo(ident="chol", nr_rows=1)
shockinfo.loc[0, 'shock'] = 'US.y'
irf_result = model.irf(n_ahead=24, shockinfo=shockinfo)
```

### FEVD

```python
from pyBGVAR import compute_fevd

fevd_result = compute_fevd(irf_result, var_slct=['US.y', 'EA.y'])
```

### 예측

```python
fcast = model.predict(n_ahead=8, save_store=True)
```

### 시각화

```python
from pyBGVAR import plot

plot.plot_irf(irf_result, resp=['US.y'], shock=0)
plot.plot_fevd(fevd_result, resp='US.y')
plot.plot_pred(fcast, resp=['US.y'])
```

---

## ❓ 자주 묻는 질문

### Q: 어떤 Python 버전이 필요한가요?
**A:** Python 3.8 이상

### Q: 설치가 안 됩니다!
**A:** [GITHUB_INSTALLATION_GUIDE.md](GITHUB_INSTALLATION_GUIDE.md)의 "문제 해결" 섹션 참고

### Q: 어떤 prior를 사용해야 하나요?
**A:** 
- 일반적: `NG` (Normal-Gamma)
- 빠른 추정: `MN` (Minnesota)
- 변수 선택: `SSVS`
- 희소성: `HS` (Horseshoe)

### Q: draws와 burnin은 얼마로 설정하나요?
**A:** 
- 테스트: 100-500
- 실제 분석: 5000-10000
- 논문: 10000+ (with thinning)

### Q: 에러가 발생합니다!
**A:** 
1. `verbose=True` 설정하여 진행 상황 확인
2. draws/burnin 늘리기
3. [GITHUB_INSTALLATION_GUIDE.md](GITHUB_INSTALLATION_GUIDE.md)의 문제 해결 섹션 확인
4. GitHub Issues에 질문

### Q: 결과를 어떻게 해석하나요?
**A:** [TUTORIAL.md](TUTORIAL.md)의 "5. 결과 해석 및 보고" 섹션 참고

---

## 🛠️ 개발 환경 권장사항

### 필수
- Python 3.8+
- pip (최신 버전)
- Git

### 권장
- **IDE**: VSCode, PyCharm, Jupyter Lab
- **가상 환경**: venv 또는 conda
- **버전 관리**: Git

### 선택적
- Jupyter Notebook (대화형 분석)
- LaTeX (논문 작성)

---

## 📞 도움이 필요하신가요?

### 버그 리포트 및 이슈
🐛 [GitHub Issues](https://github.com/[사용자명]/pyBGVAR/issues)

### 질문 및 토론
💬 [GitHub Discussions](https://github.com/[사용자명]/pyBGVAR/discussions)

### 기여
🤝 [기여 가이드](GITHUB_INSTALLATION_GUIDE.md#73-기여하기-contributing)

### 원본 R 패키지
📦 [BGVAR on CRAN](https://cran.r-project.org/package=BGVAR)

---

## 📄 라이선스

이 프로젝트는 GNU General Public License v3.0 하에 배포됩니다.

---

## 🎉 마치며

pyBGVAR를 선택해주셔서 감사합니다!

이 패키지가 여러분의 연구와 분석에 도움이 되기를 바랍니다.

**행운을 빕니다!** 🚀

---

_최종 업데이트: 2025년 11월_

<div align="center">
  
Made with ❤️ by Python BGVAR Team

[⭐ Star on GitHub](https://github.com/[사용자명]/pyBGVAR) | [📖 Documentation](README.md) | [🐛 Report Bug](https://github.com/[사용자명]/pyBGVAR/issues)

</div>

