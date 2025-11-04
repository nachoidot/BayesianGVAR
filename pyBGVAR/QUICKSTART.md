# pyBGVAR 빠른 시작 가이드

## 5분 만에 시작하기

### 1. 설치

#### 방법 A: GitHub에서 직접 설치 (권장)

```bash
# 최신 버전 설치
pip install git+https://github.com/[사용자명]/pyBGVAR.git

# 또는 특정 버전 (안정)
pip install git+https://github.com/[사용자명]/pyBGVAR.git@v0.1.0
```

#### 방법 B: 로컬 개발 모드 (개발자용)

```bash
# 1. 저장소 클론
git clone https://github.com/[사용자명]/pyBGVAR.git
cd pyBGVAR

# 2. 개발 모드 설치
pip install -e .
```

> 💡 **Tip**: 가상 환경 사용을 권장합니다!
> ```bash
> python -m venv bgvar_env
> source bgvar_env/bin/activate  # Windows: bgvar_env\Scripts\activate
> pip install git+https://github.com/[사용자명]/pyBGVAR.git
> ```

**자세한 설치 가이드**: [GITHUB_INSTALLATION_GUIDE.md](GITHUB_INSTALLATION_GUIDE.md) 참고

### 2. 최소 실행 예제

```python
import numpy as np
import pandas as pd
from pyBGVAR import BGVAR, get_shockinfo

# 데이터 준비 (예제)
np.random.seed(42)
data_dict = {
    'US': pd.DataFrame({
        'y': np.random.randn(100).cumsum(),
        'Dp': np.random.randn(100) * 0.5,
        'stir': np.random.randn(100) * 0.3 + 2
    }),
    'EA': pd.DataFrame({
        'y': np.random.randn(100).cumsum(),
        'Dp': np.random.randn(100) * 0.5,
        'stir': np.random.randn(100) * 0.3 + 2
    })
}

# 가중치 행렬
W = pd.DataFrame(
    [[0.0, 1.0], [1.0, 0.0]],
    index=['US', 'EA'],
    columns=['US', 'EA']
)

# 모델 추정
model = BGVAR(
    Data=data_dict,
    W=W,
    plag=1,
    draws=100,
    burnin=100,
    prior="NG"
)

# IRF 계산
shockinfo = get_shockinfo(ident="chol", nr_rows=1)
shockinfo.loc[0, 'shock'] = 'US.y'
irf_result = model.irf(n_ahead=24, shockinfo=shockinfo)

print("완료!")
```

## 주요 기능 사용법

### 모델 추정 + 요약

```python
model = BGVAR(Data=data_dict, W=W, plag=1, draws=5000, burnin=5000)
summary = model.summary()
coefs = model.coef()
```

### IRF (3가지 방법)

```python
# 1. Cholesky
shockinfo = get_shockinfo(ident="chol", nr_rows=1)
shockinfo.loc[0, 'shock'] = 'US.y'
irf_chol = model.irf(n_ahead=24, shockinfo=shockinfo)

# 2. GIRF
shockinfo = get_shockinfo(ident="girf", nr_rows=1)
shockinfo.loc[0, 'shock'] = 'EA.Dp'
irf_girf = model.irf(n_ahead=24, shockinfo=shockinfo)

# 3. 부호 제약
from pyBGVAR import add_shockinfo
shockinfo = add_shockinfo(
    None,
    shock='US.stir',
    restriction=['US.y', 'US.Dp'],
    sign=['-', '+'],
    horizon=5
)
irf_sign = model.irf(n_ahead=24, shockinfo=shockinfo)
```

### FEVD / GFEVD

```python
from pyBGVAR import compute_fevd, gfevd

# 일반 FEVD
fevd_result = compute_fevd(irf_result)

# 일반화 FEVD
gfevd_result = gfevd(model, n_ahead=24)
```

### 예측

```python
# 무조건부 예측
fcast = model.predict(n_ahead=8, save_store=True)

# 조건부 예측
constr = np.zeros((8, K))  # K = 변수 수
constr[:, var_idx] = [...]  # 특정 변수 경로 고정
fcast_cond = model.predict(n_ahead=8, constr=constr)
```

### 예측 평가

```python
from pyBGVAR import lps, rmse

# Hold-out 샘플로 모델 추정
model = BGVAR(Data=data_dict, W=W, plag=1, draws=5000, burnin=5000, hold_out=8)
fcast = model.predict(n_ahead=8, save_store=True)

# 평가
lps_scores = lps(fcast)
rmse_scores = rmse(fcast)
```

### 진단

```python
from pyBGVAR import conv_diag, resid_corr_test, avg_pair_cc

conv_result = conv_diag(model)
resid_test = resid_corr_test(model)
avg_corr = avg_pair_cc(model)
```

### 시각화

```python
from pyBGVAR import plot
import matplotlib.pyplot as plt

plot.plot_irf(irf_result, resp=['US.y'], shock=0)
plt.savefig('irf.png')

plot.plot_fevd(fevd_result, resp='US.y')
plt.savefig('fevd.png')

plot.plot_pred(fcast, resp=['US.y'])
plt.savefig('forecast.png')
```

## 실전 팁

### 1. 실제 데이터로 작업하기

```python
# Excel에서 읽기
from pyBGVAR import excel_to_list
data_dict = excel_to_list('data.xlsx')

# CSV에서 읽기
import pandas as pd
data_matrix = pd.read_csv('data.csv', index_col=0)
from pyBGVAR import matrix_to_list
data_dict = matrix_to_list(data_matrix)
```

### 2. Prior 선택

```python
# Minnesota prior (고전적, 빠름)
model = BGVAR(Data=data, W=W, prior="MN", SV=False)

# SSVS (변수 선택, 중간)
model = BGVAR(Data=data, W=W, prior="SSVS", SV=True)

# Normal-Gamma (권장, 균형)
model = BGVAR(Data=data, W=W, prior="NG", SV=True)

# Horseshoe (희소성, 느림)
model = BGVAR(Data=data, W=W, prior="HS", SV=True)
```

### 3. MCMC 설정

```python
# 빠른 테스트용
model = BGVAR(Data=data, W=W, draws=100, burnin=100)

# 실제 분석용 (권장)
model = BGVAR(Data=data, W=W, draws=5000, burnin=5000)

# 큰 모델 (메모리 효율)
model = BGVAR(Data=data, W=W, draws=10000, burnin=5000, thin=10)
```

### 4. 병렬 처리

```python
# IRF 계산 시 병렬 처리 (구현 예정)
# irf_result = model.irf(n_ahead=24, shockinfo=shockinfo, cores=4)
```

### 5. 결과 저장

```python
import pickle

# 모델 저장
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 모델 불러오기
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## 자주 묻는 질문 (FAQ)

### Q1. 가중치 행렬 W는 어떻게 만들어야 하나요?

**A:** 경제적 연결성을 반영해야 합니다:
- 무역 가중치: 양자 간 무역량 비중
- GDP 가중치: 상대 GDP 크기
- 금융 가중치: 금융 시장 연결성

```python
# 예: 무역 가중치
trade_shares = pd.DataFrame(...)  # 무역 데이터
W = trade_shares / trade_shares.sum(axis=1, keepdims=True)
W.values[np.arange(len(W)), np.arange(len(W))] = 0  # 대각선 0
```

### Q2. 얼마나 많은 draws가 필요한가요?

**A:** 
- 테스트: 100-500
- 실제 분석: 5000-10000
- 발표/논문: 10000+ (with thinning)

### Q3. 어떤 prior를 선택해야 하나요?

**A:**
- 기본: Normal-Gamma (NG) - 균형잡힌 성능
- 변수 많음: Horseshoe (HS) - 희소성
- 빠른 추정: Minnesota (MN) - 간단

### Q4. IRF 식별 방법은 어떻게 선택하나요?

**A:**
- Cholesky: 간단, 순서 중요
- GIRF: 순서 무관, 해석 쉬움
- Sign restrictions: 경제 이론 반영

### Q5. 에러가 발생하면?

**A:**
```python
# verbose=True로 진행 상황 확인
model = BGVAR(Data=data, W=W, verbose=True)

# 안정성 검사 비활성화 (불안정한 추출 허용)
model = BGVAR(Data=data, W=W, eigen=False)

# 더 많은 burnin
model = BGVAR(Data=data, W=W, burnin=10000)
```

## 다음 단계

### 📚 더 자세히 알아보기

1. **완전 초보자**:
   - [GETTING_STARTED.md](GETTING_STARTED.md) - 전체 시작 가이드
   - `python example_usage.py` - 전체 예제 실행

2. **설치 문제가 있나요?**:
   - [GITHUB_INSTALLATION_GUIDE.md](GITHUB_INSTALLATION_GUIDE.md) - 상세 설치 및 문제 해결

3. **실전 분석 시작**:
   - [TUTORIAL.md](TUTORIAL.md) - 완전한 프로젝트 예제
   - 데이터 준비부터 결과 보고까지

4. **전체 기능 탐색**:
   - [README.md](README.md) - 모든 기능 API 문서

5. **문서 가이드**:
   - [문서_가이드_요약.md](문서_가이드_요약.md) - 어떤 문서를 읽어야 할지 모르겠다면

## 도움말

- 🐛 **버그 리포트**: [GitHub Issues](https://github.com/[사용자명]/pyBGVAR/issues)
- 💬 **질문 및 토론**: [GitHub Discussions](https://github.com/[사용자명]/pyBGVAR/discussions)
- 🤝 **기여하기**: [기여 가이드](GITHUB_INSTALLATION_GUIDE.md#73-기여하기-contributing)
- ⭐ **프로젝트 지원**: [GitHub Star](https://github.com/[사용자명]/pyBGVAR)

---

**행운을 빕니다!** 🚀

