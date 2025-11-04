# pyBGVAR: Python Implementation of Bayesian Global Vector Autoregressions

Python 포팅 버전의 Bayesian Global Vector Autoregression (BGVAR) 패키지입니다.

이 패키지는 원본 R 패키지 BGVAR의 Python 구현체입니다.

## 설치

```bash
pip install -e .
```

## 주요 기능

### 모델 추정
- **BGVAR 모델 추정**: 다양한 prior 설정(Minnesota, SSVS, NG, Horseshoe)을 사용한 Bayesian GVAR 추정
- **Stochastic Volatility**: 시변 분산 모델링 지원

### 동적 분석
- **Impulse Response Functions (IRF)**: 
  - Cholesky 분해
  - 일반화 IRF (GIRF)
  - 부호/제로 제약 (Sign/Zero restrictions)
- **Forecast Error Variance Decomposition (FEVD)**: 예측 오차 분산 분해
- **Generalized FEVD (GFEVD)**: Lanne-Nyberg (2016) 보정 GFEVD
- **Historical Decomposition (HD)**: 역사적 시계열 분해 분석

### 예측 및 평가
- **Predictions**: 조건부/비조건부 예측
- **Log-Predictive Scores (LPS)**: 로그 예측 밀도 점수
- **Root Mean Square Error (RMSE)**: 평균 제곱근 오차

### 모델 진단
- **Convergence Diagnostics**: Geweke 검정을 사용한 수렴 진단
- **Residual Autocorrelation Test**: 잔차 자기상관 F-검정
- **Average Pairwise Correlations**: 평균 쌍별 교차 상관계수
- **Deviance Information Criterion (DIC)**: 모델 선택 기준

### S3 메서드 스타일 함수
- `summary()`: 모델 요약 통계
- `coef()`: 계수 추출
- `vcov()`: 분산-공분산 행렬
- `fitted()`: 적합값
- `residuals()`: 잔차
- `logLik()`: 로그 가능도

### 유틸리티 함수
- `get_shockinfo()`: 충격 정보 데이터프레임 생성
- `add_shockinfo()`: 부호 제약 추가
- `matrix_to_list()`: 행렬을 국가별 딕셔너리로 변환
- `list_to_matrix()`: 국가별 딕셔너리를 행렬로 변환
- `excel_to_list()`: Excel 파일에서 데이터 읽기

## 사용 예제

### 기본 모델 추정

```python
import numpy as np
import pandas as pd
from pyBGVAR import BGVAR, get_shockinfo, add_shockinfo

# 데이터 준비
# Data는 dictionary 형태: {'US': DataFrame, 'EA': DataFrame, ...}
# 또는 DataFrame with columns 'COUNTRY.VARIABLE'
# W는 weight matrix (국가 간 가중치 행렬)

model = BGVAR(
    Data=testdata,
    W=W_test,
    plag=1,
    draws=5000,
    burnin=5000,
    prior="NG",
    SV=True,
    hold_out=0
)

# 모델 요약
summary = model.summary()

# 계수 및 통계량 추출
coefs = model.coef(quantile=0.50)
vcov_mat = model.vcov(quantile=0.50)
fitted_vals = model.fitted(global_model=True)
residuals_dict = model.residuals()

# 모델 선택 기준
dic_result = model.dic()
loglik = model.logLik(quantile=0.50)
```

### Impulse Response Functions

```python
# Cholesky 분해를 사용한 IRF
shockinfo = get_shockinfo(ident="chol", nr_rows=1)
shockinfo.loc[0, 'shock'] = 'US.y'
shockinfo.loc[0, 'scale'] = 1

irf_result = model.irf(n_ahead=24, shockinfo=shockinfo)

# 부호 제약을 사용한 IRF
shockinfo_sign = get_shockinfo(ident="sign", nr_rows=1)
shockinfo_sign = add_shockinfo(
    shockinfo_sign,
    shock='US.y',
    restriction=['US.y', 'US.Dp'],
    sign=['+', '+'],
    horizon=5,
    prob=0.5
)

irf_sign = model.irf(n_ahead=24, shockinfo=shockinfo_sign)
```

### Forecast Error Variance Decomposition

```python
from pyBGVAR import compute_fevd, gfevd

# 일반 FEVD
fevd_result = compute_fevd(irf_result, var_slct=['US.y', 'EA.y'])

# 일반화 FEVD (Generalized FEVD)
gfevd_result = gfevd(model, n_ahead=24, running=True)
```

### Historical Decomposition

```python
from pyBGVAR import hd

# 역사적 분해
hd_result = hd(irf_result, var_slct=['US.y', 'EA.y'])
```

### Prediction and Evaluation

```python
# 예측
fcast = model.predict(n_ahead=8, save_store=True)

# Hold-out 샘플이 있는 경우 예측 평가
model_holdout = BGVAR(
    Data=testdata,
    W=W_test,
    plag=1,
    draws=5000,
    burnin=5000,
    prior="NG",
    hold_out=8
)

fcast_eval = model_holdout.predict(n_ahead=8, save_store=True)

# 예측 평가 지표
from pyBGVAR import lps, rmse

lps_scores = lps(fcast_eval)
rmse_scores = rmse(fcast_eval)
```

### Diagnostics

```python
from pyBGVAR import conv_diag, resid_corr_test, avg_pair_cc

# 수렴 진단
convergence = conv_diag(model, crit_val=1.96)

# 잔차 자기상관 검정
resid_test = resid_corr_test(model, lag_cor=1, alpha=0.95)

# 평균 쌍별 상관계수
avg_corr = avg_pair_cc(model, digits=3)
```

### Plotting

```python
from pyBGVAR import plot

# IRF 플롯
plot.plot_irf(irf_result, resp=['US.y', 'EA.y'], shock=1)

# FEVD 플롯
plot.plot_fevd(fevd_result, resp='US.y')

# 예측 플롯
plot.plot_pred(fcast, resp=['US.y', 'EA.y'])
```

## 참고문헌

Boeck, M., Feldkircher, M. and F. Huber (2022) BGVAR: Bayesian Global Vector Autoregressions with Shrinkage Priors in R. *Journal of Statistical Software*, Vol. 104(9), pp. 1-28.

