# pyBGVAR 구현 요약

## 완료된 모듈

### 1. 핵심 모듈
- ✅ **bgvar.py**: 메인 BGVAR 클래스
  - 데이터 처리 및 검증
  - Weight matrix 처리
  - Hyperparameter 설정
  - 국가별 모델 추정 통합
  - IRF, FEVD, HD, Predict 메서드 연결

### 2. BVAR 추정 모듈
- ✅ **bvar.py**: Bayesian VAR MCMC 샘플러
  - 4가지 Prior 지원: MN, SSVS, NG, HS
  - MCMC Gibbs 샘플링 구현
  - Stochastic Volatility 지원 (기본 구현)
  - Homoskedastic case 구현
  - Posterior 저장 구조

### 3. GVAR Stacking 모듈
- ✅ **stacking.py**: 국가 모델 → 글로벌 모델 변환
  - Companion matrix 계산
  - Eigenvalue 검증 및 trimming
  - G matrix 계산 및 역행렬
  - 글로벌 계수 추출

### 4. 후처리 함수들
- ✅ **irf.py**: Impulse Response Functions
  - Cholesky identification
  - GIRF (Generalized IRF)
  - Sign restrictions (기본 구조)
  
- ✅ **fevd.py**: Forecast Error Variance Decomposition
  
- ✅ **hd.py**: Historical Decomposition
  
- ✅ **predict.py**: 예측 함수
  - 비조건부 예측
  - 조건부 예측 (기본 구조)

### 5. 유틸리티 모듈
- ✅ **utils.py**: 데이터 변환 함수들
  - `matrix_to_list()`, `list_to_matrix()`
  - `mlag()`: lag 생성
  - 데이터 형식 검증

- ✅ **helpers.py**: 헬퍼 함수들
  - `get_weights()`: Weight matrix 구성
  - `get_companion()`: Companion matrix
  - `get_V()`: Minnesota prior variance

- ✅ **diagnostics.py**: 진단 함수들
  - `conv_diag()`: 수렴 진단
  - `resid_corr_test()`: 잔차 상관 검정
  - `avg_pair_cc()`: 교차 상관 분석

### 6. 시각화 모듈
- ✅ **plot.py**: 플롯 함수들
  - `plot_bgvar()`: 모델 요약
  - `plot_irf()`: IRF 플롯
  - `plot_fevd()`: FEVD 플롯
  - `plot_hd()`: HD 플롯
  - `plot_pred()`: 예측 플롯

## 주요 특징

### Prior 구현 상태
- **MN (Minnesota)**: ✅ 기본 구현 완료
- **SSVS**: ✅ 기본 구조 구현 (Gamma, Omega 샘플링)
- **NG (Normal-Gamma)**: ✅ 기본 구조 구현 (GIG 샘플링 필요)
- **HS (Horseshoe)**: ✅ 기본 구조 구현 (전체 파라미터 업데이트 필요)

### Stochastic Volatility
- ✅ 기본 구조 구현 (Homoskedastic 완전 구현)
- ⚠️ 완전한 SV 샘플러는 추가 구현 필요 (stochvol 패키지 대체 필요)

## 사용 예제

```python
import pandas as pd
import numpy as np
from pyBGVAR import BGVAR

# 데이터 준비
data = {
    'US': pd.DataFrame({'y': [1, 2, 3, ...], 'Dp': [0.5, 0.6, 0.7, ...]}),
    'EA': pd.DataFrame({'y': [1.1, 2.1, 3.1, ...], 'Dp': [0.4, 0.5, 0.6, ...]})
}

# Weight matrix
W = np.array([[0.0, 1.0], [1.0, 0.0]])

# 모델 추정
model = BGVAR(
    data=data,
    W=W,
    plag=1,
    draws=1000,
    burnin=1000,
    prior='NG',
    SV=False,
    verbose=True
)

# IRF 계산
shockinfo = get_shockinfo('chol')
shockinfo = add_shockinfo(shockinfo, shock='US.y', scale=1.0)
irf_result = model.irf(n_ahead=24, shockinfo=shockinfo)

# FEVD
fevd_result = model.fevd(irf_result)

# 예측
pred_result = model.predict(n_ahead=8)
```

## 추가 구현 필요 사항

1. **완전한 MCMC 샘플러**
   - NG prior의 GIG 샘플링 완전 구현
   - HS prior의 전체 파라미터 업데이트
   - Stochastic Volatility 완전 구현

2. **성능 최적화**
   - Numba JIT 컴파일 적용
   - 병렬 처리 최적화

3. **테스트 및 검증**
   - R 버전과 결과 비교
   - 단위 테스트 작성

4. **문서화**
   - API 문서 생성
   - 사용 예제 보강

## 파일 구조

```
pyBGVAR/
├── setup.py
├── requirements.txt
├── README.md
├── MANIFEST.in
└── pyBGVAR/
    ├── __init__.py
    ├── bgvar.py          # 메인 클래스
    ├── bvar.py           # BVAR MCMC
    ├── stacking.py       # GVAR stacking
    ├── irf.py            # IRF
    ├── fevd.py           # FEVD
    ├── hd.py             # Historical Decomposition
    ├── predict.py        # 예측
    ├── plot.py           # 시각화
    ├── diagnostics.py    # 진단 함수
    ├── helpers.py        # 헬퍼 함수
    └── utils.py          # 유틸리티
```

## 참고사항

- 이 구현은 R 패키지의 Python 포팅입니다
- 일부 고급 기능은 기본 구조만 구현되어 있으며, 완전한 구현이 필요할 수 있습니다
- MCMC 샘플러는 R 버전과 동일한 알고리즘을 따르지만, 세부 구현은 다를 수 있습니다
- 테스트를 통해 R 버전과의 결과 일치를 확인하는 것이 권장됩니다

