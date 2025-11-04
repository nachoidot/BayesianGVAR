# pyBGVAR 실전 튜토리얼

> GitHub에서 설치한 pyBGVAR를 실제 연구/분석에 활용하는 완벽 가이드

## 목차
1. [시작하기 전에](#1-시작하기-전에)
2. [첫 번째 프로젝트: 간단한 GVAR 분석](#2-첫-번째-프로젝트-간단한-gvar-분석)
3. [실제 데이터로 분석하기](#3-실제-데이터로-분석하기)
4. [고급 기능 활용](#4-고급-기능-활용)
5. [결과 해석 및 보고](#5-결과-해석-및-보고)
6. [성능 최적화](#6-성능-최적화)

---

## 1. 시작하기 전에

### 1-1. 환경 준비 체크리스트

```bash
# 1. Python 버전 확인 (3.8 이상)
python --version

# 2. 가상 환경 생성 및 활성화
python -m venv myproject_env
source myproject_env/bin/activate  # Windows: myproject_env\Scripts\activate

# 3. pyBGVAR 설치
pip install git+https://github.com/[사용자명]/pyBGVAR.git

# 4. 설치 확인
python -c "import pyBGVAR; print('✅ 설치 성공!')"
```

### 1-2. 프로젝트 폴더 구조

```
my_bgvar_project/
├── data/
│   ├── raw/              # 원본 데이터
│   └── processed/        # 전처리된 데이터
├── scripts/
│   ├── 01_data_prep.py   # 데이터 준비
│   ├── 02_estimation.py  # 모델 추정
│   ├── 03_analysis.py    # 분석 및 시각화
│   └── utils.py          # 유틸리티 함수
├── results/
│   ├── models/           # 저장된 모델
│   ├── figures/          # 그래프
│   └── tables/           # 표
├── notebooks/            # Jupyter notebooks
└── requirements.txt
```

---

## 2. 첫 번째 프로젝트: 간단한 GVAR 분석

### 2-1. 데이터 준비 (01_data_prep.py)

```python
"""
데이터 준비 스크립트
간단한 예제 데이터를 생성합니다.
"""
import numpy as np
import pandas as pd
import pickle

def create_example_data():
    """예제 데이터 생성"""
    np.random.seed(42)
    
    # 시뮬레이션 파라미터
    T = 200  # 시계열 길이 (약 16년 분기 데이터)
    countries = ['US', 'EA', 'UK', 'JP']
    variables = ['y', 'Dp', 'stir']  # GDP, 인플레이션, 단기금리
    
    # 국가별 데이터 생성
    data_dict = {}
    for country in countries:
        # GDP: 추세 + 랜덤워크
        y = 100 + np.linspace(0, 50, T) + np.random.randn(T).cumsum() * 0.5
        
        # 인플레이션: 평균회귀 과정
        Dp = 2 + np.random.randn(T) * 0.5
        for t in range(1, T):
            Dp[t] = 0.7 * Dp[t-1] + 0.3 * 2 + np.random.randn() * 0.5
        
        # 단기금리: GDP와 인플레이션의 함수
        stir = 1 + 0.01 * (y - y[0]) + 0.5 * Dp + np.random.randn(T) * 0.3
        
        data_dict[country] = pd.DataFrame({
            'y': y,
            'Dp': Dp,
            'stir': stir
        })
    
    return data_dict, countries

def create_weight_matrix(countries):
    """가중치 행렬 생성 (무역 가중치 시뮬레이션)"""
    N = len(countries)
    
    # 예제: 무역 가중치 (실제로는 실제 데이터 사용)
    W = np.array([
        [0.0, 0.4, 0.3, 0.3],  # US
        [0.5, 0.0, 0.3, 0.2],  # EA
        [0.4, 0.4, 0.0, 0.2],  # UK
        [0.3, 0.3, 0.2, 0.0]   # JP
    ])
    
    # 정규화 (각 행의 합이 1)
    W = W / W.sum(axis=1, keepdims=True)
    
    return pd.DataFrame(W, index=countries, columns=countries)

if __name__ == '__main__':
    print("=" * 60)
    print("데이터 준비")
    print("=" * 60)
    
    # 데이터 생성
    data_dict, countries = create_example_data()
    W = create_weight_matrix(countries)
    
    # 저장
    with open('data/processed/data_dict.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    
    W.to_csv('data/processed/weight_matrix.csv')
    
    print(f"\n✅ 데이터 생성 완료")
    print(f"   - 국가 수: {len(countries)}")
    print(f"   - 변수 수: {data_dict['US'].shape[1]}")
    print(f"   - 시계열 길이: {data_dict['US'].shape[0]}")
    print(f"\n가중치 행렬:")
    print(W)
```

### 2-2. 모델 추정 (02_estimation.py)

```python
"""
BGVAR 모델 추정 스크립트
"""
import pickle
import pandas as pd
from pyBGVAR import BGVAR
import time

def estimate_bgvar_model(data_dict, W, draws=5000, burnin=5000):
    """BGVAR 모델 추정"""
    print("=" * 60)
    print("BGVAR 모델 추정 시작")
    print("=" * 60)
    
    start_time = time.time()
    
    # 모델 추정
    model = BGVAR(
        Data=data_dict,
        W=W,
        plag=2,              # 시차 2
        draws=draws,         # MCMC 추출 수
        burnin=burnin,       # Burn-in
        prior="NG",          # Normal-Gamma prior
        SV=True,             # Stochastic Volatility
        hold_out=0,          # Hold-out 샘플
        eigen=True,          # 안정성 검사
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ 추정 완료! (소요 시간: {elapsed:.2f}초)")
    
    return model

def save_model_summary(model, filename='results/models/model_summary.txt'):
    """모델 요약 저장"""
    summary = model.summary()
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("BGVAR 모델 요약\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"국가 수: {model.N}\n")
        f.write(f"전역 변수 수: {model.xglobal.shape[1]}\n")
        f.write(f"시계열 길이: {model.xglobal.shape[0]}\n")
        f.write(f"Prior: {model.prior}\n")
        f.write(f"MCMC 추출 수: {model.args.get('thindraws', 0)}\n\n")
        
    print(f"✅ 모델 요약 저장: {filename}")

if __name__ == '__main__':
    # 데이터 로드
    with open('data/processed/data_dict.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    W = pd.read_csv('data/processed/weight_matrix.csv', index_col=0)
    
    # 모델 추정 (테스트용 작은 draws)
    model = estimate_bgvar_model(data_dict, W, draws=1000, burnin=1000)
    
    # 모델 저장
    with open('results/models/bgvar_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장 완료: results/models/bgvar_model.pkl")
    
    # 요약 저장
    save_model_summary(model)
```

### 2-3. 분석 및 시각화 (03_analysis.py)

```python
"""
IRF 및 FEVD 분석 스크립트
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pyBGVAR import get_shockinfo, compute_fevd, plot

def compute_irf(model, shock_var='US.stir', n_ahead=24):
    """IRF 계산"""
    print(f"\nIRF 계산: {shock_var} 충격")
    
    # Cholesky 식별
    shockinfo = get_shockinfo(ident="chol", nr_rows=1)
    shockinfo.loc[0, 'shock'] = shock_var
    shockinfo.loc[0, 'scale'] = 1.0  # 1 표준편차
    
    irf_result = model.irf(n_ahead=n_ahead, shockinfo=shockinfo, verbose=True)
    
    return irf_result

def plot_irf_results(irf_result, shock_name='US.stir'):
    """IRF 플롯"""
    fig = plot.plot_irf(
        irf_result,
        resp=['US.y', 'EA.y', 'UK.y', 'JP.y'],
        shock=0,
        quantiles=[0.16, 0.5, 0.84]
    )
    
    plt.suptitle(f'Impulse Response to {shock_name} Shock', y=1.02)
    plt.tight_layout()
    plt.savefig(f'results/figures/irf_{shock_name.replace(".", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ IRF 플롯 저장: results/figures/irf_{shock_name.replace('.', '_')}.png")

def compute_and_plot_fevd(irf_result, var='US.y'):
    """FEVD 계산 및 플롯"""
    print(f"\nFEVD 계산: {var}")
    
    fevd_result = compute_fevd(irf_result, var_slct=[var])
    
    fig = plot.plot_fevd(fevd_result, resp=var, k_max=10)
    plt.tight_layout()
    plt.savefig(f'results/figures/fevd_{var.replace(".", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ FEVD 플롯 저장: results/figures/fevd_{var.replace('.', '_')}.png")
    
    return fevd_result

if __name__ == '__main__':
    # 모델 로드
    with open('results/models/bgvar_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("=" * 60)
    print("IRF 및 FEVD 분석")
    print("=" * 60)
    
    # 미국 금리 충격 IRF
    irf_us_stir = compute_irf(model, shock_var='US.stir', n_ahead=24)
    plot_irf_results(irf_us_stir, shock_name='US.stir')
    
    # 미국 GDP FEVD
    fevd_us_y = compute_and_plot_fevd(irf_us_stir, var='US.y')
    
    print("\n✅ 모든 분석 완료!")
```

---

## 3. 실제 데이터로 분석하기

### 3-1. Excel 데이터 읽기

```python
"""
실제 데이터로 작업하기
"""
import pandas as pd
from pyBGVAR import excel_to_list, matrix_to_list

# 방법 1: Excel 파일 직접 읽기 (시트별로 국가)
# Excel 형식: 각 시트가 국가명, 컬럼이 변수명
data_dict = excel_to_list('data/raw/economic_data.xlsx')

# 방법 2: CSV 읽기 후 변환
# CSV 형식: 컬럼명이 'COUNTRY.VARIABLE' 형태
data_matrix = pd.read_csv('data/raw/economic_data.csv', index_col=0)
data_dict = matrix_to_list(data_matrix)

print("국가 목록:", list(data_dict.keys()))
print("변수 목록:", list(data_dict[list(data_dict.keys())[0]].columns))
```

### 3-2. 데이터 전처리

```python
"""
데이터 전처리 및 검증
"""
import numpy as np
import pandas as pd

def check_missing_values(data_dict):
    """결측치 확인"""
    for country, df in data_dict.items():
        missing = df.isnull().sum()
        if missing.any():
            print(f"⚠️  {country}: 결측치 발견")
            print(missing[missing > 0])
    print("✅ 결측치 검사 완료")

def check_stationarity(data_dict):
    """정상성 대략 확인 (ADF 테스트)"""
    from scipy import stats
    
    for country, df in data_dict.items():
        for col in df.columns:
            # 간단한 추세 검사
            x = np.arange(len(df))
            slope, _, _, p_value, _ = stats.linregress(x, df[col])
            
            if p_value < 0.05 and abs(slope) > 0.01:
                print(f"⚠️  {country}.{col}: 강한 추세 존재 (차분 고려)")
    
    print("✅ 정상성 검사 완료")

def transform_data(data_dict, transformations):
    """
    데이터 변환 적용
    transformations: dict of dict
        예: {'US': {'y': 'log_diff', 'Dp': 'none', 'stir': 'none'}}
    """
    transformed = {}
    
    for country, df in data_dict.items():
        transformed[country] = df.copy()
        
        if country in transformations:
            for var, trans in transformations[country].items():
                if trans == 'log':
                    transformed[country][var] = np.log(df[var])
                elif trans == 'log_diff':
                    transformed[country][var] = np.log(df[var]).diff()
                elif trans == 'diff':
                    transformed[country][var] = df[var].diff()
        
        # 결측치 제거 (차분 등으로 생긴)
        transformed[country] = transformed[country].dropna()
    
    return transformed

# 실행 예제
if __name__ == '__main__':
    # 데이터 검증
    check_missing_values(data_dict)
    check_stationarity(data_dict)
    
    # 변환 정의
    transformations = {
        'US': {'y': 'log_diff', 'Dp': 'none', 'stir': 'none'},
        'EA': {'y': 'log_diff', 'Dp': 'none', 'stir': 'none'},
        # ... 다른 국가
    }
    
    # 변환 적용
    data_dict_transformed = transform_data(data_dict, transformations)
```

### 3-3. 실제 가중치 행렬 생성

```python
"""
실제 무역 데이터로 가중치 행렬 생성
"""
import pandas as pd
import numpy as np

def create_trade_weights(trade_matrix):
    """
    무역 데이터로 가중치 행렬 생성
    
    Parameters:
    -----------
    trade_matrix : DataFrame
        행: 수출국, 열: 수입국
        값: 무역량 (예: 백만 달러)
    
    Returns:
    --------
    W : DataFrame
        가중치 행렬 (행의 합 = 1, 대각선 = 0)
    """
    # 양방향 무역 (수출 + 수입)
    total_trade = trade_matrix + trade_matrix.T
    
    # 대각선 0
    np.fill_diagonal(total_trade.values, 0)
    
    # 정규화
    row_sums = total_trade.sum(axis=1)
    W = total_trade.div(row_sums, axis=0)
    
    return W

# 예제: 무역 데이터 읽기
trade_data = pd.read_csv('data/raw/bilateral_trade.csv', index_col=0)
W = create_trade_weights(trade_data)

print("가중치 행렬:")
print(W)
print("\n행의 합 (모두 1이어야 함):")
print(W.sum(axis=1))
```

---

## 4. 고급 기능 활용

### 4-1. 부호 제약 IRF

```python
"""
부호 제약을 사용한 충격 식별
예: 긴축 통화정책 충격
"""
from pyBGVAR import get_shockinfo, add_shockinfo

# 긴축 통화정책 충격:
# - 금리 상승 (+)
# - GDP 하락 (-)
# - 인플레이션 하락 (-)

shockinfo = get_shockinfo(ident="sign", nr_rows=1)
shockinfo = add_shockinfo(
    shockinfo,
    shock='US.stir',              # 충격 변수: 미국 금리
    restriction=['US.stir', 'US.y', 'US.Dp'],  # 제약할 변수들
    sign=['+', '-', '-'],         # 부호 제약
    horizon=4,                    # 4기까지 제약
    prob=0.65,                    # 65% 확률로 만족
    scale=1,                      # 크기
    scale_horizon=0               # 즉각 반응
)

# IRF 계산
irf_monetary = model.irf(n_ahead=24, shockinfo=shockinfo, verbose=True)

# 플롯
plot.plot_irf(irf_monetary, resp=['US.stir', 'US.y', 'US.Dp'], shock=0)
plt.suptitle('Contractionary Monetary Policy Shock (Sign Restrictions)')
plt.savefig('results/figures/irf_monetary_policy.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 4-2. 조건부 예측

```python
"""
조건부 예측: 특정 변수를 고정한 상태에서 예측
예: 금리 경로를 고정하고 GDP와 인플레이션 예측
"""
import numpy as np

def conditional_forecast(model, fixed_vars, fixed_paths, n_ahead=8):
    """
    조건부 예측
    
    Parameters:
    -----------
    model : BGVAR object
    fixed_vars : list
        고정할 변수명 리스트 (예: ['US.stir'])
    fixed_paths : dict
        변수별 고정 경로 (예: {'US.stir': [2.0, 2.5, 3.0, ...]})
    n_ahead : int
        예측 기간
    """
    # 제약 행렬 초기화
    K = model.xglobal.shape[1]
    constr = np.zeros((n_ahead, K))
    
    # 고정 변수 경로 설정
    for var, path in fixed_paths.items():
        var_idx = list(model.xglobal.columns).index(var)
        constr[:, var_idx] = path
    
    # 조건부 예측
    fcast_cond = model.predict(
        n_ahead=n_ahead,
        constr=constr,
        save_store=True,
        verbose=True
    )
    
    return fcast_cond

# 실행 예제
fixed_paths = {
    'US.stir': np.linspace(2.0, 4.0, 8)  # 금리 2%에서 4%로 증가
}

fcast_cond = conditional_forecast(
    model,
    fixed_vars=['US.stir'],
    fixed_paths=fixed_paths,
    n_ahead=8
)

# 플롯
plot.plot_pred(fcast_cond, resp=['US.y', 'US.Dp', 'US.stir'], cut=20)
plt.suptitle('Conditional Forecast (Fixed Interest Rate Path)')
plt.savefig('results/figures/conditional_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 4-3. 모델 비교

```python
"""
다양한 prior와 설정으로 모델 비교
"""
from pyBGVAR import BGVAR

def compare_models(data_dict, W, specs):
    """
    여러 모델 스펙 비교
    
    Parameters:
    -----------
    specs : list of dict
        각 모델 스펙 (예: [{'prior': 'MN', 'SV': False}, ...])
    """
    results = {}
    
    for i, spec in enumerate(specs):
        print(f"\n모델 {i+1}: {spec}")
        
        model = BGVAR(
            Data=data_dict,
            W=W,
            plag=spec.get('plag', 2),
            draws=spec.get('draws', 5000),
            burnin=spec.get('burnin', 5000),
            prior=spec['prior'],
            SV=spec.get('SV', True),
            verbose=False
        )
        
        # DIC 계산
        dic_result = model.dic()
        
        results[f"Model_{i+1}_{spec['prior']}"] = {
            'model': model,
            'DIC': dic_result['DIC'],
            'pD': dic_result['pD'],
            'spec': spec
        }
        
        print(f"  DIC: {dic_result['DIC']:.2f}")
        print(f"  pD: {dic_result['pD']:.2f}")
    
    return results

# 실행
specs = [
    {'prior': 'MN', 'SV': False},     # Minnesota, no SV
    {'prior': 'SSVS', 'SV': True},    # SSVS with SV
    {'prior': 'NG', 'SV': True},      # Normal-Gamma with SV
]

model_comparison = compare_models(data_dict, W, specs)

# 최적 모델 선택 (DIC 기준)
best_model_name = min(model_comparison, key=lambda k: model_comparison[k]['DIC'])
best_model = model_comparison[best_model_name]['model']

print(f"\n✅ 최적 모델: {best_model_name}")
```

---

## 5. 결과 해석 및 보고

### 5-1. IRF 해석 가이드

```python
"""
IRF 결과 해석 및 표 생성
"""
import pandas as pd

def extract_irf_table(irf_result, horizon=[0, 1, 4, 8, 12], 
                     responses=['US.y', 'EA.y'], shock_name='US.stir'):
    """
    IRF 결과를 표로 정리
    """
    irf_median = irf_result['posterior']['IRF.Median']
    irf_lower = irf_result['posterior']['IRF.LB']
    irf_upper = irf_result['posterior']['IRF.UB']
    
    # 충격 인덱스
    shock_idx = 0  # 첫 번째 충격
    
    # 표 생성
    table_data = []
    for resp in responses:
        resp_idx = list(irf_result['posterior']['variables']).index(resp)
        
        for h in horizon:
            median = irf_median[shock_idx, resp_idx, h]
            lower = irf_lower[shock_idx, resp_idx, h]
            upper = irf_upper[shock_idx, resp_idx, h]
            
            table_data.append({
                'Response': resp,
                'Horizon': h,
                'Median': f"{median:.4f}",
                '16% CI': f"{lower:.4f}",
                '84% CI': f"{upper:.4f}"
            })
    
    df = pd.DataFrame(table_data)
    return df

# 표 생성 및 저장
irf_table = extract_irf_table(irf_us_stir)
irf_table.to_csv('results/tables/irf_summary.csv', index=False)
irf_table.to_latex('results/tables/irf_summary.tex', index=False)

print("✅ IRF 표 저장 완료")
print(irf_table)
```

### 5-2. 논문용 그래프 생성

```python
"""
출판용 고품질 그래프
"""
import matplotlib.pyplot as plt
import seaborn as sns

# 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (12, 8)

def publication_quality_irf(irf_result, responses, shock_name, 
                           save_path='results/figures/irf_publication.pdf'):
    """
    출판용 IRF 플롯
    """
    n_resp = len(responses)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    irf_median = irf_result['posterior']['IRF.Median']
    irf_lower = irf_result['posterior']['IRF.LB']
    irf_upper = irf_result['posterior']['IRF.UB']
    
    horizons = np.arange(irf_median.shape[2])
    shock_idx = 0
    
    for i, resp in enumerate(responses):
        ax = axes[i]
        resp_idx = list(irf_result['posterior']['variables']).index(resp)
        
        median = irf_median[shock_idx, resp_idx, :]
        lower = irf_lower[shock_idx, resp_idx, :]
        upper = irf_upper[shock_idx, resp_idx, :]
        
        # 플롯
        ax.plot(horizons, median, 'b-', linewidth=2, label='Median')
        ax.fill_between(horizons, lower, upper, alpha=0.3, color='blue')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Horizon (Quarters)')
        ax.set_ylabel('Response')
        ax.set_title(f'{resp}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(f'Impulse Responses to {shock_name} Shock', 
                 fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 출판용 그래프 저장: {save_path}")

# 실행
publication_quality_irf(
    irf_us_stir, 
    responses=['US.y', 'US.Dp', 'EA.y', 'EA.Dp'],
    shock_name='US Interest Rate'
)
```

---

## 6. 성능 최적화

### 6-1. 대규모 모델 최적화

```python
"""
대규모 GVAR 모델 최적화 팁
"""

# 1. Thinning 사용 (메모리 절약)
model = BGVAR(
    Data=data_dict,
    W=W,
    plag=2,
    draws=20000,    # 많은 draws
    burnin=10000,
    thin=10,        # 10개 중 1개만 저장 -> 실제 2000개 저장
    prior="NG",
    SV=True
)

# 2. SV 비활성화 (속도 향상)
model_fast = BGVAR(
    Data=data_dict,
    W=W,
    plag=2,
    draws=5000,
    burnin=5000,
    prior="MN",     # Minnesota가 가장 빠름
    SV=False,       # SV 비활성화
)

# 3. 예측 시 메모리 절약
fcast = model.predict(
    n_ahead=8,
    save_store=False,  # 사후 분포 저장 안함
    verbose=False
)

# 4. IRF 계산 시 변수 선택
# 모든 변수 대신 관심 변수만
irf_result = compute_fevd(
    irf_result,
    var_slct=['US.y', 'US.Dp', 'US.stir']  # 일부 변수만
)
```

### 6-2. 병렬 처리 (향후 지원 예정)

```python
"""
병렬 처리 예시 (향후 버전에서 지원 예정)
"""
# # 여러 모델을 병렬로 추정
# from joblib import Parallel, delayed
# 
# def estimate_single_model(spec):
#     return BGVAR(Data=data_dict, W=W, **spec)
# 
# specs = [
#     {'prior': 'MN', 'draws': 5000, 'burnin': 5000},
#     {'prior': 'SSVS', 'draws': 5000, 'burnin': 5000},
#     {'prior': 'NG', 'draws': 5000, 'burnin': 5000},
# ]
# 
# models = Parallel(n_jobs=3)(delayed(estimate_single_model)(spec) for spec in specs)
```

---

## 부록: 유용한 유틸리티 함수

### A. 결과 요약 리포트 생성

```python
"""
자동 결과 리포트 생성
"""
import datetime

def generate_report(model, irf_results, fevd_results, output_file='results/report.txt'):
    """
    분석 결과 종합 리포트 생성
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BGVAR 분석 결과 리포트\n")
        f.write("=" * 80 + "\n")
        f.write(f"생성 일시: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 모델 정보
        f.write("1. 모델 정보\n")
        f.write("-" * 80 + "\n")
        f.write(f"   - 국가 수: {model.N}\n")
        f.write(f"   - 변수 수: {model.xglobal.shape[1]}\n")
        f.write(f"   - 시계열 길이: {model.xglobal.shape[0]}\n")
        f.write(f"   - Prior: {model.prior}\n")
        f.write(f"   - MCMC 추출 수: {model.args.get('thindraws', 0)}\n\n")
        
        # DIC
        dic_result = model.dic()
        f.write("2. 모델 선택 기준\n")
        f.write("-" * 80 + "\n")
        f.write(f"   - DIC: {dic_result['DIC']:.2f}\n")
        f.write(f"   - pD: {dic_result['pD']:.2f}\n\n")
        
        # 수렴 진단
        from pyBGVAR import conv_diag
        conv_result = conv_diag(model)
        f.write("3. 수렴 진단\n")
        f.write("-" * 80 + "\n")
        f.write(f"   - Geweke 검정 통과율: {conv_result['perc']}\n\n")
        
        f.write("4. 생성된 그래프\n")
        f.write("-" * 80 + "\n")
        f.write("   - results/figures/irf_*.png\n")
        f.write("   - results/figures/fevd_*.png\n")
        f.write("   - results/figures/forecast_*.png\n\n")
        
    print(f"✅ 리포트 생성 완료: {output_file}")

# 실행
generate_report(model, irf_us_stir, fevd_us_y)
```

---

## 다음 단계

✅ 튜토리얼 완료 후:
1. **자신의 데이터로 적용**
2. **다양한 식별 전략 시도** (Cholesky vs Sign restrictions)
3. **모델 스펙 비교** (다양한 prior, SV on/off)
4. **결과를 논문/보고서에 활용**

## 추가 자료

- **[QUICKSTART.md](QUICKSTART.md)**: 빠른 참조
- **[GITHUB_INSTALLATION_GUIDE.md](GITHUB_INSTALLATION_GUIDE.md)**: 설치 문제 해결
- **[원본 R 패키지 논문](https://www.jstatsoft.org/article/view/v104i09)**: 이론 및 방법론

---

**즐거운 분석 되세요!** 🎯

