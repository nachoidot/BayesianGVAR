"""
pyBGVAR 사용 예제
================

이 스크립트는 pyBGVAR 패키지의 주요 기능 사용법을 보여줍니다.
"""

import numpy as np
import pandas as pd
from pyBGVAR import BGVAR, get_shockinfo, add_shockinfo

# =============================================================================
# 예제 1: 데이터 준비
# =============================================================================

print("=" * 80)
print("예제 1: 데이터 준비")
print("=" * 80)

# 방법 1: Dictionary 형태의 데이터
# 각 국가가 key, DataFrame이 value
np.random.seed(42)

# 예제 데이터 생성 (실제로는 실제 데이터를 사용)
T = 100  # 시계열 길이
countries = ['US', 'EA', 'UK']

# 각 국가별 데이터 생성
data_dict = {}
for country in countries:
    data_dict[country] = pd.DataFrame({
        'y': np.random.randn(T).cumsum(),      # GDP
        'Dp': np.random.randn(T) * 0.5,        # 인플레이션
        'stir': np.random.randn(T) * 0.3 + 2   # 단기금리
    })

print(f"국가 수: {len(data_dict)}")
print(f"각 국가별 변수 수: {data_dict['US'].shape[1]}")
print(f"시계열 길이: {T}")
print("\n미국 데이터 샘플:")
print(data_dict['US'].head())

# 방법 2: 하나의 DataFrame (컬럼명: 'COUNTRY.VARIABLE')
# data_matrix = pd.DataFrame({
#     'US.y': ...,
#     'US.Dp': ...,
#     'EA.y': ...,
#     ...
# })

# =============================================================================
# 예제 2: 가중치 행렬 생성
# =============================================================================

print("\n" + "=" * 80)
print("예제 2: 가중치 행렬 (W) 준비")
print("=" * 80)

# 가중치 행렬 W: 국가 간 경제적 연결성을 나타냄
# 예: 무역 가중치, GDP 가중치 등
W = pd.DataFrame(
    np.array([
        [0.0, 0.6, 0.4],  # US: EA에 60%, UK에 40% 가중치
        [0.5, 0.0, 0.5],  # EA: US에 50%, UK에 50%
        [0.4, 0.6, 0.0]   # UK: US에 40%, EA에 60%
    ]),
    index=countries,
    columns=countries
)

print("가중치 행렬 W:")
print(W)
print("\n각 행의 합 (자기 자신 제외, 1이 되어야 함):")
print(W.sum(axis=1))

# =============================================================================
# 예제 3: BGVAR 모델 추정
# =============================================================================

print("\n" + "=" * 80)
print("예제 3: BGVAR 모델 추정")
print("=" * 80)

# 모델 추정 (작은 draws/burnin으로 빠른 테스트)
model = BGVAR(
    Data=data_dict,
    W=W,
    plag=1,              # 내생 변수 시차
    draws=100,           # MCMC 추출 수 (실제로는 5000+ 권장)
    burnin=100,          # Burn-in 기간 (실제로는 5000+ 권장)
    prior="NG",          # Normal-Gamma prior
    SV=True,             # Stochastic Volatility
    hold_out=0,          # Hold-out 샘플 (예측 평가용)
    eigen=True,          # 안정성 검사
    verbose=True
)

print("\n모델 추정 완료!")
print(f"국가 수: {model.N}")
print(f"전역 변수 수: {model.xglobal.shape[1]}")
print(f"안정적인 사후 추출 수: {model.args.get('thindraws', 0)}")

# =============================================================================
# 예제 4: 모델 요약 및 진단
# =============================================================================

print("\n" + "=" * 80)
print("예제 4: 모델 요약")
print("=" * 80)

# 요약 통계 (수렴 진단, 잔차 검정 포함)
summary = model.summary()

# 계수 추출 (중앙값)
coefs = model.coef(quantile=0.50)
print(f"\n계수 행렬 크기: {coefs.shape}")

# 분산-공분산 행렬
vcov_matrix = model.vcov(quantile=0.50)
print(f"분산-공분산 행렬 크기: {vcov_matrix.shape}")

# 적합값
fitted_values = model.fitted(global_model=True)
print(f"\n적합값 크기: {fitted_values.shape}")

# 잔차
residuals_dict = model.residuals()
print(f"잔차 국가 모델 크기: {residuals_dict['country'].shape}")
print(f"잔차 글로벌 모델 크기: {residuals_dict['global'].shape}")

# 모델 선택 기준
dic_result = model.dic()
print(f"\nDIC: {dic_result['DIC']:.2f}")
print(f"효과적 모수 수 (pD): {dic_result['pD']:.2f}")

loglik = model.logLik(quantile=0.50)
print(f"로그 가능도: {loglik:.2f}")

# =============================================================================
# 예제 5: Impulse Response Functions (IRF)
# =============================================================================

print("\n" + "=" * 80)
print("예제 5: Impulse Response Functions (IRF)")
print("=" * 80)

# 5-1. Cholesky 분해를 사용한 IRF
print("\n[5-1] Cholesky IRF")
shockinfo_chol = get_shockinfo(ident="chol", nr_rows=1)
shockinfo_chol.loc[0, 'shock'] = 'US.y'     # 미국 GDP 충격
shockinfo_chol.loc[0, 'scale'] = 1.0        # 1 표준편차 충격

irf_chol = model.irf(
    n_ahead=24,              # 24기 앞까지
    shockinfo=shockinfo_chol,
    verbose=True
)

print(f"IRF 결과 키: {list(irf_chol.keys())}")
print(f"IRF 크기: {irf_chol['posterior']['IRF.Median'].shape}")

# 5-2. 일반화 IRF (GIRF)
print("\n[5-2] Generalized IRF")
shockinfo_girf = get_shockinfo(ident="girf", nr_rows=1)
shockinfo_girf.loc[0, 'shock'] = 'EA.Dp'    # 유로존 인플레이션 충격
shockinfo_girf.loc[0, 'scale'] = 1.0

irf_girf = model.irf(
    n_ahead=24,
    shockinfo=shockinfo_girf,
    verbose=False
)

# 5-3. 부호 제약을 사용한 IRF
print("\n[5-3] 부호 제약 IRF")
shockinfo_sign = get_shockinfo(ident="sign", nr_rows=1)
shockinfo_sign = add_shockinfo(
    shockinfo_sign,
    shock='US.stir',                    # 미국 금리 충격
    restriction=['US.y', 'US.Dp'],      # 제약할 변수
    sign=['-', '+'],                    # 음(-), 양(+) 제약
    horizon=5,                          # 5기까지 제약
    prob=0.5                            # 50% 확률로 만족
)

irf_sign = model.irf(
    n_ahead=24,
    shockinfo=shockinfo_sign,
    verbose=False
)

# =============================================================================
# 예제 6: Forecast Error Variance Decomposition (FEVD)
# =============================================================================

print("\n" + "=" * 80)
print("예제 6: Forecast Error Variance Decomposition")
print("=" * 80)

from pyBGVAR import compute_fevd, gfevd

# 6-1. 일반 FEVD (IRF 결과 필요)
print("\n[6-1] 일반 FEVD")
fevd_result = compute_fevd(
    irf_chol,
    var_slct=['US.y', 'EA.y'],  # 분석할 변수
    verbose=True
)

print(f"FEVD 결과 크기: {fevd_result['FEVD'].shape}")

# 6-2. 일반화 FEVD (Lanne-Nyberg 2016)
print("\n[6-2] Generalized FEVD")
gfevd_result = gfevd(
    model,
    n_ahead=24,
    running=True,       # Running mean (메모리 효율적)
    verbose=True
)

print(f"GFEVD 결과 크기: {gfevd_result['FEVD'].shape}")

# =============================================================================
# 예제 7: Historical Decomposition
# =============================================================================

print("\n" + "=" * 80)
print("예제 7: Historical Decomposition")
print("=" * 80)

from pyBGVAR import hd as compute_hd

hd_result = compute_hd(
    irf_chol,
    var_slct=['US.y'],  # 미국 GDP 분해
    verbose=True
)

print(f"HD 결과 키: {list(hd_result.keys())}")

# =============================================================================
# 예제 8: 예측 (Prediction)
# =============================================================================

print("\n" + "=" * 80)
print("예제 8: 예측")
print("=" * 80)

# 8-1. 무조건부 예측
print("\n[8-1] 무조건부 예측")
fcast = model.predict(
    n_ahead=8,           # 8기 앞 예측
    save_store=True,     # 사후 분포 저장
    verbose=True
)

print(f"예측 결과 키: {list(fcast.keys())}")
print(f"예측값 크기: {fcast['fcast'].shape}")

# 8-2. 조건부 예측 (특정 변수 경로 고정)
print("\n[8-2] 조건부 예측")
# 예: 미국 금리를 특정 경로로 고정
constr = np.zeros((8, model.xglobal.shape[1]))  # (horizon, K)
us_stir_idx = list(model.xglobal.columns).index('US.stir')
constr[:, us_stir_idx] = np.linspace(2.0, 3.0, 8)  # 2%에서 3%로 증가

fcast_cond = model.predict(
    n_ahead=8,
    constr=constr,
    save_store=True,
    verbose=False
)

# =============================================================================
# 예제 9: 예측 평가 (Hold-out 샘플 필요)
# =============================================================================

print("\n" + "=" * 80)
print("예제 9: 예측 평가 (Hold-out 샘플 사용)")
print("=" * 80)

# Hold-out 샘플이 있는 모델 재추정
model_eval = BGVAR(
    Data=data_dict,
    W=W,
    plag=1,
    draws=100,
    burnin=100,
    prior="NG",
    SV=True,
    hold_out=8,      # 마지막 8개 관측치를 hold-out
    verbose=False
)

# 예측
fcast_eval = model_eval.predict(n_ahead=8, save_store=True, verbose=False)

# 평가 지표
from pyBGVAR import lps, rmse

try:
    lps_scores = lps(fcast_eval)
    print(f"\nLPS 점수 크기: {lps_scores.shape}")
    print(f"평균 LPS: {np.mean(lps_scores):.4f}")
except Exception as e:
    print(f"\nLPS 계산 중 오류: {e}")

try:
    rmse_scores = rmse(fcast_eval)
    print(f"\nRMSE 크기: {rmse_scores.shape}")
    print(f"평균 RMSE: {np.mean(rmse_scores):.4f}")
except Exception as e:
    print(f"\nRMSE 계산 중 오류: {e}")

# =============================================================================
# 예제 10: 진단 (Diagnostics)
# =============================================================================

print("\n" + "=" * 80)
print("예제 10: 추가 진단")
print("=" * 80)

from pyBGVAR import conv_diag, resid_corr_test, avg_pair_cc

# 수렴 진단
print("\n[10-1] 수렴 진단")
conv_result = conv_diag(model, crit_val=1.96)
print(f"Geweke 검정: {conv_result['perc']}")

# 잔차 자기상관 검정
print("\n[10-2] 잔차 자기상관 검정")
resid_test = resid_corr_test(model, lag_cor=1, alpha=0.95)
print(f"F-통계량 국가: {list(resid_test['Fstat'].keys())}")

# 평균 쌍별 상관계수
print("\n[10-3] 평균 쌍별 교차 상관계수")
avg_corr = avg_pair_cc(model, digits=3)
print(f"데이터 상관: {avg_corr['data.cor']}")
print(f"잔차 상관: {avg_corr['resid.cor']}")

# =============================================================================
# 예제 11: 시각화 (Plotting)
# =============================================================================

print("\n" + "=" * 80)
print("예제 11: 시각화")
print("=" * 80)

from pyBGVAR import plot
import matplotlib.pyplot as plt

# 11-1. IRF 플롯
print("\n[11-1] IRF 플롯 생성 중...")
try:
    plot.plot_irf(
        irf_chol,
        resp=['US.y', 'EA.y'],
        shock=0,
        quantiles=[0.16, 0.5, 0.84]
    )
    plt.savefig('irf_plot.png', dpi=300, bbox_inches='tight')
    print("IRF 플롯 저장: irf_plot.png")
    plt.close()
except Exception as e:
    print(f"IRF 플롯 오류: {e}")

# 11-2. FEVD 플롯
print("\n[11-2] FEVD 플롯 생성 중...")
try:
    plot.plot_fevd(fevd_result, resp='US.y', k_max=5)
    plt.savefig('fevd_plot.png', dpi=300, bbox_inches='tight')
    print("FEVD 플롯 저장: fevd_plot.png")
    plt.close()
except Exception as e:
    print(f"FEVD 플롯 오류: {e}")

# 11-3. 예측 플롯
print("\n[11-3] 예측 플롯 생성 중...")
try:
    plot.plot_pred(fcast, resp=['US.y', 'EA.y'], cut=20)
    plt.savefig('forecast_plot.png', dpi=300, bbox_inches='tight')
    print("예측 플롯 저장: forecast_plot.png")
    plt.close()
except Exception as e:
    print(f"예측 플롯 오류: {e}")

# =============================================================================
# 마무리
# =============================================================================

print("\n" + "=" * 80)
print("모든 예제 완료!")
print("=" * 80)
print("\n주요 포인트:")
print("1. 실제 사용 시 draws=5000, burnin=5000 이상 권장")
print("2. 가중치 행렬 W는 실제 경제 연결성을 반영해야 함")
print("3. 각 함수의 verbose=True로 진행 상황 확인 가능")
print("4. 결과는 dict 형태로 반환되며, 다양한 통계량 포함")
print("5. 시각화 함수로 결과를 쉽게 확인 가능")
print("\n자세한 내용은 README.md와 각 함수의 docstring을 참고하세요.")

