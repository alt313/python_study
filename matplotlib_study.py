# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:26:58 2020

@author: msi
"""

# Matplotlib
# =============================================================================

# Matplotlib 특징

# 파이썬의 대표적인 과학 계산용 그래프 라이브러리
# 선 그래프, 히스토그램, 산점도 등의 고품질 그래프 제공
# 저수준 api를 사용한 다양한 시각화 기능 제공
# 다양한 운영체제와 그래픽 백엔드에서 동작

# matplotlib, numpy, pandas호출
print('matplotlib, numpy, pandas호출')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matplotlib 현재 버전 출력
print('matplotlib 현재 버전 출력')
print(mpl.__version__)
print()

plt.style.use(['seaborn-notebook'])

# 라인 플롯

# plt.figure : 축과 그래프 , 텍스트 레이블을 표시하는 모든 객체를 포함하는 컨테이너
# plt.axes : 눈금과 레이블이 있는 테두리 박스로 시각화를 형성하는 플롯 요소 포함

# 하얀 도화지 출력
print('하얀 도화지 출력')
fig = plt.figure()
ax = plt.axes()
plt.show()
# plt.show() : 파이참에서 시각화를 하기위한 명령어
print()

# 지그재그 직선 출력
print('지그재그 직선 출력')
fig = plt.figure()
plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1] * 5)
plt.show()
print()

# sin곡선 출력
print('sin 곡선 출력')
x = np.arange(0, 10, 0.01)
fig = plt.figure()
plt.plot(x, np.sin(x))
plt.show() 
print()

# sin, cos 곡선 출력
print('sin, cos 곡선 출력')
plt.plot(x, np.sin(x));
plt.plot(x, np.cos(x));
plt.show()
print()

# 랜덤한 50개의 숫자 누적 합
print('랜덤한 50개의 숫자 누적 합')
plt.plot(np.random.randn(50).cumsum());
plt.show()
print()

# 라인 스타일

# '-' : 'solid'
print("'-' : 'solid'")
plt.plot(np.random.randn(50).cumsum(), linestyle = 'solid')
plt.show()
print()

# '--' : 'dashed'
print("'--' : 'dashed'")
plt.plot(np.random.randn(50).cumsum(), linestyle = 'dashed')
plt.show()
print()

# '-.' : 'dashdot'
print("'-.' : 'dashdot'")
plt.plot(np.random.randn(50).cumsum(), linestyle = 'dashdot')
plt.show()
print()

# ':' : 'dotted'
print("':' : 'dotted'")
plt.plot(np.random.randn(50).cumsum(), linestyle = 'dotted')
plt.show()
print()

# 모두 합치기
print('모두 합치기')
plt.plot(np.random.randn(50).cumsum(), linestyle = '-')
plt.plot(np.random.randn(50).cumsum(), linestyle = '--')
plt.plot(np.random.randn(50).cumsum(), linestyle = '-.')
plt.plot(np.random.randn(50).cumsum(), linestyle = ':')
plt.show()

# 색상 스타일

# 색상 지정 출력
print('색상 지정 출력1')
plt.plot(np.random.randn(50).cumsum(), color = 'g')
plt.plot(np.random.randn(50).cumsum(), color = '#1243FF')
plt.plot(np.random.randn(50).cumsum(), color = (0.2, 0.4, 0.6))
plt.plot(np.random.randn(50).cumsum(), color = 'darkblue')
plt.show()
print()

print('색상 지정 출력2') 
plt.plot(np.random.randn(50).cumsum(), color = 'skyblue')
plt.plot(np.random.randn(50).cumsum(), color = 'dodgerblue')
plt.plot(np.random.randn(50).cumsum(), color = 'royalblue')
plt.plot(np.random.randn(50).cumsum(), color = 'navy')
plt.show()
print()

# 라인스타일과 색상 함께 지정 출력
print('라인스타일과 색상 함께 지정 출력')
plt.plot(np.random.randn(50).cumsum(), 'b-')
plt.plot(np.random.randn(50).cumsum(), 'g--')
plt.plot(np.random.randn(50).cumsum(), 'c-.')
plt.plot(np.random.randn(50).cumsum(), 'm:')
plt.show()
print()

# 플롯 축

# =============================================================================
