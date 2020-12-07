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

# 하얀 도화지(플롯) 출력
print('하얀 도화지(플롯) 출력')
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
print()

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

# 랜덤한 50개 출력
print('랜덤한 50개 출력')
plt.plot(np.random.randn(50))
plt.show()
print()

# x,y축 범위 지정
print('x,y축 범위 지정1')
plt.plot(np.random.randn(50))
plt.xlim(-1, 50)
plt.ylim(-5, 5)
plt.show()
print()

print('x,y축 범위 지정2')
plt.plot(np.random.randn(50))
plt.axis([-1, 50, -5, 5])
plt.show()
print()

# x,y축 범위에 딱 맞게 출력
print('x,y축 범위에 딱 맞게 출력')
plt.plot(np.random.randn(50))
plt.axis('tight')
plt.show()
print()
# x,y축 범위 널널하게 출력
print('x,y축 범위 널널하게 출력')
plt.plot(np.random.randn(50))
plt.axis('equal')
plt.show()
print()

# 플롯 레이블

# 시각화 된 표에 이름 정해서 출력
print('시각화 된 표에 이름 정해서 출력')
plt.plot(np.random.randn(50))
plt.title('title')
plt.xlabel('x')
plt.ylabel('random.randn')
plt.show()
print()

# 각 그래프에 이름 정해주고 범례 출력
print('각 그래프에 이름 정해주고 범례 출력')
plt.plot(np.random.randn(50), label = 'A')
plt.plot(np.random.randn(50), label = 'B')
plt.plot(np.random.randn(50), label = 'C')
plt.title('title')
plt.xlabel('x')
plt.ylabel('random.randn')
plt.legend()
plt.show()
print()

# 폰트 관리자

# 지정해서 쓸수있는 폰트
print('지정해서 쓸수있는 폰트')
print(set(f.name for f in mpl.font_manager.fontManager.ttflist))
print()

font1 = {'family' : 'Dejavu Sans', 'size' : 24, 'color': 'black'}
font2 = {'family' : 'Jokerman', 'size' : 18, 'weight' : 'bold', 'color': 'darkred'}
font3 = {'family' : 'STIXGeneral', 'size' : 16, 'weight' : 'light', 'color': 'black'}

# 각 그래프 이름에 폰트와 크기 컬러 굵기 지정
print('각 그래프 이름에 폰트와 크기 컬러 굵기 지정')
plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
plt.title('title', fontdict = font1)
plt.xlabel('x', fontdict = font2)
plt.ylabel('random.randn', fontdict = font3)
plt.show()
print()

# 플롯 범례

# 범례 기본값(best)
print('범례 기본값(best)')
fig, ax = plt.subplots()
ax.plot(np.random.randn(10), '-r', label = 'A')
ax.plot(np.random.randn(10), ':g', label = 'B')
ax.plot(np.random.randn(10), '--b', label = 'C')
ax.axis('equal')
ax.legend()
fig.show()
plt.show()
print()

# 범례 lower right
print('범례 lower right')
fig, ax = plt.subplots()
ax.plot(np.random.randn(10), '-r', label = 'A')
ax.plot(np.random.randn(10), ':g', label = 'B')
ax.plot(np.random.randn(10), '--b', label = 'C')
ax.legend(loc = 'lower right')
fig.show()
plt.show()
print()

# 범례 upper center
print('범례 upper center')
fig, ax = plt.subplots()
ax.plot(np.random.randn(10), '-r', label = 'A')
ax.plot(np.random.randn(10), ':g', label = 'B')
ax.plot(np.random.randn(10), '--b', label = 'C')
ax.legend(loc = 'upper center', frameon = False, ncol = 2)
fig.show()
plt.show()
print()

# 범례에 여려가지 효과
print('범례에 여려가지 효과')
fig, ax = plt.subplots()
ax.plot(np.random.randn(10), '-r', label = 'A')
ax.plot(np.random.randn(10), ':g', label = 'B')
ax.plot(np.random.randn(10), '--b', label = 'C')
ax.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1)
fig.show()
plt.show()
print()

# 범례 3개만 표현1
print('범례 3개만 표현1')
plt.figure(figsize = (8, 4))
x = np.linspace(0, 10, 1000)
y = np.cos(x[:, np.newaxis] * np.arange(0, 2, 0.2))
lines = plt.plot(x, y)
plt.legend(lines[:3], ['c1', 'c2', 'c3'])
plt.show()
print()

# 범례 3개만 표현2
print('범례 3개만 표현2')
plt.figure(figsize = (8, 4))
x = np.linspace(0, 10, 1000)
y = np.cos(x[:, np.newaxis] * np.arange(0, 2, 0.2))
plt.plot(x, y[:, 0], label = 'c1')
plt.plot(x, y[:, 1], label = 'c2')
plt.plot(x, y[:, 2], label = 'c3')
plt.plot(x, y[:, 3:])
plt.legend(framealpha = 1, frameon = True)
plt.show()
print()

# 컬러바 출력
print('컬러바 출력')
x = np.linspace(0, 20, 100)
I = np.cos(x) - np.cos(x[:, np.newaxis])
plt.imshow(I)
plt.colorbar()
plt.show()
print()

# 컬러바 색상 지정해서 출력(파란색)
print('컬러바 색상 지정해서 출력(파란색)')
x = np.linspace(0, 20, 100)
I = np.cos(x) - np.cos(x[:, np.newaxis])
plt.imshow(I, cmap = 'Blues')
plt.colorbar()
plt.show()
print()

# 컬러바 색상 지정해서 출력(파란색 + 빨간색)
print('컬러바 색상 지정해서 출력(파란색 + 빨간색)')
x = np.linspace(0, 20, 100)
I = np.cos(x) - np.cos(x[:, np.newaxis])
plt.imshow(I, cmap = 'RdBu')
plt.colorbar()
plt.show()
print()

# 컬러바 범위와 컬러바 생김새 변화해서 출력
print('컬러바 범위와 컬러바 생김새 변화해서 출력')
x = np.linspace(0, 20, 100)
I = np.cos(x) - np.cos(x[:, np.newaxis])
speckles = (np.random.random(I.shape) < 0.01)
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))
plt.imshow(I, cmap = 'RdBu')
plt.colorbar(extend = 'both')
plt.clim(-1, 1)
plt.show()
print()

# 컬러바 범위를 5개로 지정해서 출력
print('컬러바 범위를 5개로 지정해서 출력')
x = np.linspace(0, 20, 100)
I = np.cos(x) - np.cos(x[:, np.newaxis])
plt.imshow(I, cmap = plt.cm.get_cmap('Blues', 5))
plt.colorbar()
plt.clim(-1, 1)
plt.show()
print()

# 다중 플롯

# 플롯 안에 또 다른 플롯 출력
print('플롯 안에 또 다른 플롯 출력')
ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.5, 0.2, 0.3])
plt.show()
print()

# 반복문으로 여러개 플롯 출력
print('반복문으로 여러개 플롯 출력')
for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.text(0.5, 0.5, str((3, 3, i)), ha = 'center')
plt.show()
print()

# 플롯끼리 겹쳐서 출력된것 간격줘서 출력
print('플롯끼리 겹쳐서 출력된것 간격줘서 출력')
fig = plt.figure()
fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.text(0.5, 0.5, str((3, 3, i)), ha = 'center')
plt.show()
print()

# 반복문 쓰지않고 여러개 한번에 출력
print('반복문 쓰지않고 여러개 한번에 출력')
fig, ax = plt.subplots(3, 3, sharex = 'col', sharey = 'row')
plt.show()
print()

# 위 플롯에 위치값 넣어주기
print('위 플롯에 위치값 넣어주기')
fig, ax = plt.subplots(3, 3, sharex = 'col', sharey = 'row')
for i in range(3):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)), ha = 'center')
plt.show()
print()

# 위치값을 주어 플롯 크기 다르게 하기
print('위치값을 주어 플롯 크기 다르게 하기')
grid = plt.GridSpec(2, 3, wspace = 0.4, hspace = 0.4)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])
plt.show()
print()

# 반복문을 통한 서브플롯
print('반복문을 통한 서브플롯')
plt.figure(figsize = (5, 6))
x = range(1, 21)
columns = [np.random.randn(20) * i for i in range(1, 7)]
i = 0
for c in columns:
    i += 1
    plt.subplot(3, 2, i)
    plt.plot(x, c, marker = 'o', linewidth = 1, label = c)
    plt.xlim(-1, 21)
    plt.ylim(c.min() - 1, c.max() + 1)
plt.show()
print()

# 텍스트와 주석

# 텍스트로 플롯에 위치치정해서 출력1
print('텍스트로 플롯에 위치치정해서 출력1')
fig, ax = plt.subplots()
ax.axis([0, 10, 0, 10])
ax.text(3, 6, '. transData(3, 6)', transform = ax.transData) # 데이터에 실 위치
ax.text(0.2, 0.4, '. transAxes(0.2, 0.4)', transform = ax.transAxes) # 축에의한 위치
ax.text(0.2, 0.2, '. transFigure(0.2, 0.2)', transform = fig.transFigure) # 실체 플롯의 위치
plt.show()
print()

# 텍스트로 플롯에 위치치정해서 출력2
print('텍스트로 플롯에 위치치정해서 출력2')
fig, ax = plt.subplots()
ax.axis([0, 10, 0, 10])
ax.text(3, 6, '. transData(3, 6)', transform = ax.transData) # 데이터에 실 위치
ax.text(0.2, 0.4, '. transAxes(0.2, 0.4)', transform = ax.transAxes) # 축에의한 위치
ax.text(0.2, 0.2, '. transFigure(0.2, 0.2)', transform = fig.transFigure) # 실체 플롯의 위치
ax.set_xlim(-6, 10)
ax.set_ylim(-6, 10)
plt.show()
print()

# 플롯에 화살표로 가르키는 그림 출력1
print('플롯에 화살표로 가르키는 그림 출력1')
x = np.arange(1, 40)
y = x * 1.1
plt.scatter(x, y, marker = '.')
plt.axis('equal')
plt.annotate('interesting point', xy = (4, 5), xytext = (20, 10),
             arrowprops = dict(shrink = 0.05))
print()

# 플롯에 화살표로 가르키는 그림 출력2
print('플롯에 화살표로 가르키는 그림 출력2')
x1 = np.random.normal(30, 3, 100)
x2 = np.random.normal(20, 3, 100)
x3 = np.random.normal(10, 3, 100)
plt.plot(x1, label = 'p1')
plt.plot(x2, label = 'p2')
plt.plot(x3, label = 'p3')
plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 0, ncol = 3,
           mode= 'extend', borderaxespad = 0.)
plt.annotate('important value', (50, 20), xytext = (5, 40),
             arrowprops = dict(arrowstyle = '->'))
plt.annotate('incorrect value', (40, 30), xytext = (50, 40),
             arrowprops = dict(arrowstyle = '->'))
plt.show()
print()

# =============================================================================


