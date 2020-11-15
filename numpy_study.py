# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:55:48 2020

@author: msi
"""
# 주석 단축키

# ctrl + 1

# =============================================================================
# ctrl + 4
# =============================================================================


# numpy 호출 및 버전확인
# =============================================================================
import numpy as np
print('numpy 현재 버전 출력')
print(np.__version__)
# numpy 현재 버전 출력
print()
# =============================================================================

# 배열 생성
# =============================================================================
a1 = np.array([1, 2, 3, 4, 5])

print('배열 출력')
print(a1)
print()
# 배열 출력

print('타입 출력')
print(type(a1))
print()
# 타입 출력

print('배열의 차원출력')
print(a1.shape)
# 배열의 모양을 출력해주는 명령어
print()

print('배열의 색인을 통한 출력')
print(a1[0], a1[1], a1[2], a1[3], a1[4])
# 배열에 색인을 통해서 출력
print()

print('색인을 통해 값 수정')
a1[0] = 4
a1[1] = 5
a1[2] = 6
print(a1)
# 색인을 통해 값 수정
print()

a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 2차원 배열 생성

print('2차원 배열 출력')
print(a2)
print()

print('배열의 형태 출력')
print(a2.shape)
print()

print('2차원 배열 색인 출력')
print(a2[0, 0], a2[1, 1], a2[2, 2])
# 2차원 배열 색인 방법

a3 = np.array([ [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ],
                [ [10, 11, 12], [13, 14, 15], [16, 17, 18] ],
                [ [19, 20, 21], [22, 23, 24], [25, 26, 27] ] ] )
# 3차원 배열 생성

print('3차원 배열 출력')
print(a3)
print()

print('배열의 형태 출력')
print(a3.shape)
print()

# 배열 생성 및 초기화

# zeros() : 모든 요소를 0으로 초기화
print('zeros() : 모든 요소를 0으로 초기화')
print(np.zeros(10))
print()

# ones() : 모든 요소를 1로 초기화
print('ones() : 모든 요소를 1로 초기화')
print(np.ones((3, 3))) # 생성시 배열의 형태 입력가능
print()

# full() : 모든 요소를 지정한 값으로 초기화
print('full() : 모든 요소를 지정한 값으로 초기화')
print(np.full((3, 3), 7))
print()

# eye() : 단위행렬 생성(정사각 행렬)
#         주대각선 원소가 1이고 나머지는 0
print('eye() : 단위행렬 생성(정사각 행렬)')
print(np.eye(5))
print()

# tri() : 삼각행렬 생성(정사각 행렬)
print('tri() : 삼각행렬 생성(정사각 행렬)')
print(np.tri(5))
print()

# empty() : 초기화되지 않은 배열 생성
#           초기화가 없어서 배열생성비용이 저렴하고 빠름
#           초기화되지 않아 기존 메모리 위치에 존재하는 값이 있다.
print('empty() : 초기화되지 않은 배열 생성')
print(np.empty(5))
print()

# _like() : 지정된 배열과 shape이 같은 행렬 생성
#   np.zeros_like()
#   np.ones_like()
#   np.full_like()
#   np.empty_like()
print('_like() : 지정된 배열과 shape가 같은 행렬 생성')
print('np.zeros_like() : 배열 a1')
print(a1)
print(np.zeros_like(a1))
print()

print('np.ones_like()) : 배열 a2')
print(a2)
print(np.ones_like(a2))
print()

print('np.full_like() : 배열 a3')
print(a3)
print(np.full_like(a3, 10))
print()

# 생성한 값으로 배열 생성

# arange() : 정수 범위로 배열 생성
print('arange() : 정수 범위로 배열 생성')
print(np.arange(0, 30, 1))
print()

# linspace() : 범위 내에서 균등 간격의 배열 생성
print('linspace() : 범위 내에서 균등 간격의 배열 생성')
print(np.linspace(0,10, 5))
print()

# logspace() : 범위 내에서 균등 간격으로 로그 스케일로 배열 생성
print('logspace() : 범위 내에서 균등 간격으로 로그 스케일로 배열 생성')
print(np.logspace(0.1, 1, 10))
print()

# 랜덤값으로 배열 생성

# random.random() : 랜덤한 수의 배열 생성
print('random.random() : 랜덤한 수의 배열 생성')
print(np.random.random((3, 3)))
print()

# random.randint() : 일정 구간의 랜덤 정수의 배열 생성
print('random.randint() : 일정 구간의 랜덤 정수의 배열 생성')
print(np.random.randint(0, 10, (3, 3))) # 0부터 10까지 3 x 3 배열 생성
print()

# random.normal() : 정규분포를 고려한 랜덤한 수의 배열 생성
print('random.normal() : 정규분포를 고려한 랜덤한 수의 배열 생성')
print(np.random.normal(0, 1, (3, 3))) # 평균=0 표준편차=1 3 x 3배열 생성
print()

# random.rand() : 균등분포를 고려한 랜덤한 수의 배열 생성
print('random.rand() : 균등분포를 고려한 랜덤한 수의 배열 생성')
print(np.random.rand(3, 3))
print()

# random.randn() : 표준 정규 분포를 고려한 랜덤한 수의 배열 생성
print('random.randn() : 표준 정규 분포를 고려한 랜덤한 수의 배열 생성')
print(np.random.randn(3, 3))
print()
# =============================================================================

