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

# numpy호출
import numpy as np
# numpy 현재 버전 출력
print('numpy 현재 버전 출력')
print(np.__version__)
print()
# =============================================================================




# 배열 생성
# =============================================================================

# 1차원 배열 생성
a1 = np.array([1, 2, 3, 4, 5])

# 배열 출력
print('배열 출력')
print(a1)
print()

# 타입 출력
print('타입 출력')
print(type(a1))
print()

# 배열의 모양을 출력해주는 명령어
print('배열의 형태출력')
print(a1.shape)
print()

# 배열에 색인을 통해서 출력
print('배열의 색인을 통한 출력')
print(a1[0], a1[1], a1[2], a1[3], a1[4])
print()

# 색인을 통해 값 수정
print('색인을 통해 값 수정')
a1[0] = 4
a1[1] = 5
a1[2] = 6
print(a1)
print()


# 2차원 배열 생성
a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# 2차원 배열 출력
print('2차원 배열 출력')
print(a2)
print()

# 배열의 형태 출력
print('배열의 형태 출력')
print(a2.shape)
print()

# 2차원 배열 색인 방법
print('2차원 배열 색인 출력')
print(a2[0, 0], a2[1, 1], a2[2, 2])
print()

# 3차원 배열 생성
a3 = np.array([ [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ],
                [ [10, 11, 12], [13, 14, 15], [16, 17, 18] ],
                [ [19, 20, 21], [22, 23, 24], [25, 26, 27] ] ] )

# 3차원 배열 출력
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

# 표준데이터 타입 

# type을 정수
print('type을 정수')
print(np.zeros(20, dtype = int))
print()

# type을 true(1), false(0)
print('type을 true(1), false(0)')
print(np.ones((3,3), dtype = bool))
print()

# type을 실수
print('type을 실수')
np.full((3, 3), 1, dtype = float)
print()

# 날짜/시간 배열 생성

# 2020-01-01 문자열을 날짜로 만들기 
print('2020-01-01 문자열을 날짜로 만들기 ')
date = np.array('2020-01-01', dtype = np.datetime64)
print(date)
print()

# 날짜로만든 2020-01-01에 배열 0부터 11를 더함
print('날짜로만든 2020-01-01에 배열 0부터 11를 더함')
print(date + np.arange(12))
print()

# 날짜와 시간 표현
print('날짜와 시간 표현')
datetime = np.datetime64('2020-06-01 12:00')
print(datetime)
print()

# 날짜 시분초.나노초 표현
print('날짜 시분초.나노초 표현')
datetime = np.datetime64('2020-06-01 12:00:11.11', 'ns')
print(datetime)
print()
# =============================================================================



# 배열 조회
# =============================================================================

# 배열 속성정보
def array_info(array):
    print(array)
    print('ndim : ', array.ndim)         # 배열의 차원
    print('shape : ', array.shape)       # 배열의 모양
    print('dtype : ', array.dtype)       # 배열의 데이터 타입
    print('size : ', array.size)         # 배열에 있는 값 갯수
    print('itemsize : ', array.itemsize) # 각 값이 가지고 있는 byte
    print('nbytes : ', array.nbytes)     # 전체 byte
    print('strides : ', array.strides)   # 각 하나의 byte와 다음 줄로 넘어갈때 byte 출력
    # 여러개의 정보를 한번에 보기 위해 함수를 생성
array_info(a2)
print('''
ndim : 배열의 차원
shape : 배열의 모양
dtype : 배열의 데이터 타입
size : 배열에 있는 값 갯수
itemsize : 각 값이 가지고 있는 byte
nbytes : 전체 byte
strides : 각 하나의 byte와 다음 줄로 넘어갈때 byte 출력
      ''')
print()

# 인덱싱

# 1차원 배열 인덱싱
print('순서대로 전체, 첫번째, 세번째, 뒤에서 첫번째, 뒤에서 두번째 출력')
print(a1)
print(a1[0])
print(a1[2])
print(a1[-1])
print(a1[-2])
print()

# 2차원 배열 인덱싱
print('순서대로 전체, 첫줄 첫번째, 첫줄 세번째, 두번째줄 뒤에서 첫번째, 세번째줄 뒤에서 두번째 출력')
print(a2)
print(a2[0, 0])
print(a2[0, 2])
print(a2[1, -1])
print(a2[2, -2])
print()

# 3차원 배열 인덱싱
print('순서대로 전체, 1층 첫줄 첫번째, 2층 두번째줄 두번째, 3층 세번째줄 세번째, 3층 세번째줄 뒤에서 첫번째 출력')
print(a3)
print(a3[0, 0, 0])
print(a3[1, 1, 1])
print(a3[2, 2, 2])
print(a3[2, -1, -1])
print()

# =============================================================================
