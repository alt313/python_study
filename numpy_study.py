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

# 슬라이싱

# 1차원 배열 슬라이싱 방법 배열[start:stop:step]
print('1차원 배열 슬라이싱 방법 배열[start:stop:step]')
print(a1)       # 전체
print(a1[0:2])  # 처음부터 세번째 전 까지
print(a1[0:])   # 처음부터 끝까지
print(a1[:1])   # 처음부터 두번째 전 까지
print(a1[::2])  # 처음부터 끝까지 2칸 간격으로 출
print(a1[::-1]) # 처음부터 끝까지 뒤에서부터 출력
print()

# 2차원 배열 슬라이싱 방법 배열[start:stop:step, start:stop:step]
print('2차원 배열 슬라이싱 방법 배열[start:stop:step, start:stop:step]')
print(a2)
print(a2[1])
print(a2[1, :])
print(a2[:2, :2])
print(a2[1:, ::-1])
print(a2[::-1, ::-1])
print()

# 불리언 인덱싱

# 불리언 값이 True인 위치인 것만 출력
print('불리언 값이 True인 위치인 것만 출력')
print(a1)
bi = [False, True, True, False, True]
print(bi)
print(a1[bi])
bi = [True, False, True, True, False]
print(bi)
print(a1[bi])
print()

# 2차원 불리언 인덱싱 방법
print('2차원 불리언 인덱싱 방법')
print(a2)
bi_2 = np.random.randint(0, 2, (3, 3), dtype = bool)
print(bi_2)
print(a2[bi_2])
print()

# 팬시 인덱싱

# 1차원 배열 팬시 인덱싱
print('1차원 배열 팬시 인덱싱')
print(a1)
print([a1[0], a1[2]])
ind = [0, 2]
print(ind)
print(a1[ind])
# 1차원 배열이라도 2차원 배열처럼 출력할 수 있다.
print('1차원 배열이라도 2차원 배열처럼 출력할 수 있다.')
ind_2 = np.array([[0, 1],
                 [2, 0]])
print(ind_2)
print(a1[ind_2]) 
print()

# 2차원 배열 팬시 인덱싱
print('2차원 배열 팬시 인덱싱')
print(a2)
row = np.array([0, 2])
print(row)
col = np.array([1, 2])
print(col)
print(a2[row, col])
print(a2[row, :])
print(a2[:, col])
print(a2[row, 1])
print(a2[2, col])
print(a2[row, 1:])
print(a2[1:, col])
print()
# =============================================================================



# 배열 값 삽입/수정/삭제/복사
# =============================================================================
 
# 배열 값 삽입

# insert() : 배열의 특정 위치에 값 삽입
# axis를 지정하지 않으면 1차원 배열로 변환
# 추가할 방향을 axis로 지정
# 원본 배열 변경없이 새로운 배열 반환
print('1차원 배열에 값 삽입')
print(a1)
b1 = np.insert(a1, 0, 10)
print(b1)
c1 = np.insert(a1, 2, 10)
print(c1)
print()

print('2차원 배열에 값 삽입')
print(a2)
b2 = np.insert(a2, 1, 10, axis = 0) # axis : 축 지정
print(b2)
c2 = np.insert(a2, 1, 10, axis = 1)
print(c2)

# 배열 값 수정

# 배열의 인덱싱으로 접근하여 값 수정
# 1차원 배열의 값 수정
print('1차원 배열의 값 수정')
print(a1)
a1[0] = 1
a1[1] = 2
a1[2] = 3
print(a1)
a1[:1] = 9
print(a1)
idx = np.array([1, 3, 4])
print(idx)
a1[idx] = 0
print(a1)
a1[idx] += 4
print(a1)
print()

print('2차원 배열의 값 수정')
print(a2)
a2[0, 0] = 1
a2[1, 1] = 2
a2[2, 2] = 3
a2[0] = 1
print(a2)
print()

# 배열 값 삭제

# delete() : 배열의 특정 위치에 값 삭제
# axis를 지정하지 않으면 1차원 배열로 변환
# 삭제할 방향을 axis로 지정
# 원본 배열 변경없이 새로운 배열 반환

print('1차원 배열 값 삭제')
print(a1)
b1 = np.delete(a1, 1)
print(b1)
print(a1)
print()

print('2차원 배열 값 삭제')
b2 = np.delete(a2, 1, axis =0)
print(b2)
print(a2)
print()


# 배열 복사

# 일반적인 배열 복사 방식
print('일반적인 배열 복사 방식')
print(a2)
print(a2[:2, :2])
a2_sub = a2[:2, :2]
print(a2_sub)
a2_sub[:, 1] = 0
print(a2_sub) 
print(a2)
print()
# 원본배열의 값도 바뀐다.
# 원본배열의 값은 바뀌면 안되기 때문에 copy() 메서드 사용
# copy() : 배열이나 하위 배열 내의 값을 명시적으로 복사

# copy() 메서드로 배열 복사
print('copy() 메서드로 배열 복사')
a2_sub_copy = a2[:2, :2].copy()
print(a2_sub_copy)
a2_sub_copy[:, 1] = 1
print(a2_sub_copy)
print(a2)
print()
# 이번에는 원본배열의 값은 바뀌지않았다.
# =============================================================================



# 배열 변환
# =============================================================================

# 배열 전치 및 축 변경

print('배열 전치 및 축 변경')
print(a2)
print(a2.T)
print(a3)
print(a3.T)

# 배열 재구조화

# reshape() : 배열의 형상을 변경
print('reshape() : 배열의 형상을 변경')
n1 = np.arange(1, 10)
print(n1)
print(n1.reshape(3, 3))
print()

# newaxis() : 새로운 축 추가
print('newaxis() : 새로운 축 추가')
print(n1)
print(n1[np.newaxis, :5])
print(n1[:5, np.newaxis])
print()

# 배열 크기 변경

# 배열 모양 변경
print('배열 모양 변경')
n2 = np.random.randint(0, 10, (2, 5))
print(n2)
n2.resize((5, 2)) # 바로 변경된다.
print(n2)
print()

# 배열 크기 증가
# 남은 공간은 0으로 채워진다.
print('배열 크기 증가')
n2.resize((5, 5), refcheck = False)
# 현재 python 3.8버전 에서만 그런지 모르지만 강의에서는 뒤에 옵션을 붙이지 않고
# 사용했지만 refcheck=False 옵션을 써줘야 가능하다.
print(n2)
print()

# 배열 크기 감소
# 포함되지 않은 값은 삭제됨
print('배열 크기 감소')
n2.resize((3, 3), refcheck = False)
print(n2)
print()

# 배열 추가

# append() : 배열 끝에 값 추가
# axis지정이 없다면 1차원 배열 형태로 변형되어 결합된다.
print('append() : 배열 끝에 값 추가')
a2 = np.arange(1, 10).reshape(3, 3)
print(a2)
b2 = np.arange(10, 19).reshape(3, 3)
print(b2)
c2 = np.append(a2, b2)
print(c2)
print()

# axis를 0으로 지정
# shape[0]을 제외한 나머지 shape은 같아야 함
print('axis를 0으로 지정')
c2 = np.append(a2, b2, axis = 0)
print(c2)
print()

# axis를 1로 지정
# shape[1]을 제외한 나머지 shape은 같아야 함
print('axis를 1로 지정')
c2 = np.append(a2, b2, axis = 1)
print(c2)
print()

# 배열 연결

# concatenate() : 튜플이나 배열의 리스트를 인수로 사용해 배열 연결
print('concatenate() : 튜플이나 배열의 리스트를 인수로 사용해 배열 연결')
a1 = np.array([1, 3, 5])
print(a1)
b1 = np.array([2, 4, 6])
print(b1)
print(np.concatenate([a1, b1]))
c1 = np.array([7, 8, 9])
print(c1)
print(np.concatenate([a1, b1, c1]))
print()

print('axis=1로 추가')
a2 = np.array([[1, 2, 3],
                [4, 5, 6]])
print(a2)
print(np.concatenate([a2, a2], axis = 1))

# vstack() : 수직 스택, 1차원으로 연결
print('vstack() : 수직 스택, 1차원으로 연결')
np.vstack([a2, a2])
print()

# hstack() : 수평 스택, 2차원으로 연결
print('hstack() : 수평 스택, 2차원으로 연결')
np.hstack([a2, a2])
print()

# dstack() : 깊이 스택, 3차원으로 연결
print('dstack() : 깊이 스택, 3차원으로 연결')
np.dstack([a2, a2])
print()

# stack() : 새로운 차원으로 연결
print('stack() : 새로운 차원으로 연결')
np.stack([a2, a2])
print()

# 배열 분할


# =============================================================================


