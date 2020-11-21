# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:37:36 2020

@author: msi
"""

# pandas 호출 및 버전 확인
# =============================================================================

# pandas 호출
print('pandas와 numpy 호출')
import numpy as np
import pandas as pd
print()

# pandas 현재 버전 출력
print('pandas 현재 버전 출력')
print(pd.__version__)
print()
# =============================================================================




# pandas 객체
# =============================================================================

# series 객체
s = pd.Series([0, 0.25, 0.5, 0.75, 1.0])
print(s)
print(s.values) # 값만 출력
print(s.index)  # 인덱스 출력
print(s[1])
print(s[1:4])
print()

s = pd.Series([0, 0.25, 0.5, 0.75, 1.0],
              index = ['a', 'b', 'c', 'd', 'e'])
print(s)
print(s['c'])
print(s[['c', 'd', 'e']])
print('b' in s)
print()

s = pd.Series([0, 0.25, 0.5, 0.75, 1.0],
              index = [2, 4, 6, 8, 10])
print(s)
print(s[4])
print(s[2:])
print(s.unique()) # Series에서 유니크한 값만 출력해준다
print(s.value_counts()) # Series에서 각 값들의 갯수 출력
print(s.isin([0.25, 0.75])) # 값이 있으면 True, 없으면 False로 알려준다
print()

pop_tuple = {'서울특별시' : 9720846,
             '부산광역시' : 3404423,
             '인천광역시' : 2947217,
             '대구광역시' : 2427954,
             '대전광역시' : 1471040,
             '광주광역시' : 1455048}
print(pop_tuple)
population = pd.Series(pop_tuple)
print(population)
# =============================================================================
