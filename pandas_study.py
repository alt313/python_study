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

# Series 객체
print('Series 객체')
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

# 인구수
print('인구수')
pop_tuple = {'서울특별시' : 9720846,
             '부산광역시' : 3404423,
             '인천광역시' : 2947217,
             '대구광역시' : 2427954,
             '대전광역시' : 1471040,
             '광주광역시' : 1455048}
print(pop_tuple)
population = pd.Series(pop_tuple)
print(population)
print(population['서울특별시':'인천광역시'])
print()

# DataFrame 객체
print('DataFrame 객체')
print(pd.DataFrame([{'A' : 2, 'B' : 4, 'D' : 3},
                    {'A' : 4, 'B' : 5, 'C' : 7}]))
print(pd.DataFrame(np.random.rand(5, 5),
                   columns = ['A', 'B', 'C', 'D', 'E'],
                   index = [1, 2, 3, 4, 5]))
print()

# 남자인구수
print('남자 인구수')
male_tuple = {'서울특별시' : 4732275,
              '부산광역시' : 1668618,
              '인천광역시' : 1476813,
              '대구광역시' : 1198815,
              '대전광역시' : 734441,
              '광주광역시' : 720060}
male = pd.Series(male_tuple)
print(male)
print()

# 여자인구수
print('여자 인구수')
female_tuple = {'서울특별시' : 4988571,
                '부산광역시' : 1735805,
                '인천광역시' : 1470404,
                '대구광역시' : 1229139,
                '대전광역시' : 736599,
                '광주광역시' : 734988}
female = pd.Series(female_tuple)
print(female)
print()

# 인구수, 남자인구수, 여자인구수 하나의 DataFrame으로 만들기
print('인구수, 남자인구수, 여자인구수, 하나의 DataFrame으로 만들기')
korea_df = pd.DataFrame({'인구수': population,
                         '남자인구수': male,
                         '여자인구수': female})
print(korea_df)
print(korea_df.index)
print(korea_df.columns)
print(korea_df['여자인구수'])
print(korea_df['서울특별시':'인천광역시'])
print()

# Index 객체

# Index : 일반적인 index객체이며, numpy배열 형식으로 축의 이름 표현
# Int64Index : 정수 값을 위한 index
# MultiIndex : 단일 축에 여러 단계 색인을 표현하는 계층적 index객체
# DatetimeIndex : numpy의 datetime64타입으로 타임스탬프 저장
# PeriodIndex : 기간 데이터를 위한 index
idx = pd.Index([2, 4, 6, 8, 10])
print(idx)
print(idx[1])
print(idx[1:2:2])
print(idx[-1::])
print(idx[::2])
print(idx.size)
print(idx.shape)
print(idx.ndim)
print(idx.dtype)
print()

# Index 연산

print('Index 연산')
idx1 = pd.Index([1, 2, 4, 6, 8])
idx2 = pd.Index([2, 4, 5, 6, 7])
print(idx1)
print(idx2)
print()

# append : 색인 객체를 추가한 새로운 색인 반환
print('append : 색인 객체를 추가한 새로운 색인 반환')
print(idx1.append(idx2))
print()


# difference : 색인의 차집합 반환
print('difference : 색인의 차집합 반환')
print(idx1.difference(idx2))
print()

# intersection : 색인의 교집합 반환
print('intersection : 색인의 교집합 반환')
print(idx1.intersection(idx2))
print(idx1 & idx2)
print()

# union : 색인의 합집합 반환
print('union : 색인의 합집합 반환')
print(idx1.union(idx2))
print(idx1 | idx2)
print()

# delete : 색인이 삭제된 새로운 색인 반환
print('delete : 색인이 삭제된 새로운 색인 반환')
print(idx1.delete(0))
print()

# drop : 값이 삭제된 새로운 색인 반환
print('drop : 값이 삭제된 새로운 색인 반환')
print(idx1.drop(1))
print()

# insert : 색인이 추가된 새로운 색인 반환
# is_monotonic : 색인이 단조정을 가지면 True
# is_union : 중복되는 색인이 없다면 False
# unique : 색인에서 중복되는 요소 제거하고 유일한 값만 반환
# =============================================================================





# 인덱싱(Indexing)
# =============================================================================

# 인덱싱
print('인덱싱')
s = pd.Series([0, 0.25, 0.5, 0.75, 1.0],
              index = ['a', 'b', 'c', 'd', 'e'])
print(s['b'])
print('b' in s)
print(s.keys())
print(list(s.items()))
s['f'] = 1.25
print(s)
print(s[0:4])
print(s[(s > 0.4) & (s < 0.8)])
print()

# Series 인덱싱
print('Series 인덱싱')
s = pd.Series(['a', 'b', 'c', 'd', 'e'],
              index = [1, 3, 5, 7, 9])
print(s)
print(s[1])
print(s[2:4])
print()

# iloc : 인덱싱을 숫자로 한다.
print(s.iloc[1])
print(s.iloc[2:4])
print()

# reindex : 인덱스를 재구성
print(s.reindex(range(10)))
print(s.reindex(range(10), method = 'bfill')) # 'bfill' : 비어있다면 전값으로 채움
print()

# DataFrame 인덱싱

# df[ ] or df.~ : 일반 색인
print('df[ ] or df.~ : 일반 색인')
print(korea_df['남자인구수'])
print(korea_df.남자인구수)
print(korea_df.여자인구수)
korea_df['남여비율'] = (korea_df['남자인구수'] * 100 / korea_df['여자인구수'])
print(korea_df.남여비율)
print(korea_df.values)
print(korea_df.T)
print(korea_df.values[0])
print(korea_df['인구수'])
print()

# df.loc : 라벨값으로 색인
# df.loc[로우, 컬럼]
print('df.loc : 라벨값으로 색인')
print(korea_df.loc[:'인천광역시', :'남자인구수'])
print(korea_df.loc[(korea_df.여자인구수 > 1000000)])
print(korea_df.loc[(korea_df.인구수 < 2000000)])
print(korea_df.loc[(korea_df.인구수 > 2500000)])
print(korea_df.loc[korea_df.남여비율 > 100])
print(korea_df.loc[(korea_df.인구수 > 2500000) & (korea_df.남여비율 > 100)])
print()

# df.iloc : 정수로 색인
# df.iloc[로우, 컬럼]
print('df.iloc : 정수로 색인')
print(korea_df.iloc[:3, :2])
print()

# 다중인덱싱

# 다중 인덱스 Series
print('다중 인덱스 Series')
idx_tuples = [('서울특별시', 2010), ('서울특별시', 2020),
              ('부산광역시', 2010), ('부산광역시', 2020),
              ('인천광역시', 2010), ('인천광역시', 2020),
              ('대구광역시', 2010), ('대구광역시', 2020),
              ('대전광역시', 2010), ('대전광역시', 2020),
              ('광주광역시', 2010), ('광주광역시', 2020)]
print(idx_tuples)
pop_tuples = [10312545, 9720846,
              2567910, 3404423,
              2758296, 2947217,
              2511676, 2427954,
              1503664, 1471040,
              1454636, 1455048]
print(pop_tuples)
population = pd.Series(pop_tuples, index = idx_tuples)
print(population)
print()

midx = pd.MultiIndex.from_tuples(idx_tuples)
print(midx)
population = population.reindex(midx)
print(population)
print(population[:, 2020])
print(population['대전광역시', :])
print()

korea_mdf = population.unstack()
print(korea_mdf)
print(korea_mdf.stack())
print()

male_tuples = [5111259, 4732275,
              1773170, 1668618,
              1390356, 1476813,
              1255245, 1198815,
              753648, 734441,
              721780, 720060]
print(male_tuples)
print()

female_tuples = [5201286, 4988571,
                 1794740, 1735805,
                 1367940, 1470404,
                 1256431, 1229139,
                 750016, 736599,
                 732856, 734988]
print(female_tuples)
print()

korea_mdf = pd.DataFrame({'총인구수' : population,
                          '남자인구수' : male_tuples,
                          '여자인구수' : female_tuples})
print(korea_mdf)
print()

ratio = korea_mdf['남자인구수'] * 100 / korea_mdf['여자인구수']
print(ratio)
print(ratio.unstack())
print()

korea_mdf = pd.DataFrame({'총인구수' : population,
                          '남자인구수' : male_tuples,
                          '여자인구수' : female_tuples,
                          '남여비율' : ratio})
print(korea_mdf)
print()

# 다중 인덱스 생성

print('다중 인덱스 생성')
df = pd.DataFrame(np.random.rand(6, 3),
                  index = [['a', 'a', 'b', 'b', 'c', 'c'], [1, 2, 1, 2, 1, 2]],
                  columns = ['c1', 'c2', 'c3'])
print(df)
print(pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b', 'c', 'c'], [1, 2, 1, 2, 1, 2]]))
print(pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2), ('c', 1), ('c', 2)]))
print(pd.MultiIndex.from_product([['a', 'b', 'c'], [1, 2]]))
print(pd.MultiIndex(levels = [['a', 'b', 'c'], [1, 2]],
                    codes = [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]]))
print()

population.index.names = ['행정구역', '년도']
print(population)
print()

idx = pd.MultiIndex.from_product([['a', 'b', 'c'], [1, 2]],
                                 names = ['name1', 'name2'])
cols = pd.MultiIndex.from_product([['c1', 'c2', 'c3'], [1, 2]],
                                  names = ['col_names1', 'col_names2'])
data = np.round(np.random.rand(6, 6), 2)
print(idx)
print(cols)
print(data)
mdf = pd.DataFrame(data, index = idx, columns = cols)
print(mdf)
print(mdf['c2'])
print()

# 인덱싱 및 슬라이싱

print('인덱싱 및 슬라이싱')
print(population)
print(population['인천광역시', 2010])
print(population[:, 2010])
print(population[population > 3000000])
print(population[['대구광역시', '대전광역시']])
print()

print(mdf)
print(mdf['c2', 1])
print(mdf.iloc[:3, :4])
print(mdf.loc[:, ('c2', 1)])
idx_slice = pd.IndexSlice
print(mdf.loc[idx_slice[:, 2], idx_slice[:, 2]])
print()

# 다중 인덱스 재정렬

print('다중 인덱스 재정렬')
print(korea_mdf)
korea_mdf = korea_mdf.sort_index()
print(korea_mdf)
print(korea_mdf['서울특별시':'인천광역시'])
print(korea_mdf.unstack(level = 0))
print(korea_mdf.unstack(level = 1))
print(korea_mdf.stack())
print()

idx_flat = korea_mdf.reset_index(level = 0)
print(idx_flat)
idx_flat2 = korea_mdf.reset_index(level = (0, 1))
print(idx_flat2)
print(idx_flat2.set_index(['행정구역','년도']))
print()
# =============================================================================




# 데이터 연산
# =============================================================================

s = pd.Series(np.random.randint(0, 10, 5))
print(s)
df = pd.DataFrame(np.random.randint(0, 10, (3, 3)),
                  columns = ['A', 'B', 'C'])
print(df)
print(np.exp(s))
print(np.cos(df * np.pi / 4))
print()

s1 = pd.Series([1, 3, 5, 7, 9], index = [0, 1, 2, 3, 4])
s2 = pd.Series([2, 4, 6, 8, 10], index = [1, 2, 3, 4, 5])
print(s1)
print(s2)
print(s1 + s2) # index가 없는것 끼리는 계산을 못해서 NaN출력
print()

# fill_value = 0 옵션 : NaN자리가 있다면 0으로 채움
print('fill_value = 0 옵션 : NaN자리가 있다면 0으로 채움')
print(s1.add(s2, fill_value = 0)) 
print() 

df1 = pd.DataFrame(np.random.randint(0, 20, (3, 3)),
                   columns = list('ACD'))
df2 = pd.DataFrame(np.random.randint(0, 20, (5, 5)),
                   columns = list('BAECD'))
print(df1)
print(df2)
print(df1 + df2)
fvalue = df1.stack().mean()
# stack()을 해준후 한줄로 만들어서 평균을 구해서 변수에 할당
print(fvalue)
print(df1.add(df2, fill_value = fvalue))
print()

# 연산자 범용 함수

# add() : 더하기
print('add() : 더하기')
a = np.random.randint(1, 10, (3, 3))
print(a)
print(a + a[0])
df = pd.DataFrame(a, columns = list('ABC'))
print(df)
print(df + df.iloc[0])
print(df.add(df.iloc[0]))
print()

# sub() / subtract() : 빼기
print('sub() / subtract() : 빼기')
print(a)
print(a- a[0])
print(df)
print(df - df.iloc[0])
print(df.sub(df.iloc[0]))
print(df.subtract(df['B'], axis = 0))

# mul() / multiply() : 곱하기
print('mul() / multiply() : 곱하기')
print(a)
print(a * a[1])
print(df)
print(df * df.iloc[1])
print(df.mul(df.iloc[1]))
print(df.multiply(df.iloc[2]))
print()

# truediv() / div() / divide() : 나누기, floordiv() : 몫
print('truediv() / div() / divide() : 나누기, floordiv() : 몫')
print(a)
print(a / a[0])
print(df)
print(df / df.iloc[0])
print(df.truediv(df.iloc[0]))
print(df.div(df.iloc[0]))
print(df.divide(df.iloc[0]))
print(a // a[0])
print(df.floordiv(df.iloc[0]))
print()

# mid() : 나머지 
print('mod() : 나머지')
print(a)
print(a % a[0])
print(df.mod(df.iloc[0]))
print()

# pow() : 제곱
print('pow() : 제곱')
print(a)
print(a ** a[0])
print(df)
print(df.pow(df.iloc[0]))
print()


# 정렬(Sort)

s = pd.Series(range(5), index = ['A', 'D', 'B', 'C', 'E'])
print(s)
print(s.sort_index())
print(s.sort_values())
print()

df = pd.DataFrame(np.random.randint(0, 10, (4, 4)),
                  index = [2, 4, 1, 3],
                  columns = list('BDAC'))
print(df)
print(df.sort_index())
print(df.sort_values(by = 'A')) # A컬럼을 기준으로 정렬
print(df.sort_values(by = ['A', 'C'])) # A를 기준으로 정렬하고 그다음 C를 정렬
print(df.sort_index(axis = 1)) # 컬럼을 정렬
print()

# 순위(Ranking)

# 메소드
# average : 기본값, 순위에 같은 값을 가지는 항목들의 평균값을 사용
# min : 같은 값을 가지는 그룹을 낮은 순위로 지정
# max : 같은 값을 가지는 그룹을 높은 순위로 지정
# first : 데이터 내의 위치에 따라 순위 지정
# dense : 같은 그룹 내에서 모두 같은 순위를 적용하지 않고 1씩 증가
s = pd.Series([-2, 4, 7, 3, 0, 7, 5, -4, 2, 6])
print(s)
print(s.rank())
print(s.rank(method = 'first'))
print(s.rank(method = 'max'))
print()

# 고성능 연산

nrows, ncols = 100000, 100
df1, df2, df3, df4 = (pd.DataFrame(np.random.rand(nrows, ncols)) for i in range(4))
# %timeit df1 + df2 + df3 + df4
# %timeit pd.eval('df1 + df2 + df3 + df4')
# pd.eval을 쓰면 연산이 조금더 빨라진다.
# %timeit df1 * -df2 / (-df3 * df4)
# %timeit pd.eval('df1 * -df2 / (-df3 * df4)'
# %timeit (df1 < df2) & (df2 <= df3) & (df3 != df4)
# %timeit pd.eval('(df1 < df2) & (df2 <= df3) & (df3 != df4)')
# 아나콘다 Spyder에서는 되지만 파이참에서는 실행되지 않음
print()

df = pd.DataFrame(np.random.rand(1000000, 5), columns = ['A', 'B', 'C', 'D', 'E'])
print(df.head())
# %timeit df['A'] + df['B'] / df['C'] - df['D'] * df['E']
# %timeit pd.eval('df.A + df.B / df.C - df.D * df.E')
# %timeit df.eval('A + B / C - D * E') # 컬럼명만 입력해도 된다.
df.eval('R = A + B / C - D * E', inplace = True) # 바로 R컬럼을 생성해서 추가한다.
print(df.head())
print()

col_mean = df.mean(1)
print(df['A'] + col_mean)
print(df.eval('A + @col_mean'))
# @를 쓰고 변수명을쓰면 외부에 있는 변수를 쓸 수 있다.
print(df[(df.A < 0.5) & (df.B < 0.5) & (df.C > 0.5)])
print(pd.eval('df[(df.A < 0.5) & (df.B < 0.5) & (df.C > 0.5)]'))
print(df.query('(A < 0.5) and (B < 0.5) and (C > 0.5)'))
print()

col_mean = df['D'].mean()
print(df[(df.A < col_mean) & (df.B < col_mean)])
print(df.query('(A < @col_mean) and (B < @col_mean)'))
print()
# =============================================================================
