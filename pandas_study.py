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





# 데이터 결합
# =============================================================================

# concat() / append()

# concat()
print('concat()')
s1 = pd.Series(['a', 'b'], index = [1, 2])
s2 = pd.Series(['c', 'd'], index = [3, 4])
print(s1)
print(s2)
print(pd.concat([s1, s2]))
print()

# DataFrame concat(세로) axis = 0(기본)
print('DataFrame concat(세로) axis = 0(기본)')
def create_df(cols, idx):
    data = {c: [str(c.lower()) + str(i) for i in idx] for c in cols}
    return pd.DataFrame(data, idx)
df1 = create_df('AB', [1, 2])
df2 = create_df('AB', [3, 4])
print(df1)
print(df2)
print(pd.concat([df1, df2]))
print()

# DataFrame concat(가로) axis = 1
print('DataFrame concat(가로) axis = 1')
df3 = create_df('AB', [0, 1])
df4 = create_df('CD', [0, 1])
print(df3)
print(df4)
print(pd.concat([df3, df4], axis = 1))
print()

# index무시하고 새로생성하여 합치기 ignore_index = True
print('index무시하고 새로생성하여 합치기 ignore_index = True')
print(pd.concat([df1, df3]))
print(pd.concat([df1, df3], ignore_index=True))
print()

# 멀티인덱스 생성하여 합치기 keys = []
print('멀티인덱스 생성하여 합치기 keys = []')
print(pd.concat([df1, df3], keys = ['X', 'Y']))
print()

# join = 'inner' : 둘다 존재하는 데이터만 합친다
print("join = 'inner' :  둘다 존재하는 데이터만 합친다.")
df5 = create_df('ABC', [1, 2])
df6 = create_df('BCD', [3, 4])
print(df5)
print(df6)
print(pd.concat([df5, df6]))
print(pd.concat([df5, df6], join = 'inner'))
print()

# append()
print('append()')
print(df5.append(df6))
print()

# 병합과 조인

df1 = pd.DataFrame({'학생' : ['홍길동', '이순신', '임꺽정', '김유신'],
                    '학과' : ['경영학과', '교육학과', '컴퓨터학과', '통계학과']})
df2 = pd.DataFrame({'학생' : ['홍길동', '이순신', '임꺽정', '김유신'],
                    '입학년도' : [2012, 2016, 2019, 2020]})
print(df1)
print(df2)
print()

df3 = pd.merge(df1, df2)
print(df3)
print()

df4 = pd.DataFrame({'학과' : ['경영학과', '교육학과', '컴퓨터학과', '통계학과'],
                    '학과장' : ['황희', '장영실', '안창호', '정약용']})
print(df4)
print(pd.merge(df3, df4))
print()

df5 = pd.DataFrame({'학과' : ['경영학과', '교육학과', '교육학과', '컴퓨터학과', '컴퓨터학과', '통계학과'],
                    '과목' : ['경영개론', '기초수학', '물리학', '프로그래밍', '운영체제', '확률론']})
print(df5)
print(pd.merge(df1, df5))
print()

df6 = pd.DataFrame({'이름' : ['홍길동', '이순신', '임꺽정', '김유신'],
                    '성적' : ['A', 'A+', 'B', 'A+']})
print(df6)
print(pd.merge(df1, df6, left_on='학생', right_on='이름'))
print(pd.merge(df1, df6, left_on='학생', right_on='이름').drop('이름', axis = 1))
print()

mdf1 = df1.set_index('학생')
mdf2 = df2.set_index('학생')
print(mdf1)
print(mdf2)
print(pd.merge(mdf1, mdf2, left_index = True, right_index = True))
print(mdf1.join(mdf2))
print(pd.merge(mdf1, df6, left_index = True, right_on = '이름'))
print()

df7 = pd.DataFrame({'이름' : ['홍길동', '이순신', '임꺽정'],
                    '주문음식' : ['햄버거', '피자', '짜장면']})
df8 = pd.DataFrame({'이름' : ['홍길동', '이순신', '김유신'],
                    '주문음료' : ['콜라', '사이다', '커피']})
print(df7)
print(df8)
print(pd.merge(df7, df8))
print(pd.merge(df7, df8, how = 'inner'))
print(pd.merge(df7, df8, how = 'outer'))
print(pd.merge(df7, df8, how = 'left'))
print(pd.merge(df7, df8, how = 'right'))
print()

df9 = pd.DataFrame({'이름' : ['홍길동', '이순신', '임꺽정', '김유신'],
                    '순위' : [3, 2, 4, 1]})
df10 = pd.DataFrame({'이름' : ['홍길동', '이순신', '임꺽정', '김유신'],
                     '순위' : [4, 2, 3, 1]})
print(df9)
print(df10)
print(pd.merge(df9, df10, on = '이름'))
print(pd.merge(df9, df10, on = '이름', suffixes = ['_인기', '_성적']))
print()
# =============================================================================
 



# 데이터 집계와 그룹 연산
# =============================================================================

# 집계연산

# count : 전체 개수
# min, max : 최소값, 최대값
# cummin, cummax : 누적 최소값, 누적 최대값
# mean, median : 평균값, 중앙값
# mad : 절대 평균 편차
df = pd.DataFrame([[1, 1.2, np.nan],
                   [2.4, 5.5, 4.2],
                   [np.nan, np.nan, np.nan],
                   [0.44, -3.1, -4.1]],
                  index = [1, 2, 3, 4],
                  columns = ['A', 'B', 'C'])
print(df)

# head, tail : 앞의 항목 일부 반환, 뒤의 항목 일부 반환
print('head, tail : 앞의 항목 일부 반환, 뒤의 항목 일부 반환')
print(df.head(2))
print(df.tail(2))
print()

# describe : Series, DataFrame의 각 컬럼에 대한 요약 통계
print('describe : Series, DataFrame의 각 컬럼에 대한 요약 통계')
print(df)
print(df.describe())

# argmin, argmax : 최소값과 최대값의 색인 위치
print('argmin, argmax : 최소값과 최대값의 색인 위치')
print(df)
print(np.argmin(df), np.argmax(df))
print()

# idxmin, idxmax : 최소값과 최대값의 색인값
print('idxmin, idxmax : 최소값과 최대값의 색인값')
print(df)
print(df.idxmin())
print(df.idxmax())
print()

# std, var : 표준편차, 분산
print('std, var : 표준편차, 분산')
print(df)
print(df.std())
print(df.var())
print()

# skew, kurt : 왜도값 계산, 첨도값 계산
print('skew, kurt : 왜도값 계산, 첨도값 계산')
print(df)
print(df.skew())
print(df.kurt())
print()

# sum, cumsum : 전체 항목 합, 누적합
print('sum, cumsum : 전체 항목 합, 누적합')
print(df)
print(df.sum())
print(df.cumsum())
print()

# prod, cumprod : 전체 항목 곱, 누적곱
print('prod, cumprod : 전체 항목 곱, 누적곱')
print(df)
print(df.prod())
print(df.cumprod())
print()

# diff : 1차 산술차 계산
print('diff : 1차 산출차 계산')
print(df)
print(df.diff())
print()

# quantile : 0부터 1까지의 분위수 계산
print('quantile : 0부터 1까지의 분위수 계산')
print(df)
print(df.quantile())
print()

# pct_change : 퍼센트 변화율 계산
print('pct_change : 퍼센트 변화율 계산')
print(df)
print(df.pct_change())
print()

# corr, cov : 상관관계, 공분산 계산
print('corr, cov : 상관관계, 공분산 계산')
print(df)
print(df.corr())
print(df.corrwith(df.B))
print(df.cov())
print()

# GroupBy 연산

df = pd.DataFrame({'c1' : ['a', 'a', 'b', 'b', 'c', 'd', 'b'],
                   'c2' : ['A', 'B', 'B', 'A', 'D', 'C', 'C'],
                   'c3' : np.random.randint(7),
                   'c4' : np.random.random(7)})
print(df)
print(df.dtypes)
print(df['c3'].groupby(df['c1']).mean())
print(df['c4'].groupby(df['c2']).std())
print(df['c4'].groupby([df['c1'], df['c2']]).mean())
print(df['c4'].groupby([df['c1'], df['c2']]).mean().unstack())
print(df.groupby('c1').mean())
print(df.groupby(['c1', 'c2']).mean())
print()

for c1, group in df.groupby('c1'):
    print(c1)
    print(group)
print()
    
for (c1, c2), group in df.groupby(['c1', 'c2']):
    print((c1, c2))
    print(group)
print()

print(df.groupby(['c1', 'c2'])['c4'].mean())
print(df.groupby('c1')['c3'].quantile())
print(df.groupby('c1')['c3'].count())
print(df.groupby('c1')['c4'].median())
print(df.groupby('c1')['c4'].std())
print(df.groupby(['c1', 'c2'])['c4'].agg(['mean', 'min', 'max']))
print(df.groupby(['c1', 'c2'], as_index=False)['c4'].mean())
print(df.groupby(['c1', 'c2'], group_keys=False)['c4'].mean())
print()

def top(df, n = 3, column = 'c1'):
    return df.sort_values(by = column)[-n:]
print(top(df, n=5))
print()

print(df.groupby('c1').apply(top))
print()

# 피벗 테이블

# values : 집계하려는 컬럼 이름 혹은 이름의 리스트. 기본적으로 모든 숫자 컬럼 집계
# index : 피벗테이블의 로우를 그룹으로 묶을 컬럼 이름이나 그룹 키
# columns : 피벗테이블의 컬럼을 그룹으로 묶을 컬럼 이름이나 그룹 키
# dropna : True인 경우 모든 항목이 NA인 컬럼은 포함하지 않음
print(df.pivot_table(['c3', 'c4'],
                     index = ['c1'],
                     columns = ['c2']))
print()

# margins : 부분합이나 총계를 담기 위한 로우/컬럼 추가 여부. 기본값은 False
print('margins : 부분합이나 총계를 담기 위한 로우/컬럼 추가 여부. 기본값은 False')
print(df.pivot_table(['c3', 'c4'],
                     index = ['c1'],
                     columns = ['c2'],
                     margins = True))
print()

# aggfunc : 집계 함수나 함수 리스트. 기본값으로 mean이 사용
print('aggfunc : 집계 함수나 함수 리스트. 기본값으로 mean이 사용')
print(df.pivot_table(['c3', 'c4'],
                     index = ['c1'],
                     columns = ['c2'],
                     margins = True,
                     aggfunc = sum))
print()

# fill_value : 결과 테이블에서 누락된 값 대체를 위한 값
print('fill_value : 결과 테이블에서 누락된 값 대체를 위한 값')
print(df.pivot_table(['c3', 'c4'],
                     index = ['c1'],
                     columns = ['c2'],
                     margins = True,
                     fill_value = 0))
print()

# 범주형 데이터

# add_categories : 기존 카테고리에 새로운 카테고리 추가
# as_ordered : 카테고리에 순서 지정
# as_unordered : 카테고리에 순서 미지정
# remove_categories : 카테고리 제거
# rename_categories : 카테고리 이름 변경
# reorder_categories : 새로운 카테고리에 순서 지정

s = pd.Series(['c1', 'c2', 'c1', 'c2', 'c1'] * 2)
print(s)
print(pd.unique(s))
print(pd.value_counts(s))
print()

code = pd.Series([0, 1, 0, 1, 0] * 2)
print(code)
d = pd.Series(['c1', 'c2'])
print(d)
print(d.take(code))
print()

df = pd.DataFrame({'id' : np.arange(len(s)),
                   'c' : s,
                   'v' : np.random.randint(1000, 5000, size = len(s))})
print(df)
print()

c = df['c'].astype('category')
print(c)
print(c.values)
print(c.values.categories)
print(c.values.codes)
print()

df['c'] = c
print(df.c)
print()

c = pd.Categorical(['c1', 'c2', 'c3', 'c1', 'c2'])
print(c)
print()

categories = ['c1', 'c2', 'c3']
codes = [0, 1, 2, 0, 1]
c = pd.Categorical.from_codes(codes, categories)
print(c)
print(pd.Categorical.from_codes(codes, categories, ordered = True))
print(c.as_ordered())
print(c.codes)
print(c.categories)
print()

# set_categories : 새로운 카테고리로 변경
print('set_categories : 새로운 카테고리로 변경')
c = c.set_categories(['c1', 'c2', 'c3', 'c4', 'c5'])
print(c.categories)
print(c.value_counts())
print(c[c.isin(['c1', 'c3'])])
print()

# remove_unused_categories :사용안하는 카테고리 제거
print('remove_unused_categories :사용안하는 카테고리 제거')
c = c.remove_unused_categories()
print(c.categories)
print()
# =============================================================================





# 문자열 연산
# =============================================================================

# 문자열 연산자

name_tuple = [ 'Suan Lee', 'Steven Jobs', 'Larry Page', 'Elon Musk', None, 'Mark Zuckerberg', 'Jeff Bezos']
names = pd.Series(name_tuple)
print(names)
print()

# lower() : 모든 대소문자가 소문자로 변환된 문자열을 반환
print('lower() : 모든 대소문자가 소문자로 변환된 문자열을 반환')
print(names.str.lower())
print()

print(names.str.len())
print(names.str.split())
print()

# 기타 연산자

# slice() : 각 요소에 슬라이스 적용
# slice_replace() : 각 요소의 슬라이스를 특정 값으로 대체
# cat() : 문자열 연결
# normalize() : 문자열의 유니코드 형태로 반환
# pad() : 문자열의 왼쪽, 오른쪽, 또는 양쪽 공백 추가
# wrap() : 긴 문자열을 주어진 너비보다 짧은 길이의 여러 줄로 나눔
# get_dummies() : DataFrame으로 가변수 추출
print(names.str[0:4])
print()

# get() : 각 요소에 인덱스 지정
print('get() : 각 요소에 인덱스 지정')
print(names.str.split().str.get(-1))
print()

# repeat() : 값 반복
print('repeat() : 값 반복')
print(names.str.repeat(2))
print()

# join() : Series의 각 요소에 있는 문자열을 전달된 구분자와 결합
print('join() : Series의 각 요소에 있는 문자열을 전달된 구분자와 결합')
print(names.str.join('*'))
print()

# 정규표현식

# extract() : 각 요소에.re.match()호출. 문자열로 매칭된 그룹 반환
# replace() : 패턴이 발생한 곳을 다른 문자열로 대체
# contains() : 각 요소에 re.search()호출. 불리언 값 반환
# count() : 패턴 발생 건수 집계
# split() : str.split()과 동일하지만 정규표현식 사용
# rsplit() : str.rsplit()과 동일하지만 정규표현식 사용

# match() : 각 요소에 re.match()호출. 불리언 값 반환
print('match() : 각 요소에 re.match()호출. 불리언 값 반환')
print(names.str.match('([A-Za-z]+)'))
print()

# findall() : 각 요소에 re.findall()호출. 
print('findall() : 각 요소에 re.findall()호출. ')
names.str.findall('([A-Za-z]+)')
print()
# =============================================================================




# 시계열 처리
# =============================================================================
idx = pd.DatetimeIndex(['2019-01-01', '2020-01-01', '2020-02-01', '2020-02-02', '2020-03-01'])
s = pd.Series([0, 1, 2, 3, 4], index = idx)
print(s)
print(s['2020-01-01':])
print(s[:'2020-01-01'])
print(s['2019'])

# 시계열 데이터 구조

from datetime import datetime
dates = pd.to_datetime(['12-12-2019', datetime(2020, 1, 1), '2nd of Feb, 2020', '2020-Mar-4', '20200701'])
print(dates)
print(dates.to_period('D'))
print(dates - dates[0])
print(pd.date_range('2020-01-01', '2020-07-01'))
print(pd.date_range('2020-01-01', periods = 7)) # periods=n: n개 출력
print(pd.date_range('2020-01-01', periods = 7, freq = 'M')) # freq='M': 월단위 말일 출력
print(pd.date_range('2020-01-01', periods = 7, freq = 'H')) # freq='H': 시간단위 출력
print()

idx = pd.to_datetime(['2020-01-01 12:00:00', '2020-01-02 00:00:00'] + [None])
print(idx)
print(idx[2])
print(pd.isnull(idx))
print()

# 시계열 기본

dates = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 4), datetime(2020, 1, 7),
         datetime(2020, 1, 10), datetime(2020, 1, 11), datetime(2020, 1, 15)]
print(dates)
ts = pd.Series(np.random.randn(7), index = dates)
print(ts)
print(ts.index)
print(ts.index[0])
print(ts[ts.index[2]])
print(ts['20200104'])
print(ts['1/4/2020'])
print()

ts = pd.Series(np.random.randn(1000),
               index = pd.date_range('2017-10-01', periods = 1000))
print(ts)
print(ts['2020']) # 2020년에 해당하는 것만 출력
print(ts['2020-06']) # 2020년 06월 해당하는 것만 출력
print(ts[datetime(2020, 6, 20):])
print(ts['2020-06-10':'2020-06-20'])
print()

tdf = pd.DataFrame(np.random.randn(1000, 4), # 1000개, 4개의 컬럼
                   index = pd.date_range('2017-10-01', periods = 1000),
                   columns = ['A', 'B', 'C', 'D'],)
print(tdf)
print(tdf['2020'])
print(tdf.loc['2020-06'])
print(tdf['2020-06-20':])
print(tdf['C'])
print()

ts = pd.Series(np.random.randn(10),
               index = pd.DatetimeIndex(['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03',
                                         '2020-01-04', '2020-01-05', '2020-01-05', '2020-01-06', '2020-01-07']))
print(ts)
print(ts.index.is_unique) # 인덱스가 유니크한지 유니크하지 않으면 False
print(ts['2020-01-01'])
print(ts.groupby(level = 0).mean())
print(pd.date_range('2020-01-01', '2020-07-01'))
print(pd.date_range(start = '2020-01-01', periods = 10))
print(pd.date_range(end = '2020-07-01', periods = 10))
print(pd.date_range('2020-07-01', '2020-07-7', freq = 'B')) # freq='B': 주말제외 영업일만 출력
print()

# 주기와 오프셋

# freq='H', 'T', 'B', 'S': 시간단위 출력, 분단위 출력, 영업가능일(평일) 출력, 초단위 출력
print("freq='H', 'T', 'B', 'S': 시간단위 출력, 분단위 출력, 영업가능일(평일) 출력, 초단위 출력")
print(pd.timedelta_range(0, periods = 12, freq = 'H')) 
print(pd.timedelta_range(0, periods = 60, freq = 'T'))
print(pd.timedelta_range(0, periods = 10, freq = '1H30T')) # 1시간 30분 단위 출력
print(pd.date_range('2020-01-01', periods = 20, freq = 'B'))
print(pd.date_range('2020-01-01', periods = 30, freq = '2H'))
print(pd.date_range('2020-01-01', periods = 20, freq = 'S'))

# 시프트

ts = pd.Series(np.random.randn(5),
               index = pd.date_range('2020-01-01', periods = 5, freq = 'B'))
print(ts)
print(ts.shift(1))
print(ts.shift(3))
print(ts.shift(-2))
print(ts.shift(3, freq = 'B'))
print(ts.shift(2, freq = 'W'))
print()

# 시간대 처리

# 국제 표준시를 기준으로 떨어진 거리만큼 오프셋으로 시간대 처리
# 전 세계의 시간대 정보를 모아놓은 올슨 데이터베이스를 활용한 라이브러리인 pytz사용
import pytz

print(pytz.common_timezones)
tz = pytz.timezone('Asia/Seoul')
dinx = pd.date_range('2020-01-01 09:00', periods = 7, freq = 'B')
ts = pd.Series(np.random.randn(len(dinx)), index = dinx)
print(ts)
print()

print(pd.date_range('2020-01-01 09:00', periods = 7, freq = 'B', tz = 'UTC'))
print()

ts_utc = ts.tz_localize('UTC')
print(ts_utc)
print(ts_utc.index)
print(ts_utc.tz_convert('Asia/Seoul'))
print()

ts_seoul = ts.tz_localize('Asia/seoul')
print(ts_seoul)
print(ts_seoul.tz_convert('UTC'))
print(ts_seoul.tz_convert('Europe/Berlin'))
print(ts.index.tz_localize('America/New_York'))
print()

stamp = pd.Timestamp('2020-01-01 12:00')
stamp_utc = stamp.tz_localize('UTC')
print(stamp_utc)
print(stamp_utc.value)
print(stamp_utc.tz_convert('Asia/Seoul'))
print(stamp_utc.tz_convert('Asia/Seoul').value)
print()

stamp_ny = pd.Timestamp('2020-01-01 12:00', tz = 'America/New_York')
print(stamp_ny)
print(stamp_utc.value)
print(stamp_ny.value)
print(stamp_utc.tz_convert('Asia/Shanghai'))
print()

stamp = pd.Timestamp('2020-01-01 12:00', tz = 'Asia/Seoul')
print(stamp)
from pandas.tseries.offsets import Hour
print(stamp + Hour())
print(stamp + 3 * Hour())
print()

ts1 = ts_utc[:5].tz_convert('Europe/Berlin')
ts2 = ts_utc[2:].tz_convert('America/New_York')
print(ts1)
print(ts2)
ts = ts1 + ts2
print(ts.index) # 결과는 UTC로 나온다
print()

# 기간과 기간 연산

p = pd.Period(2020, freq = 'A-JAN')
print(p)
print(p + 2)
print(p - 3)
print()

p1 = pd.Period(2010, freq = 'A-JAN')
p2 = pd.Period(2020, freq = 'A-JAN')
print(p2 - p1)
print()

pr = pd.period_range('2020-01-01', '2020-06-30', freq = 'M')
print(pd.Series(np.random.randn(6), index = pr))
print()

pidx = pd.PeriodIndex(['2020-1', '2020-2', '2020-4'], freq = 'M')
print(pidx)
print()

p = pd.Period('2020', freq = 'A-FEB')
print(p)
print(p.asfreq('M', how = 'start'))
print(p.asfreq('M', how = 'end'))
print()

p = pd.Period('2020', freq = 'A-OCT')
print(p)
print(p.asfreq('M', how = 'start'))
print(p.asfreq('M', how = 'end'))
print()

pr = pd.period_range('2010', '2020', freq = 'A-JAN')
ts = pd.Series(np.random.randn(len(pr)), index = pr)
print(ts)
print(ts.asfreq('M', how = 'start'))
print(ts.asfreq('B', how = 'end'))  
print()

p = pd.Period('2020Q2', freq = 'Q-JAN')
print(p)
print(p.asfreq('D', how = 'start'))
print(p.asfreq('D', how = 'end'))
print()

pr = pd.period_range('2019Q3', '2020Q3', freq = 'Q-JAN')
ts = pd.Series(np.arange(len(pr)), index = pr)
print(ts)
print()

pr = pd.date_range('2020-01-01', periods = 5, freq = 'Q-JAN')
ts = pd.Series(np.random.randn(5), index = pr)
print(ts)
print(ts.to_period())
print()

pr = pd.date_range('2020-01-01', periods = 5, freq = 'D')
ts = pd.Series(np.random.randn(5), index = pr)
print(ts)
print(ts.to_period('M'))
print(ts.to_period('M').to_timestamp(how = 'start'))
print()

# 리샘플링

# freq : 리샘플링 빈도
# axis : 리샘플링 축
# fill_method : 업샘플링시 보간 수행
# limit : 보간법을 사용할 때 보간을 적용할 최대 기간
# convention : 기간을 리샘플링할 떄 하위 빈도 기간에서 상위 빈도로 변환 시 방식

# kind : 기간또는 타임스탬프집계 구분
print('kind : 기간또는 타임스탬프집계 구분')
dr = pd.date_range('2020-01-01', periods = 200, freq = 'D')
ts = pd.Series(np.random.randn(len(dr)), index = dr)
print(ts)
print(ts.resample('M').mean())
print(ts.resample('M', kind = 'period').mean())
print()

# closed : 다운샘플링 시 각 간격의 포함 위치
print('closed : 다운샘플링 시 각 간격의 포함 위치')
dr = pd.date_range('2020-01-01', periods = 10, freq = 'T')
ts = pd.Series(np.arange(10), index = dr)
print(ts)
print(ts.resample('2T', closed = 'left').sum())
print(ts.resample('2T', closed = 'right').sum())
print()

# label : 다운샘플링 시 집계된 결과 라벨 결정
print('label : 다운샘플링 시 집계된 결과 라벨 결정')
print(ts.resample('2T', closed = 'right', label = 'right').sum())
print()

# loffset : 나뉜 그룹의 라벨을 맞추기 위한 오프셋
print('loffset : 나뉜 그룹의 라벨을 맞추기 위한 오프셋')
print(ts.resample('2T', closed = 'right', label = 'right', loffset = '-1s').sum())
print()

df = pd.DataFrame(np.random.randn(10, 4),
                  index = pd.date_range('2019-10-01', periods = 10, freq = 'M'),
                  columns = (['C1', 'C2', 'C3', 'C4']))
print(df)
print(df.resample('Y').asfreq())
print(df.resample('W-FRI').asfreq())
print(df.resample('H').asfreq())
print(df.resample('H').ffill())
print(df.resample('Q-DEC').mean())
print(df.resample('Y').mean())

# 무빙 윈도우

df = pd.DataFrame(np.random.randn(300, 4),
                  index = pd.date_range('2020-01-01', periods = 300, freq = 'D'),
                  columns = ['C1', 'C2', 'C3', 'C4'])
print(df)
print(df.rolling(30).mean())
df.rolling(30).mean().plot()
df.rolling(60).mean().plot()
df.C1.rolling(60, min_periods = 10).std().plot()
df.rolling(60, min_periods = 10).std()[10:50].plot()
df.rolling(60, min_periods = 10).std().expanding().mean().plot()
df.rolling(60).mean().plot(logy = True)
df.rolling('20D').mean().plot()
df['C1'].rolling(30, min_periods = 20).mean().plot(style = '--', label = 'Semple MA')
df['C1'].ewm(span = 30).mean().plot(style = '-', label = 'EWMA')
df['C1'].rolling(100, min_periods = 50).corr(df['C3']).plot()
df['C2'].rolling(100, min_periods = 50).corr(df['C4']).plot()
# =============================================================================

 


# 데이터 읽기 및 저장
# =============================================================================

# read_clipboard : 클립보드에 있는 데이터 읽기, 웹페이지에 있는 표를 읽어올 때 유용
# read_excel : 엑셀 파일에서 표 형식 데이터 열기
# read_html : HTML문서 내의 모든 테이블 데이터 읽기
# read_json : JSON에서 데이터 읽기
# read_sas : SAS시스템의 사용자 정의 저장 포맷 데이터 읽기
# read_sql : SQL 질의 결과를 DataFrame형식으로 읽기

# 텍스트 파일 읽기/쓰기



# read_csv : 파일URL, 객체로부터 구분된 데이터 읽기(기본구분자 : ,)
# %%writefile example1.csv
# a, b, c, d, e, text
# 1, 2, 3, 4, 5, hi
# 6, 7, 8, 9, 10, pandas
# 11, 12, 13, 14, 15, csv
# 파이참에서는 안됨
print('read_csv : 파일URL, 객체로부터 구분된 데이터 읽기(기본구분자 : ,)')
print(pd.read_csv('example1.csv'))
print()

# %%writefile example2.csv
# 1, 2, 3, 4, 5, hi
# 6, 7, 8, 9, 10, pandas
# 11, 12, 13, 14, 15, csv
print(pd.read_csv('example2.csv', header = None))
print(pd.read_csv('example2.csv', names = ['a', 'b', 'c', 'd', 'e', 'text']))
print(pd.read_csv('example2.csv', names = ['a', 'b', 'c', 'd', 'e', 'text'], index_col = 'text'))
print()

# read_table : 파일URL, 객체로부터 구분된 데이터 읽기(기본구분자 : \t)
print('read_table : 파일URL, 객체로부터 구분된 데이터 읽기(기본구분자 : \t)')
# %%writefile example3.txt
#       a     b     c
# 1   0.1   0.2   0.3
# 2   0.4   0.5   0.6
# 3   0.7   0.8   0.9
print(pd.read_table('example3.txt', sep='\s+'))

# =============================================================================
