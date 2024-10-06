import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


df = pd.read_excel('2024-09-23 Sotsiologicheskii opros.xlsx')
df.drop([
    'Как часто вы берете инициативу в свои руки? / Баллы',
    'Как часто вы пропускаете завтраки? / Баллы',
    'Какая культура ближе / Баллы',
    'Выпиваете алкоголь / Баллы',
    'Формат работы / Баллы',
    'Любимое время года? / Баллы',
    'Что пьют родители / Баллы',
    'Какие напитки любите / Баллы',
    'Набрано баллов',
    'Всего баллов',
    'Результат теста',
           ], axis=1, inplace=True)

#print(df)

#print(df.info())

outcome = 'Что вы предпочитаете?'
predictors = list(df.columns)
predictors.remove(outcome)

label_encoders = {}
for column in predictors:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

scaler = StandardScaler()
df[predictors] = scaler.fit_transform(df[predictors])

x = df[predictors]
y = df[outcome]

print('-' * 65)
for k in range(3, 12, 2):
  kNN = KNeighborsClassifier(n_neighbors=k)
  kNN.fit(x, y)

  print('k =', k)
  for num_string in range(5):
    new_record = df.loc[num_string:num_string, predictors]

    prediction = kNN.predict(new_record)
    probabilities = kNN.predict_proba(new_record)

    print(f'{num_string + 1}) Предсказание: {prediction[0]}; Вероятности:{probabilities[0]}')
  print('-' * 65)