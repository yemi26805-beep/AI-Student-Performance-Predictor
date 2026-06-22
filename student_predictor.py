import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('student_data.csv')

print('First 5 rows:\n', df.head())

X = df[['Study_Hours', 'Attendance', 'Previous_Score', 'Assignment_Score']]
y = df['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print('\nMean Absolute Error:', mae)

print('\nEnter student details:')

study_hours = float(input('Study Hours: '))
attendance = float(input('Attendance: '))
previous_score = float(input('Previous Score: '))
assignment_score = float(input('Assignment Score: '))

new_data = [[study_hours, attendance, previous_score, assignment_score]]
prediction = model.predict(new_data)

print('\nPredicted Final Score:', round(prediction[0], 2))

plt.scatter(df['Attendance'], df['Final_Score'])
plt.xlabel('Attendance')
plt.ylabel('Final Score')
plt.title('Attendance vs Final Score')
plt.show()
