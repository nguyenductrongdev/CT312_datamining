import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Pie chart
labels = ['Cammeo', 'Osmancik']
sizes = [1630, 2180]
# only "explode" the 2nd slice (i.e. 'Hogs')
# explode = (0, 0.1, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.show()

df = pd.read_csv('Rice_Osmancik_Cammeo_Dataset.csv')
for col in df.columns:
    sns.distplot(df[col])
    plt.show()
