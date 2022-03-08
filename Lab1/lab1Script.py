import pandas as pd
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel
from random import shuffle

df = pd.read_csv("Data1.csv")
df = df.set_index('Unnamed: 0')
df.plot()
plt.title("Exercise 1.3")
plt.figure(1)
df.hist()
plt.suptitle("Exercise 1.4")
plt.figure(2)
df.plot.kde()
plt.title("Exercise 1.5")
plt.figure(3)

df2 = df.loc[pd.to_datetime(df.index).year == 2018, ["theta_1", "theta_2", "theta_3", "theta_4"]]
df2.plot()
plt.title("Exercise 1.6.1")
plt.figure(4)
df2.hist()
plt.suptitle("Exercise 1.6.2")
plt.figure(5)
df2.plot.kde()
plt.title("Exercise 1.6.3")
plt.figure(6)

F = 7
L = 6
y = [0] * F + [1] * L
shuffle(y)
data = {'N': F + L,
             'y': y}
model = CmdStanModel(stan_file='bern_1.stan')
sample = model.sample(data)
theta = sample.stan_variable('theta')
summary = sample.summary()
plt.figure(7)
plt.hist(theta, bins=20)
plt.title("Exercise 2")
plt.axvline(summary['5%']['theta'], color='r')
plt.axvline(summary['95%']['theta'], color='g')
plt.axvline(summary['50%']['theta'], color='b')
plt.axvline(summary['Mean']['theta'], color='y')
plt.show()