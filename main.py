import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
from scipy.stats import ttest_ind,mannwhitneyu
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 350)
df = pd.read_csv('.venv/toolwindow_data.csv')
df.head(30)
df.count()

#sorting by user id and timestamps to see whole open-close log for every usear
df = df.sort_values(by=['user_id','timestamp'])
df = df.reset_index(drop=True)
df.head(30)

# events opened without closing will be droped so as closed windows without being open
# looking for open-close pairs and calculating a open time. results are save in dataframe

results = []

# iterating thrtough users
for user, group in df.groupby("user_id"):
    group = group.reset_index(drop=True)
    open_event = None

    # no index needed:
    for _, row in group.iterrows():
        if row["event"] == "opened":
            open_event = row

        elif row["event"] == "closed" and open_event is not None:  # open close pair was found

            open_time = row["timestamp"] - open_event["timestamp"]

            # saving
            results.append({
                "user_id": user,
                "open_type": open_event["open_type"],
                "time": open_time
            })
            # reset:
            open_event = None

# result dataframe:
df_pairs = pd.DataFrame(results)
df_pairs.head(15)

#checking if difference between amout of data is significant:
df_pairs["open_type"].value_counts()
#there is 378 less observations in one of feature so it's 37,8% difference.


#histogram showing a distrbution of time for each open type
bins = 15

for otype in df_pairs["open_type"].unique():
    subset = df_pairs[df_pairs["open_type"] == otype]
    plt.hist(subset["time"], bins=bins, alpha=0.7, label=otype)

plt.xlabel("Time Opened [ms]")
plt.ylabel("Count")
plt.title("Distribution of Time Opened by Open Type")
plt.legend(title="Open Type")
plt.show()



#histogram does not give much inside so:
df_pairs.groupby("open_type")["time"].describe()
#maximal time in auto is more than twice as long as max value in manual opening

#making more time cloumns in different units:
df_pairs["time_min"] = df_pairs["time"]/(1000*60)
df_pairs["time_sec"] = df_pairs["time"]/(1000)

summary = df_pairs.groupby("open_type")["time_min"].describe()
print("summary:\n",summary)
#From summary:
#auto: mean = 105 min, max = 6831 min -> extreamly skewed
#auto: mean = 24.5 min, max = 3015 min ->  skewed

#boxplot to check
sns.boxplot(x="open_type", y="time_min", data=df_pairs)
plt.ylabel("Time Opened [minutes]")
plt.title("Tool Window Open Time by Open Type")
plt.show()

#statystical sygnificance testing :
#because data is skewed I will  not make t-test.
#instead I decided to Log transform a durations

manual_times = df_pairs[df_pairs["open_type"] == "manual"]["time_min"]
auto_times = df_pairs[df_pairs["open_type"] == "auto"]["time_min"]

u_stat, p_value = mannwhitneyu(manual_times, auto_times, alternative="two-sided")
print("U-statistic:", u_stat)
print("p-value:", p_value)

#analysing p-value
# it is extreamaly small (1.99* 10^-63)
#it means that There is a statistically significant difference between the durations of manually and automatically opened tool windows.
#The probability that this difference happened by chance is essentially zero.
# Median vs mean shows the data is highly skewed, especially for auto opens.
# Most auto opens last longer than manual opens — some extreme cases inflate the mean.

