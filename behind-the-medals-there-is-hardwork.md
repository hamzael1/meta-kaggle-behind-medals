
# Behind the Medals, there is ... <span style="color: red">HARDWORK</span> !


*Hamza El Bouatmani*  - March 2019

----



# Introduction
Although registered two years ago, I just started using the platform a couple of months ago, and was curious about how to become a Kaggle Expert or Master, so I decided to use the Kaggle Meta Dataset to dig into and examine some statistics related to Top Kernels and Top Kernel Authors.

I hope this Kernel will be useful.


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff
init_notebook_mode(connected=True)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


from IPython.display import display

import os
input_data = os.listdir("../input")
#print('Files: ', input_data)
```


<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script><script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window._Plotly) {require(['plotly'],function(plotly) {window._Plotly=plotly;});}</script>


# 1. Some Filtering <a id="1" />


In order to get the purest insights, we must take some time to clean the data we'll be working with. 
Not all Kernels in the dataset are relevant to us. We filter the kernels as follows:
- **Exclude Kernels with no UrlSlug**
- **Exclude Kernels whose authors are members of the Kaggle Team (we're interested in normal users and how they get medals)**
- **Exclude Kernels which are solutions to exercises (from Kaggle Learn section)**
- **Exclude Kernels where CreationDate & MadePublicDate are BOTH NULL ( we don't have any idea when they were created, maybe it's a Data Collection problem )**


```python
kernels_df = pd.read_csv('../input/Kernels.csv')
users_df = pd.read_csv('../input/Users.csv')

init_nbr_kernels = len(kernels_df)
#print('Number of Kernels Before Filtering: ', init_nbr_kernels)
#print('Filtering ... ')

# Parse Date Columns
kernels_df['MadePublicDate'] = pd.to_datetime(kernels_df['MadePublicDate'])
kernels_df['CreationDate'] = pd.to_datetime(kernels_df['CreationDate'])
kernels_df['MedalAwardDate'] = pd.to_datetime(kernels_df['MedalAwardDate'])

# Exclude Kernels without Url Slug ( just two for at the time of writing )
kernels_df = kernels_df[~kernels_df['CurrentUrlSlug'].isnull()]

#Extract Kernels of the Kaggle Team
kaggle_team_users = users_df[users_df['PerformanceTier'] == 5]
kaggle_team_kernels = kernels_df[kernels_df['AuthorUserId'].isin(kaggle_team_users['Id'])]

# Exclude Kernels from the Kaggle Team
normal_user_kernels = kernels_df[~kernels_df['AuthorUserId'].isin(kaggle_team_users['Id'])]

# Exclude Exercise Kernels which are Forked (most if not all are just forks of exercises to learn a topic from the section Kaggle Learn, they're not relevant to us)
normal_user_kernels = normal_user_kernels[ ~((normal_user_kernels['CurrentUrlSlug'].str.contains('exercise')) & (~normal_user_kernels['ForkParentKernelVersionId'].isnull()) ) ]
assert len(normal_user_kernels[ (~normal_user_kernels['ForkParentKernelVersionId'].isnull()) & (normal_user_kernels['CurrentUrlSlug'].str.contains('exercise'))]) == 0


# Rows that have MadePublicDate NULL and CreationDate NOT NULL: we replace the NULL in MadePublicDate with the value of CreationDate
for i, r in normal_user_kernels.iterrows():
    if pd.isnull(r['MadePublicDate']) and not pd.isnull(r['CreationDate']):
        normal_user_kernels.at[i, 'MadePublicDate'] = r['CreationDate']
normal_user_kernels[(~normal_user_kernels['CreationDate'].isnull()) & (normal_user_kernels['MadePublicDate'].isnull())]

# Drop all other rows which have both dates NULL ( ~ 190 at time of writing )
normal_user_kernels = normal_user_kernels.drop(normal_user_kernels[normal_user_kernels['MadePublicDate'].isnull()].index)

# Join Kernels with Authors ( to have the author name on the same row to make things easy)
normal_user_kernels = normal_user_kernels.join(users_df.set_index('Id'), on='AuthorUserId')

# Replace NaN in 'Medal' with Zero
normal_user_kernels['Medal'] = normal_user_kernels['Medal'].fillna(0)

oldest_date = normal_user_kernels['MadePublicDate'].min()
newest_date = normal_user_kernels['MadePublicDate'].max()
range_dates_str = f"{oldest_date.strftime('%b %Y')} ~ {newest_date.strftime('%b %Y')}"

after_nbr_kernels = len(normal_user_kernels)
print('Number of Kernels after Filtering (', range_dates_str , '): ', after_nbr_kernels, f'({init_nbr_kernels - after_nbr_kernels} Kernels filtered)')
```

    Number of Kernels after Filtering ( Mar 2015 ~ Feb 2019 ):  206962 (17317 Kernels filtered)


# 2. Medals are *<span style="color: #dca917">RARE</span>* ! <a id="2" />

Most of us, beginners in Kaggle, might look at the Top Kernels and Top authors and think that getting Medals is easy. However, a quick look into the Meta Kaggle data shows that Kernels awarded with Medals are ***extremely RARE*** compared to the number of kernels without medals.

**(You can hover to see the actual number values)**


```python
# Pie Chats for Total Number of Kernels

PLOT_BG_COLOR = '#f1f1f1'

kernel_colors = ['#FF9999', '#66B3FF'] # ['awarded', 'not awarded']
medal_colors  = ['#ffd448', '#e9e9e9', '#f0ba7c' ] # ['Gold', 'Silver', 'Bronze']

medal_kernels = normal_user_kernels[normal_user_kernels['Medal'] > 0]

# First Pie Chart: Awarded vs Not Awarded Kernels
vals = []
vals.append(len(medal_kernels))
vals.append(len(normal_user_kernels[normal_user_kernels['Medal'] == 0]))
chart1 = {
            'type': 'pie',
            'title': 'Awarded vs Not Awarded',
            'titlefont': {'size': 16},
            'labels': ['Kernels Awarded', 'Kernels Not Awarded'],
            'values': vals,
            'hoverinfo': 'label+value',
            'textinfo': 'percent',
            'textposition': 'inside',
            'textfont': {'size': 12},
            'marker': {'colors': kernel_colors, 'line': {'color': 'white', 'width': 2,}},
            'domain': {'x': [0, 0.4], 'y': [0, 1]}
        }

# Second Pie Chart: Gold, Silver & Bronze Kernels
vals = []
vals.append(len(medal_kernels[medal_kernels['Medal'] == 1]))
vals.append(len(medal_kernels[medal_kernels['Medal'] == 2]))
vals.append(len(medal_kernels[medal_kernels['Medal'] == 3]))



chart2 = {
            'type': 'pie',
            'title': 'Gold, Silver & Bronze',
            'titlefont': {'size': 16},
            'showlegend': False,
            'labels': ['Gold', 'Silver', 'Bronze'],
            'values': vals,
            'hoverinfo': 'label+value',
            'textinfo': 'percent+label',
            'textfont': {'size': 12},
            'marker': {'colors': medal_colors, 'line': {'color': 'white', 'width': .5,}},
            'domain': {'x': [0.6, 1], 'y': [0,1]}
        }

fig = {
    'data': [ chart1, chart2 ],
    'layout': {
        'height': 500,
        'title': {
            'text': f'Total number of Public Kernels ({range_dates_str})',
            'font': {'size': 18}
        }, 'legend': {
            'orientation': 'h'
        }
    }
}

iplot(fig)
```


<div id="6af4ff4d-a7d6-48ae-ab94-87c48e545dc2" style="height: 500px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("6af4ff4d-a7d6-48ae-ab94-87c48e545dc2", [{"domain": {"x": [0, 0.4], "y": [0, 1]}, "hoverinfo": "label+value", "labels": ["Kernels Awarded", "Kernels Not Awarded"], "marker": {"colors": ["#FF9999", "#66B3FF"], "line": {"color": "white", "width": 2}}, "textfont": {"size": 12}, "textinfo": "percent", "textposition": "inside", "title": {"text": "Awarded vs Not Awarded", "font": {"size": 16}}, "values": [8691, 198271], "type": "pie", "uid": "7280ea0e-9ac0-4df4-89b2-7133e8f8c3e6"}, {"domain": {"x": [0.6, 1], "y": [0, 1]}, "hoverinfo": "label+value", "labels": ["Gold", "Silver", "Bronze"], "marker": {"colors": ["#ffd448", "#e9e9e9", "#f0ba7c"], "line": {"color": "white", "width": 0.5}}, "showlegend": false, "textfont": {"size": 12}, "textinfo": "percent+label", "title": {"text": "Gold, Silver & Bronze", "font": {"size": 16}}, "values": [745, 1571, 6375], "type": "pie", "uid": "181bbfd0-9c29-423b-bdb1-f91234514deb"}], {"height": 500, "legend": {"orientation": "h"}, "title": {"font": {"size": 18}, "text": "Total number of Public Kernels (Mar 2015 ~ Feb 2019)"}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("6af4ff4d-a7d6-48ae-ab94-87c48e545dc2"));});</script>



```python
years = list(normal_user_kernels.MadePublicDate.dt.year.unique())
years.remove(2019) # Remove this year because it has just started
years.sort()
nbr_all_vals = []
nbr_awarded_vals = []
nbr_not_awarded_vals = []
nbr_gold_vals = []
nbr_silver_vals = []
nbr_bronze_vals = []

for y in years:
    if not np.isnan(y):
        years_kernels = normal_user_kernels[normal_user_kernels['MadePublicDate'].dt.year == y]
        nbr_all_vals.append(len(years_kernels))
        golds = len(years_kernels[years_kernels['Medal'] == 1])
        silvers = len(years_kernels[years_kernels['Medal'] == 2])
        bronzes = len(years_kernels[years_kernels['Medal'] == 3])
        nbr_awarded_vals.append(golds+silvers+bronzes)
        nbr_not_awarded_vals.append(nbr_all_vals[-1] - nbr_awarded_vals[-1])
        nbr_gold_vals.append(golds)
        nbr_silver_vals.append(silvers)
        nbr_bronze_vals.append(bronzes)
#print(years, nbr_all_vals, nbr_awarded_vals, nbr_gold_vals, nbr_silver_vals, nbr_bronze_vals)
```


```python
# Bar Chart for Number of Kernels per year

fig = { 'data': [
        {   'type': 'bar',
            'name': 'Kernels w/o Awards',
            'x': years,
            'y': nbr_not_awarded_vals,
            'marker': {'color': kernel_colors[1] },
            'xaxis': 'x1',
            'yaxis': 'y1'
        },
        {   'type': 'bar',
            'name': 'Awarded Kernels',
            'x': years,
            'y': nbr_awarded_vals,
            'marker': {'color': kernel_colors[0] },
            #'line': {'color': '#ffd448' },
            'xaxis': 'x1',
            'yaxis': 'y1'
        }
], 'layout': {
        'plot_bgcolor': PLOT_BG_COLOR,
        'height': 600,
        'title': 'Change of Number of Kernels per year',
        'legend': {'orientation': 'h'},
        'xaxis': {'dtick': 1},
        'yaxis': {'dtick': 5000, 'title': 'Number of Kernels'}
    }}
iplot(fig)
```


<div id="a2218066-d6ef-48c6-b1ed-d9b370f317aa" style="height: 600px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("a2218066-d6ef-48c6-b1ed-d9b370f317aa", [{"marker": {"color": "#66B3FF"}, "name": "Kernels w/o Awards", "x": [2015, 2016, 2017, 2018], "xaxis": "x", "y": [11746, 68457, 65515, 43726], "yaxis": "y", "type": "bar", "uid": "18998150-0bc4-402e-b351-f8d6686d28d1"}, {"marker": {"color": "#FF9999"}, "name": "Awarded Kernels", "x": [2015, 2016, 2017, 2018], "xaxis": "x", "y": [392, 1168, 2547, 4125], "yaxis": "y", "type": "bar", "uid": "4bbfb221-a51f-4d08-99be-4637c0f2a063"}], {"height": 600, "legend": {"orientation": "h"}, "plot_bgcolor": "#f1f1f1", "title": {"text": "Change of Number of Kernels per year"}, "xaxis": {"dtick": 1}, "yaxis": {"dtick": 5000, "title": {"text": "Number of Kernels"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("a2218066-d6ef-48c6-b1ed-d9b370f317aa"));});</script>



```python
# Bar & Line Chart for Awarded Kernels per Year by Medal

fig = { 'data': [
        {
            'type': 'bar',
            'name': 'Gold',
            'x': years,
            'y': nbr_gold_vals,
            'marker': {'color': medal_colors[0]},
        },
        {
            'type': 'bar',
            'name': 'Silver',
            'x': years,
            'y': nbr_silver_vals,
            'marker': {'color': medal_colors[1]},
        },
        {
            'type': 'bar',
            'name': 'Bronze',
            'x': years,
            'y': nbr_bronze_vals,
            'marker': {'color': medal_colors[2]},
        },
        {   'type': 'scatter',
            'name': 'Number of Awarded Kernels',
            'x': years,
            'y': nbr_awarded_vals,
            'line': {'color': kernel_colors[0] },
        },
    ], 'layout': {
        'title': 'Change of Number of Awarded Kernel per year',
        'legend': {'orientation': 'h'},
        'height': 600,
        'plot_bgcolor': PLOT_BG_COLOR,
        'xaxis': {'dtick': 1},
        'yaxis': {'dtick': 250, 'title': 'Number of Kernels'}
        
    }
}
iplot(fig)
```


<div id="cfbed344-c167-412b-9094-5b6ca84286c9" style="height: 600px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("cfbed344-c167-412b-9094-5b6ca84286c9", [{"marker": {"color": "#ffd448"}, "name": "Gold", "x": [2015, 2016, 2017, 2018], "y": [15, 67, 223, 399], "type": "bar", "uid": "a1498fb3-31da-47ab-912d-3658faae6371"}, {"marker": {"color": "#e9e9e9"}, "name": "Silver", "x": [2015, 2016, 2017, 2018], "y": [66, 175, 440, 815], "type": "bar", "uid": "d407b50b-069a-4a56-98a8-44af0f40cd9f"}, {"marker": {"color": "#f0ba7c"}, "name": "Bronze", "x": [2015, 2016, 2017, 2018], "y": [311, 926, 1884, 2911], "type": "bar", "uid": "16a61f38-6404-4d78-b6f8-dc8b1608d8f5"}, {"line": {"color": "#FF9999"}, "name": "Number of Awarded Kernels", "x": [2015, 2016, 2017, 2018], "y": [392, 1168, 2547, 4125], "type": "scatter", "uid": "629e1f35-1c7f-41f9-b285-c6f0a866856d"}], {"height": 600, "legend": {"orientation": "h"}, "plot_bgcolor": "#f1f1f1", "title": {"text": "Change of Number of Awarded Kernel per year"}, "xaxis": {"dtick": 1}, "yaxis": {"dtick": 250, "title": {"text": "Number of Kernels"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("cfbed344-c167-412b-9094-5b6ca84286c9"));});</script>


### **<span style="color: red">Observation</span>**: Although the number of Kernels not awarded ***is much greater*** than the number of Kernels awarded, we can note a ***steady increase*** in the number of Awarded Kernels each year

# 3. Who publishes Top Kernels ? <a id="3" />

Let's now have a quick look at some of the Top Authors in Kaggle


```python
nbr_medals_per_author = pd.crosstab( [normal_user_kernels['AuthorUserId'], normal_user_kernels['DisplayName']], normal_user_kernels['Medal'])
nbr_medals_per_author = nbr_medals_per_author.rename(columns={0: 'NotAwarded', 1.0: 'Gold', 2.0: 'Silver', 3.0: 'Bronze'})
nbr_medals_per_author['Awarded'] = nbr_medals_per_author['Gold'] + nbr_medals_per_author['Silver'] + nbr_medals_per_author['Bronze']
nbr_medals_per_author = nbr_medals_per_author.sort_values(by='Awarded', ascending=False)
nbr_medals_per_author = nbr_medals_per_author.reset_index(level=1) # Make DisplayName a column

n = 30
top = nbr_medals_per_author[:n]


fig = {
    'data': [
        {
            'type': 'bar',
            'y': top['Bronze'].values,
            'x': top['DisplayName'].values,
            'name': 'Bronze',
            'marker': {'color': medal_colors[2]}
        }, {
            'type': 'bar',
            'y': top['Silver'].values,
            'x': top['DisplayName'].values,
            'name': 'Silver',
            'marker': {'color': medal_colors[1]}
        }, {
            'type': 'bar',
            'y': top['Gold'].values,
            'x': top['DisplayName'].values,
            'name': 'Gold',
            'marker': {'color': medal_colors[0]}
        }
    ], 'layout': {
        'title': f'Top {n} Kernel Authors ({range_dates_str})',
        'barmode': 'stack',
        'yaxis': {'title': 'Number of Awarded Kernels'},
        'legend': {'x': 0.92, 'y': 1},
        'margin': {'r': 0},
        #'plot_bgcolor': PLOT_BG_COLOR,
    }
}

iplot(fig)
```


<div id="4ed8b8ca-cac2-4f1a-899c-fe33f860fa24" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("4ed8b8ca-cac2-4f1a-899c-fe33f860fa24", [{"marker": {"color": "#f0ba7c"}, "name": "Bronze", "x": ["ASHISH PATEL", "Scirpus", "Bojan Tunguz", "Jonathan Bouchet", "Andy Harless", "Kevin Mader", "SRK", "Bukun", "Aleksey Bilogur", "Gabriel Preda", "Shivam Bansal", "beluga", "DSangeetha", "olivier", "Umesh", "ZFTurbo", "anokas", "Andrew Lukyanenko", "Nick Brooks", "Omri Goldstein", "Pranav Pandya", "amrrs", "JohnM", "Tilii", "DATAI", "Paulo Pinto", "Troy Walters", "YouHan Lee", "Anisotropic", "Shujian L."], "y": [53, 55, 40, 59, 44, 40, 23, 33, 24, 20, 12, 15, 31, 12, 26, 11, 13, 9, 20, 27, 17, 17, 14, 13, 7, 12, 13, 16, 5, 12], "type": "bar", "uid": "8c74a24f-2a76-4012-96e9-34da362d6cc4"}, {"marker": {"color": "#e9e9e9"}, "name": "Silver", "x": ["ASHISH PATEL", "Scirpus", "Bojan Tunguz", "Jonathan Bouchet", "Andy Harless", "Kevin Mader", "SRK", "Bukun", "Aleksey Bilogur", "Gabriel Preda", "Shivam Bansal", "beluga", "DSangeetha", "olivier", "Umesh", "ZFTurbo", "anokas", "Andrew Lukyanenko", "Nick Brooks", "Omri Goldstein", "Pranav Pandya", "amrrs", "JohnM", "Tilii", "DATAI", "Paulo Pinto", "Troy Walters", "YouHan Lee", "Anisotropic", "Shujian L."], "y": [15, 10, 22, 10, 16, 11, 4, 12, 13, 14, 10, 13, 7, 7, 6, 14, 7, 6, 8, 1, 9, 10, 9, 7, 10, 10, 9, 6, 3, 6], "type": "bar", "uid": "2756ffc3-2200-44ff-b9b2-e2705135388d"}, {"marker": {"color": "#ffd448"}, "name": "Gold", "x": ["ASHISH PATEL", "Scirpus", "Bojan Tunguz", "Jonathan Bouchet", "Andy Harless", "Kevin Mader", "SRK", "Bukun", "Aleksey Bilogur", "Gabriel Preda", "Shivam Bansal", "beluga", "DSangeetha", "olivier", "Umesh", "ZFTurbo", "anokas", "Andrew Lukyanenko", "Nick Brooks", "Omri Goldstein", "Pranav Pandya", "amrrs", "JohnM", "Tilii", "DATAI", "Paulo Pinto", "Troy Walters", "YouHan Lee", "Anisotropic", "Shujian L."], "y": [4, 6, 9, 0, 3, 8, 24, 2, 7, 9, 20, 10, 0, 16, 0, 6, 11, 15, 2, 1, 3, 2, 4, 7, 9, 3, 3, 3, 16, 5], "type": "bar", "uid": "bc096ca4-caae-423f-9d40-3ad742b6880b"}], {"barmode": "stack", "legend": {"x": 0.92, "y": 1}, "margin": {"r": 0}, "title": {"text": "Top 30 Kernel Authors (Mar 2015 ~ Feb 2019)"}, "yaxis": {"title": {"text": "Number of Awarded Kernels"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("4ed8b8ca-cac2-4f1a-899c-fe33f860fa24"));});</script>


### **<span style="color: red">Impressive !</span> **   [Mr Ashish Patel](https://www.kaggle.com/ashishpatel26) comes in top with 72 Awarded Kernels, but there is a lot of competition between the Top 4.

*(Note: There are many other great Kernel Authors not figuring in the list. You can fork this kernel and play with the n parameter in the code, to get the Top n Authors in terms of number of awarded Kernels)*

# 4. It's not always **<span style="color: #dca917">Gold</span>** ! <a id="4" />

After seeing the last graph, one might think that those authors are ***Super Humans*** ! However, the **DATA** says that it's the product of **<span style="color: red">HARDWORK</span>** and **<span style="color: red">PASSION</span>**!

The following graph illustrates the proportion of Awarded & Not Awarded Kernels for each of the Top Authors :


```python

fig = {
    'data': [
        {
            'type': 'bar',
            'y': top['NotAwarded'].values,
            'x': top['DisplayName'].values,
            'name': 'Not Awarded Kernels',
            'marker': {'color': kernel_colors[1]}
        },{
            'type': 'bar',
            'y': top['Awarded'].values,
            'x': top['DisplayName'].values,
            'name': 'Awarded Kernels',
            'marker': {'color': kernel_colors[0]}
        }
    ], 'layout': {
        'title': f'Top {n} Kernel Authors ({range_dates_str})',
        'barmode': 'stack',
        'yaxis': {'title': 'Number of Kernels'},
        'legend': {'x': 0.78, 'y': 1},
        'plot_bgcolor': PLOT_BG_COLOR,
        'margin': {'r': 0}
    }
}

iplot(fig)
```


<div id="47f17b16-620c-436d-81ca-bbc44cb42f00" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("47f17b16-620c-436d-81ca-bbc44cb42f00", [{"marker": {"color": "#66B3FF"}, "name": "Not Awarded Kernels", "x": ["ASHISH PATEL", "Scirpus", "Bojan Tunguz", "Jonathan Bouchet", "Andy Harless", "Kevin Mader", "SRK", "Bukun", "Aleksey Bilogur", "Gabriel Preda", "Shivam Bansal", "beluga", "DSangeetha", "olivier", "Umesh", "ZFTurbo", "anokas", "Andrew Lukyanenko", "Nick Brooks", "Omri Goldstein", "Pranav Pandya", "amrrs", "JohnM", "Tilii", "DATAI", "Paulo Pinto", "Troy Walters", "YouHan Lee", "Anisotropic", "Shujian L."], "y": [101, 87, 133, 100, 176, 288, 15, 42, 117, 20, 0, 11, 27, 15, 0, 27, 115, 9, 24, 14, 1, 11, 2, 4, 0, 18, 15, 8, 7, 0], "type": "bar", "uid": "8f9024a6-f0c4-4042-a18d-d013a0db0c85"}, {"marker": {"color": "#FF9999"}, "name": "Awarded Kernels", "x": ["ASHISH PATEL", "Scirpus", "Bojan Tunguz", "Jonathan Bouchet", "Andy Harless", "Kevin Mader", "SRK", "Bukun", "Aleksey Bilogur", "Gabriel Preda", "Shivam Bansal", "beluga", "DSangeetha", "olivier", "Umesh", "ZFTurbo", "anokas", "Andrew Lukyanenko", "Nick Brooks", "Omri Goldstein", "Pranav Pandya", "amrrs", "JohnM", "Tilii", "DATAI", "Paulo Pinto", "Troy Walters", "YouHan Lee", "Anisotropic", "Shujian L."], "y": [72, 71, 71, 69, 63, 59, 51, 47, 44, 43, 42, 38, 38, 35, 32, 31, 31, 30, 30, 29, 29, 29, 27, 27, 26, 25, 25, 25, 24, 23], "type": "bar", "uid": "03b7a659-50ad-4e52-b93a-2ebfa78c005c"}], {"barmode": "stack", "legend": {"x": 0.78, "y": 1}, "margin": {"r": 0}, "plot_bgcolor": "#f1f1f1", "title": {"text": "Top 30 Kernel Authors (Mar 2015 ~ Feb 2019)"}, "yaxis": {"title": {"text": "Number of Kernels"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("47f17b16-620c-436d-81ca-bbc44cb42f00"));});</script>


### **<span style="color: red">Observation</span>**: Many Top Authors wrote *several non-awarded Kernels*. This shows that their success is the product of *<span style="color: red">HARDWORK</span>* ! It's not always *<span style="color: #dca917">GOLD</span>* !

*(Note: It is possible that an author deletes some of his previously published Kernels. The graph is based on the Meta Kaggle dataset which only shows currently public Kernels )*

# 5. You <span style="color: red">CAN</span> do it too ! <a id="5" />

### One might also think that these Top Authors are the ones who publish the most in Kaggle. However, the **DATA** says something else !

The following graph shows the how many kernels were published by each Author Category (Preformance Tier):


```python
nbr_awarded_kernels_per_tier = medal_kernels['PerformanceTier'].value_counts()
nbr_awarded_kernels_per_tier = nbr_awarded_kernels_per_tier.sort_index()
#nbr_awarded_kernels_per_tier

tier_colors = ['#5AC995', '#00BBFF', '#976591', '#F96517', '#DCA917']

fig = {
    'data': [{
        'type': 'pie',
        'labels': ['Novices', 'Contributors', 'Experts', 'Masters', 'Grandmasters'],
        'values': nbr_awarded_kernels_per_tier,
        'hole': .3,
        'textinfo': 'percent+label',
        'marker': {'colors': tier_colors}
    }],
    'layout': {
        'title': f'Number of Awarded Kernels by Author Tier',
        'showlegend': False
    }
}
iplot(fig)
```


<div id="5ce14e36-9602-438a-b668-af855acc5b3f" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("5ce14e36-9602-438a-b668-af855acc5b3f", [{"hole": 0.3, "labels": ["Novices", "Contributors", "Experts", "Masters", "Grandmasters"], "marker": {"colors": ["#5AC995", "#00BBFF", "#976591", "#F96517", "#DCA917"]}, "textinfo": "percent+label", "values": [1152, 2312, 2944, 1533, 676], "type": "pie", "uid": "05e65857-b532-4466-a349-6fa234e5ac1f"}], {"showlegend": false, "title": {"text": "Number of Awarded Kernels by Author Tier"}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("5ce14e36-9602-438a-b668-af855acc5b3f"));});</script>


### **<span style="color: red">Observation</span>**: <span style="color: #976591"> Experts</span> produce the most number of Kernels ( probably because they want to become Masters). Then followed by <span style="color: #00BBFF">Contributors</span> ! After them come <span style="color: #F96517">Masters</span> and then <span style="color: #5AC995">Novices</span> ! Lastly we have <span style="color: #DCA917">Grandmasters</span> producing ~7.8% of the Awarded Kernels (probably because there only according to the [first chart](#2) only 8.5% of the authors have reached that level)

# 6. When did the Top Authors start publishing ?<a id="6" />

Another interesting thing to examine, is when did the Top authors start publishing their first Kernels. In other words, how much time did it take them to reach this level.


```python
# Date of first kernel
n = 50
top = nbr_medals_per_author.head(n)
oldest_kernel_dates = []
for userid in top.index:
    oldest_kernel_date = normal_user_kernels[normal_user_kernels['AuthorUserId'] == userid]['MadePublicDate'].min()
    oldest_kernel_dates.append(oldest_kernel_date)
oldest_kernel_dates = pd.Series(oldest_kernel_dates, index=top.index).sort_values()

# Date of first kernel

#fig = ff.create_distplot(s, [f'Top {n} authors'])


fig= {
    'data': [
        {
            'type': 'histogram',
            #'bin_size': .4,
            'marker': {"color": 'red', 'line': {'color': 'white', 'width': 2}}, 
            'x': oldest_kernel_dates,
            "opacity": 0.5, 
        }
    ],
    'layout': {
        'title': f'Dates of First Published Kernel (top {n} Authors)',
        'xaxis': {'title': 'Date of First Published Kernel'},
        'yaxis': {'title': 'Number of Authors'}
    }
}

iplot(fig)
```


<div id="f9797aad-df07-402b-8dc7-7a9365397810" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("f9797aad-df07-402b-8dc7-7a9365397810", [{"marker": {"color": "red", "line": {"color": "white", "width": 2}}, "opacity": 0.5, "x": ["2015-04-23", "2015-06-06", "2015-06-26", "2015-07-17", "2015-08-14", "2015-09-24", "2015-09-30", "2015-10-07", "2015-10-26", "2016-01-07", "2016-01-30", "2016-02-05", "2016-02-19", "2016-03-07", "2016-03-08", "2016-03-14", "2016-04-15", "2016-05-13", "2016-07-02", "2016-07-12", "2016-07-28", "2016-08-22", "2016-09-01", "2016-10-15", "2016-10-18", "2016-10-19", "2016-11-01", "2016-11-06", "2016-11-30", "2016-12-10", "2016-12-16", "2017-01-04", "2017-01-21", "2017-02-27", "2017-04-11", "2017-04-15", "2017-04-22", "2017-07-11", "2017-07-15", "2017-07-17", "2017-07-19", "2017-10-07", "2017-10-25", "2017-11-27", "2018-01-09", "2018-02-09", "2018-02-14", "2018-03-21", "2018-03-22", "2018-11-10"], "type": "histogram", "uid": "3a233ab8-3cf7-4aa7-b134-9e3cf0a086f4"}], {"title": {"text": "Dates of First Published Kernel (top 50 Authors)"}, "xaxis": {"title": {"text": "Date of First Published Kernel"}}, "yaxis": {"title": {"text": "Number of Authors"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("f9797aad-df07-402b-8dc7-7a9365397810"));});</script>


### **<span style="color: red">Observation</span>**:  **More than 50%** of the Top Authors started before January 2017 ( **more than one year ago** )

(*Note: This only shows how much experience the top authors have on Kaggle, we're not taking into account their Data Science experience prior to Kaggle*)

# 7. Tags in Top Kernels <a id="7" />

Lastly, we examine the most used tags in the awarded Kernels.


```python
tags_df = pd.read_csv('../input/Tags.csv')
normal_user_kernel_tags = pd.read_csv('../input/KernelTags.csv')
normal_user_kernel_tags = normal_user_kernel_tags[normal_user_kernel_tags['KernelId'].isin(normal_user_kernels['Id'])]
medal_kernels_tag_ids = normal_user_kernel_tags[normal_user_kernel_tags['KernelId'].isin(medal_kernels['Id'])]['TagId']
tags_df = tags_df.set_index('Id')
tags_dic = tags_df[['Slug', 'Name']].to_dict('index')
slugs = []
for tid in medal_kernels_tag_ids:
    slugs.append(tags_dic[tid]['Slug'])

wc=WordCloud(width=800, height=400).generate(' '.join(slugs))
plt.clf()
plt.figure( figsize=(16,9) )
plt.title('Most used Tags in Awarded Kernels', fontsize=20)
plt.imshow(wc)
plt.axis('off')
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](output_31_1.png)



```python
# Bar chart for Top used Tags in Awarded Kernels

n = 30
slug_counts = pd.Series(slugs).value_counts().head(n)

fig = {
    'data': [
        {
            'type': 'bar',
            #'orientation': 'h',
            'y': slug_counts,
            'x': slug_counts.index,
            'marker': {'color': slug_counts, 'colorscale': 'Viridis', 'showscale': True}
        }
    ], 'layout': {
        'title': f'Top {n} most used Tags in Awarded Kernels',
        'yaxis': {'title': 'Number of times used'}
    }
}
iplot(fig)
```


<div id="555d6469-3d76-4b02-9766-9df7f0a20cb5" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("555d6469-3d76-4b02-9766-9df7f0a20cb5", [{"marker": {"color": [1526, 1245, 1054, 741, 506, 422, 363, 286, 272, 193, 133, 132, 121, 121, 95, 90, 90, 83, 79, 79, 77, 76, 75, 75, 74, 73, 66, 66, 63, 61], "colorscale": "Viridis", "showscale": true}, "x": ["data-visualization", "beginner", "eda", "tutorial", "feature-engineering", "classification", "starter-code", "deep-learning", "data-cleaning", "nlp", "neural-networks", "xgboost", "finance", "cnn", "time-series", "ensembling", "gradient-boosting", "image-processing", "text-mining", "food-and-drink", "random-forest", "text-data", "video-games", "geospatial-analysis", "crime", "regression-analysis", "logistic-regression", "survey-analysis", "healthcare", "internet"], "y": [1526, 1245, 1054, 741, 506, 422, 363, 286, 272, 193, 133, 132, 121, 121, 95, 90, 90, 83, 79, 79, 77, 76, 75, 75, 74, 73, 66, 66, 63, 61], "type": "bar", "uid": "0340f5e0-1fe8-46e4-aff6-78b5481a24ce"}], {"title": {"text": "Top 30 most used Tags in Awarded Kernels"}, "yaxis": {"title": {"text": "Number of times used"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("555d6469-3d76-4b02-9766-9df7f0a20cb5"));});</script>




### **<span style="color: red">Observation</span>**:  It seems that top Kernels are directed towards **Beginners**, and are centered around **Exploratory Data Analysis** and **Visualization**

# 8. Time for some Correlation ! <a id="8" />


```python
# Add MonthsOfExperience Field
#nbr_medals_per_author.loc[859104]
oldest_kernel_dates = []
for userid in nbr_medals_per_author.index:
    oldest_kernel_date = normal_user_kernels[normal_user_kernels['AuthorUserId'] == userid]['MadePublicDate'].min()
    oldest_kernel_dates.append(oldest_kernel_date)

oldest_kernel_dates = pd.Series(oldest_kernel_dates, index=nbr_medals_per_author.index)
oldest_kernel_dates = ((pd.Timestamp.now() - oldest_kernel_dates)/ np.timedelta64(1,'M')).astype('int')
nbr_medals_per_author['MonthsOfExperience'] = oldest_kernel_dates
#nbr_medals_per_author.head()

# Add NumberPublishedKernels Field
nbr_medals_per_author['NbrPublishedKernels'] = nbr_medals_per_author['Awarded'] + nbr_medals_per_author['NotAwarded']
nbr_medals_per_author.head()

top_tags = tags_df[tags_df['Slug'].isin(slug_counts.index)].index
#top_tags

nbr_medals_per_author['NbrTopTagsUsed'] = 0
for userid in nbr_medals_per_author.index:
    kernel_ids = normal_user_kernels[normal_user_kernels['AuthorUserId'] == userid]['Id']
    tag_ids = normal_user_kernel_tags[normal_user_kernel_tags['KernelId'].isin(kernel_ids)]['TagId']
    relevant_tag_ids = [tid for tid in tag_ids if tid in top_tags]
    nbr_medals_per_author.at[userid, 'NbrTopTagsUsed'] = len(relevant_tag_ids)
#nbr_medals_per_author['NbrTopTagsUsed']

df = nbr_medals_per_author.drop(['Gold', 'Silver', 'Bronze', 'NotAwarded', 'DisplayName'], axis=1)

corr = df.corr()
fig = {
    'data': [
        {
            'type': 'heatmap',
            'z': corr,
            'x': df.columns,
            'y': df.columns,
            'colorscale': 'Reds'
        }
    ],
    'layout': {
        'title': 'Correlation Heatmap',
        'margin':{
            'l': 140
        }
    }
}

iplot(fig)
```


<div id="ddc320f6-673f-4e61-bff7-04d6b02134ac" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("ddc320f6-673f-4e61-bff7-04d6b02134ac", [{"colorscale": "Reds", "x": ["Awarded", "MonthsOfExperience", "NbrPublishedKernels", "NbrTopTagsUsed"], "y": ["Awarded", "MonthsOfExperience", "NbrPublishedKernels", "NbrTopTagsUsed"], "z": [[1.0, 0.0010957868742481863, 0.5094347584054578, 0.5725698107593068], [0.0010957868742481863, 1.0, 0.07733015716123556, -0.12649543502806443], [0.5094347584054578, 0.07733015716123556, 1.0, 0.3243154172083067], [0.5725698107593068, -0.12649543502806443, 0.3243154172083067, 1.0]], "type": "heatmap", "uid": "f86ee19c-e860-4fe1-a3c9-9246b6056764"}], {"margin": {"l": 140}, "title": {"text": "Correlation Heatmap"}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("ddc320f6-673f-4e61-bff7-04d6b02134ac"));});</script>


### **<span style="color: red">Observation</span>**:  We can see that Number of Published Kernels & Number of Top Tags Used are positively correlated with Number of Awarded Kernels

# Conclusions:

* There are of course many criteria other than the ones mentioned for a Kernel to be successful, for example: It must be written and formatted nicely, must add value to the reader and be interesting.

* Many Awarded Kernels are Tutorials and Visualizations

* In Data Science as in Every Discipline, Success comes with two principal ingredients: **<span style="color: red">PASSION & HARDWORK</span>**

## References:

* [Kaggle Progression System](https://www.kaggle.com/progression) (Page)
* [Kaggle Trends](https://www.kaggle.com/gaborfodor/kaggle-trends) (Kernel)
* [How to get upvotes in Kaggle](https://www.kaggle.com/aleksandradeis/how-to-get-upvotes-for-a-kernel-on-kaggle) (Kernel)
