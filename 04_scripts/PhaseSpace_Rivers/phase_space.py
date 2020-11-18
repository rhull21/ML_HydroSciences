# %% Modules
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# %% Functions
def norm_it(startdate, enddate, dfin, dfname, l_back=1, toscale=True):
    """
    This function noramlizes a column of data from a pandas dataframe
    using the MinMaxScaler feature of sklearn
    to create a ~ normal distribution
    and allow for better fitting between different types of variables
    in a multivariable regression

    It also lags the data by a specified number of weeks (look_back)

    Takes:
    A start and end date (strings)
        startdate =
        enddate =
    A dataframe (pandas)
        dfin =
    The name of a single column of data to normalize (string)
        dfname =
    A specified number of look backs (integer)
        l_back = [1]
    A Boolean to decide if to scale the data (i.e. if not desired or already done)
        toscale = [True]

    Returns:
    The dataframe with a column of normalized, and lagged, data
    The scaler model that can be used to 'inverse' transform
        """

    # # subset
    dfin = dfin.loc[startdate:enddate]
    if toscale is True: 
        # # normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        # # add normalized to dataset
        dfin[dfname+'_norm'] = scaler.fit_transform(dfin[
                                                dfname].to_numpy().reshape(-1, 1))
        # # lag
        dfin[dfname+'_norm'+str(l_back)] = dfin[dfname+'_norm'].shift(l_back)
        return dfin, scaler
    else:
        dfin[dfname+str(l_back)] = dfin[dfname].shift(l_back)
        return dfin

def denorm_it(val, scaler):
    """De normalizes a single value

    Takes:
    A scaled value (a single number)
        val =
    A scaler from sklearn
        scaler =
    """
    # # inverse transform a single value
    newval = scaler.inverse_transform(val.reshape(1, -1))
    return newval


# %% Globals
start = '1989-01-01'
end = '2020-12-31'

# %%
# read in data
site = '09506000'
url = "https://waterdata.usgs.gov/nwis/dv?cb_00060=on" \
      "&format=rdb&site_no="+site+"&referred_module=sw&" \
      "period=&begin_date="+start+"&end_date="+end

flow_df = pd.read_table(url, sep='\t', skiprows=30,
                        names=['agency_cd', 'site_no',
                               'datetime', 'flow', 'code'],
                        parse_dates=['datetime'],
                        index_col='datetime'
                        )

# re-instantiate data with just the natural log of
# its flow values (to be used later)
flow_df = np.log(flow_df[['flow']])
flow_df.index = flow_df.index.tz_localize(tz="UTC")


# %%
# resamples flow through time
flow_df = flow_df.resample("Y").mean()
# # subset, normalize, and lag
flow_df, scale1 = norm_it(start, end, flow_df, 'flow', l_back=2)
# lag again
flow_df = norm_it(start, end, flow_df, 'flow_norm', l_back=4, toscale=False)

# %%
# create year column
flow_df['year'] = flow_df.index.year

# %%
# plot in time-series space
fig, ax = plt.subplots()
ax.plot(flow_df['year'], flow_df['flow_norm'])
ax.plot(flow_df['year'], flow_df['flow_norm2'])
ax.plot(flow_df['year'], flow_df['flow_norm4'])

plt.show()
# %%
# plot in 2-D phase space
fig, ax = plt.subplots()
print('no lag v 2 year lag')
ax.plot(flow_df['flow_norm'], flow_df['flow_norm2'], marker='D')
plt.show()

fig, ax = plt.subplots()
print('no lag v 4 year lag')
ax.plot(flow_df['flow_norm'], flow_df['flow_norm4'], marker='D', color='orange')
plt.show()

fig, ax = plt.subplots()
print('2 year lag v 4 year lag')
ax.plot(flow_df['flow_norm2'], flow_df['flow_norm4'], marker='D', color='green')
plt.show()

# %%
# from https://matplotlib.org/3.1.1/gallery/mplot3d/scatter3d.html

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(flow_df['flow_norm'], flow_df['flow_norm2'], flow_df['flow_norm4'], marker='D')

ax.set_xlabel('No Lag')
ax.set_ylabel('2 Lag')
ax.set_zlabel('4 Lag')

plt.show()

# %%
# from: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
# and from: https://stackoverflow.com/questions/10252412/matplotlib-varying-color-of-line-to-capture-natural-time-parameterization-in-da

print('no lag v 2 year lag in 2 dimensions')
x = flow_df['flow_norm'].values
y = flow_df['flow_norm2'].values
t = flow_df['year'].values

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# make the collection of segments
lc = LineCollection(segments, cmap=plt.get_cmap('jet'))
lc.set_array(t) # color the segments by our parameter

# plot the collection
plt.gca().add_collection(lc) # add the collection to the plot
plt.xlim(0, 1) # line collections don't auto-scale the plot
plt.ylim(0, 1)

plt.show()

# %%

print('no lag v 2 year lag v 3 year lag in 3 dimensions')
x = flow_df['flow_norm'].values
y = flow_df['flow_norm2'].values
z = flow_df['flow_norm4'].values
t = flow_df['year'].values

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 3 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 3 (for x and y)
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# make 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# make the collection of segments
lc = Line3DCollection(segments, cmap=plt.get_cmap('jet'))
lc.set_array(t) # color the segments by our parameter
fig.colorbar(lc)

# plot the collection
plt.gca().add_collection(lc) # add the collection to the plot
ax.set_xlabel('No Lag')
ax.set_ylabel('2 Lag')
ax.set_zlabel('4 Lag')
plt.show()



# %%
# from: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

x = np.linspace(0, 3 * np.pi, 500)
y = np.sin(x)
dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs[0].add_collection(lc)
fig.colorbar(line, ax=axs[0])

# Use a boundary norm instead
cmap = ListedColormap(['r', 'g', 'b'])
norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs[1].add_collection(lc)
fig.colorbar(line, ax=axs[1])

axs[0].set_xlim(x.min(), x.max())
axs[0].set_ylim(-1.1, 1.1)
plt.show()
# %%
