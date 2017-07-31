"""References:
	3D clustering: https://plot.ly/python/3d-point-clustering/
	2D clustering/density plot: https://plot.ly/python/density-plots/"""
#
import plotly
plotly.tools.set_credentials_file(username='JeanRaltique', api_key='aux6iaauw3')
from plotly.tools import FigureFactory as FF

import numpy as np

t = np.linspace(-1, 1.2, 2000)
x = (t**3) + (0.3 * np.random.randn(2000))
y = (t**6) + (0.3 * np.random.randn(2000))

colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]

fig = FF.create_2D_density(
    x, y, colorscale=colorscale,
    hist_color='rgb(255, 237, 222)', point_size=3
)

plotly.offline.plot(fig, filename='histogram_subplots')