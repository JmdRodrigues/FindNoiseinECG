import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as grid
import seaborn as sns
import pandas
import itertools
import matplotlib.patheffects as pte
import numpy as np
from scipy.stats import kendalltau
from matplotlib.font_manager import FontProperties
from matplotlib.figure import SubplotParams
from matplotlib.markers import MarkerStyle
from FeaturesANDClustering.RemoveUnwantedPoints import RemoveUglyDots

def plotLinearData(time, FeatureMatrix, OSignal, FeatureNames, pp):
	"""

	:param time: Time array
	:param FeatureMatrix: Matrix with the arrays of all the features
	:param OSignal: Original Signal
	:param SegmNoise: Array with noise segmentation
	:return: plot object of these params
	"""

	# color
	face_color_r = 248 / 255.0
	face_color_g = 247 / 255.0
	face_color_b = 249 / 255.0

	# pars
	left = 0.05  # the left side of the subplots of the figure
	right = 0.95  # the right side of the subplots of the figure
	bottom = 0.05  # the bottom of the subplots of the figure
	top = 0.92  # the top of the subplots of the figure
	wspace = 0.2  # the amount of width reserved for blank space between subplots
	hspace = 2  # the amount of height reserved for white space between subplots

	pars = SubplotParams(left, bottom, right, top, wspace, hspace)

	#specify Font properties with fontmanager---------------------------------------------------
	font0 = FontProperties()
	font0.set_weight('medium')
	font0.set_family('monospace')
	#Specify Font properties of Legends
	font1 = FontProperties()
	font1.set_weight('normal')
	font1.set_family('sans-serif')
	font1.set_style('italic')
	font1.set_size(12)
	#Specify font properties of Titles
	font2 = FontProperties()
	font2.set_size(15)
	font2.set_family('sans-serif')
	font2.set_weight('medium')
	font2.set_style('italic')
	#Set Figure Size
	MatrixSize = np.shape(FeatureMatrix)
	fig, axes = plt.subplots(MatrixSize[1], 1)
	fig.set_dpi(96)
	fig.set_figheight(1080 / 96)
	fig.set_figwidth(1920 / 96)

	for i in range(0, MatrixSize[1]):

		axes[i].plot(time, FeatureMatrix[:, i] + 10, linewidth=0.5)
		axes[i].plot(time, OSignal, linewidth=0.5, alpha=.7)
		axes[i].set_ylabel(" Amplitude (r.u.) ", fontproperties=font1)
		axes[i].set_title("Signal Feature: " + FeatureNames[i], fontproperties=font2)
		axes[i].axis('tight')
		axes[i].axes.get_xaxis().set_visible(False)

	axes[MatrixSize[1] - 1].set_xlabel(" Time (s) ", fontproperties=font1)
	axes[MatrixSize[1] - 1].axes.get_xaxis().set_visible(True)

	pp.savefig(fig)


def plotClusters(y_pred, OSignal, time, XPCA, n_clusters, pp):
	"""
	:param y_pred: Matrix with Classification values
	:param OSignal: Original Signal
	:param time: Time array
	:param XPCA: Matrix after PCA. XPCA[0] - feature 1 and XPCA[1] - feature 2
	:param n_clusters: Number of clusters
	:return: plot object of clusters
	"""
	#Specify plot parameters
	# color
	face_color_r = 248 / 255.0
	face_color_g = 247 / 255.0
	face_color_b = 249 / 255.0

	# pars
	left = 0.05  # the left side of the subplots of the figure
	right = 0.95  # the right side of the subplots of the figure
	bottom = 0.05  # the bottom of the subplots of the figure
	top = 0.92  # the top of the subplots of the figure
	wspace = 0.5  # the amount of width reserved for blank space between subplots
	hspace = 0.4  # the amount of height reserved for white space between subplots

	pars = SubplotParams(left, bottom, right, top, wspace, hspace)

	#specify Font properties with fontmanager---------------------------------------------------
	font0 = FontProperties()
	font0.set_weight('medium')
	font0.set_family('monospace')
	#Specify Font properties of Legends
	font1 = FontProperties()
	font1.set_weight('normal')
	font1.set_family('sans-serif')
	font1.set_style('italic')
	font1.set_size(17)
	#Specify font properties of Titles
	font2 = FontProperties()
	font2.set_size(20)
	font2.set_family('sans-serif')
	font2.set_weight('medium')
	font2.set_style('italic')

	#Cluster colors---------------------------------------------------------------------------------------------
	# scatColors = np.array([x for x in ['darkseagreen', 'indianred', 'cornflowerblue', 'darkorange', 'indigo']])
	#scatColors = np.array([x for x in ['#93d1ff', '#ffc425', '#fc3366', '#032569']]) #pallete 1
	#scatColors = np.array([x for x in ['#ce3635', '#2caae2', '#2ce2aa', '#c38ce3']]) #pallete Gamut
	scatColors = np.array([x for x in ['#3366CC', '#79BEDB', '#E84150', '#FFB36D', '#6aba8f', '#78136f', '#236B5D',
	                                   '#AB5436','#3366CC', '#AB5436']])
	markers = np.array([x for x in {'o', 'v', 's', '*', '8', 'D', 'd', '+', 'o', 'v'}])
	Colors = itertools.cycle(scatColors)
	Markers = itertools.cycle(markers)

	#Create Grid Frame for Clustering Representation----------------------------------------
	PCAFrame = pandas.DataFrame(data=XPCA, columns=['Var1', 'Var2'])
	g = sns.JointGrid('Var1', 'Var2', PCAFrame)
	f2 = g.fig
	f2.set_dpi(96)
	f2.set_figheight(1080 / 96)
	f2.set_figwidth(1920 / 96)

	#Create figure for signal representation with clusters-----------------------------------
	f, axes = plt.subplots(n_clusters+1, 1)
	f.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	f.set_dpi(96)
	f.set_figheight(1080/96)
	f.set_figwidth(1920/96)
	f.set_facecolor((face_color_r, face_color_g, face_color_b))

	axes[0].plot(time, OSignal)
	lines = []
	labels = []

	#Cycle for figure plotting------------------------------------------------------------------------------
	for s in range(0, n_clusters):
		x = XPCA[np.where(y_pred== s)[0], 0]
		y = XPCA[np.where(y_pred== s)[0], 1]
		color = next(Colors)
		marker = next(Markers)
		line = mlines.Line2D([], [], color=color, marker=marker, markersize=15, markeredgecolor='gray',
		                     markerfacecolor=color, markeredgewidth=2, label='Cluster' + str(s + 1))
		lines.append(line)
		cmap = sns.light_palette(color, as_cmap=True, n_colors=120, reverse=True)
		ax = sns.kdeplot(x, y, cmap=cmap, shade=True, shade_lowest=False, alpha=1, ax=g.ax_joint)
		g.ax_joint.plot(np.mean(x), np.mean(y), marker=marker, markersize=15, markeredgecolor=color,
		                markerfacecolor=color, markeredgewidth=2)
		# txt = ax.text(np.mean(x), np.mean(y), "Cluster " + str(s + 1), fontproperties=font0, size=30, color=color)
		# txt.set_path_effects([pte.Stroke(linewidth=2.5, foreground='white'), pte.Normal()])
		sns.distplot(XPCA[np.where(y_pred == s)[0], 0], bins=50, kde=True, kde_kws={"color":color, 'shade':True, 'lw':2, 'alpha':0.3},
		             hist=False, rug=False, ax=g.ax_marg_x)
		sns.distplot(XPCA[np.where(y_pred == s)[0], 1], bins=50, kde=True, vertical=True,
		             kde_kws={"color": color, 'shade': True, 'lw': 2, 'alpha': 0.3},
		             hist=False, rug=False, ax=g.ax_marg_y)

		##################
		# Piece of array for line plotting
		##################

		Array1 = np.where(y_pred == s)[0]
		Array11 = np.flip(np.where(y_pred == s)[0], 0)
		DifArray1 = np.diff(Array1)
		DifArray11 = np.diff(Array11)

		Array2 = Array1[np.where(DifArray1 > 1)[0]]
		Array2 = np.insert(Array2, len(Array2), Array1[-1])
		Array22 = np.flip(Array11[np.where(DifArray11 < -1)[0]], 0)
		Array22 = np.insert(Array22, 0, Array1[0])

		#plot signal---------------------------------------------------------------------------------------------
		sns.despine(fig=f)
		# with sns.axes_style("darkgrid"):
		axes[s + 1].plot(time, OSignal, color="gray", alpha=0.2)
		# axes[s + 1].plot(time[np.where(y_pred == s)[0]], OSignal[np.where(y_pred == s)[0]], '.', color=color, marker = marker)
		# axes[s + 1].patch.set_facecolor('ivory')
		if (len(Array2) == 0):
			axes[s + 1].plot(time[Array1], OSignal[Array1], '-', color=color, linewidth=0.5)
		for i in range(0, len(Array2)):
			axes[s + 1].plot(time[Array22[i]:Array2[i]], OSignal[Array22[i]:Array2[i]], '-', color=color, linewidth=0.5)
		axes[s + 1].axis('tight')
		axes[s + 1].tick_params(labelsize=15)
		axes[s + 1].set_ylabel(" Amplitude (r.u.) ", fontproperties=font1)
		axes[s + 1].set_title("Signal representation from Cluster: " + str(s), fontproperties=font2)
		axes[s + 1].grid(True, linestyle='-', color = 'white')
		axes[s + 1].axes.get_xaxis().set_visible(False)

	axes[n_clusters].set_xlabel(" Time (s) ", fontproperties=font1)
	axes[n_clusters].axes.get_xaxis().set_visible(True)
	#print(markers[y_pred.tolist()])

	for c in range(0, n_clusters):
		# xTemp = XPCA[np.where(y_pred == c)[0], 0]
		# yTemp = XPCA[np.where(y_pred == c)[0], 1]
		# XPCATemp = XPCA[np.where(y_pred == c)[0], :]

		labels = [h.get_label() for h in lines]

	g.fig.legend(handles=lines, labels=labels, labelspacing=1.5, bbox_to_anchor=(0.85, 0.83), loc='lower left',
		             fontsize=10)
		# g.ax_joint.plot(xTemp[Tempor], yTemp[Tempor], color='black', marker='o')

		#Find Dots for that Cluster
		# Selection, Tempor = RemoveUglyDots(XPCATemp)

		# for _X, _Y, _color, _marker in zip(xTemp[Tempor], yTemp[Tempor], scatColors[y_pred.tolist()], markers[y_pred.tolist()]):
		# 	g.ax_joint.plot(_X, _Y, color=scatColors[c], marker=markers[c], alpha=.5, linewidth=.5)
		# g.ax_joint.plot(xTemp[Tempor], yTemp[Tempor], color='black', marker='o')

	# for _Data, _color, _marker in zip(XPCA, scatColors[y_pred.tolist()], markers[y_pred.tolist()]):
	# 	g.ax_joint.plot(_Data[0], _Data[1] , color = _color, marker=_marker, alpha=.5, linewidth=.5)
	# g.plot_joint(plt.scatter, s=10, color=scatColors[y_pred.tolist()], alpha=.35, linewidth=.5)
	#g.ax_joint.scatter(XPCA[:,0], XPCA[:,1], color=scatColors[y_pred.tolist()], alpha=.35, linewidth=.5)
	g.set_axis_labels("Feature 1", "Feature 2", fontsize=20, fontproperties=font1)
	g.fig.suptitle("Clusters after selecting 2 features with PCA ", color="gray", alpha=.9, fontsize=15)
	g.fig.tight_layout()

	pp.savefig(f2)
	pp.savefig(f)

	plt.close('all')

def plotClustersBW(y_pred, OSignal, time, XPCA, n_clusters, pp):
	"""
	:param y_pred: Matrix with Classification values
	:param OSignal: Original Signal
	:param time: Time array
	:param XPCA: Matrix after PCA. XPCA[0] - feature 1 and XPCA[1] - feature 2
	:param n_clusters: Number of clusters
	:return: plot object of clusters
	"""
	#Specify plot parameters
	# color
	face_color_r = 248 / 255.0
	face_color_g = 247 / 255.0
	face_color_b = 249 / 255.0

	# pars
	left = 0.05  # the left side of the subplots of the figure
	right = 0.95  # the right side of the subplots of the figure
	bottom = 0.05  # the bottom of the subplots of the figure
	top = 0.92  # the top of the subplots of the figure
	wspace = 0.5  # the amount of width reserved for blank space between subplots
	hspace = 0.4  # the amount of height reserved for white space between subplots

	pars = SubplotParams(left, bottom, right, top, wspace, hspace)

	#specify Font properties with fontmanager---------------------------------------------------
	font0 = FontProperties()
	font0.set_weight('medium')
	font0.set_family('monospace')
	#Specify Font properties of Legends
	font1 = FontProperties()
	font1.set_weight('normal')
	font1.set_family('sans-serif')
	font1.set_style('italic')
	font1.set_size(17)
	#Specify font properties of Titles
	font2 = FontProperties()
	font2.set_size(13)
	font2.set_family('sans-serif')
	font2.set_weight('medium')
	font2.set_style('italic')

	#Cluster colors---------------------------------------------------------------------------------------------
	# scatColors = np.array([x for x in ['darkseagreen', 'indianred', 'cornflowerblue', 'darkorange', 'indigo']])
	#scatColors = np.array([x for x in ['#93d1ff', '#ffc425', '#fc3366', '#032569']]) #pallete 1
	#scatColors = np.array([x for x in ['#ce3635', '#2caae2', '#2ce2aa', '#c38ce3']]) #pallete Gamut
	scatColors = np.array([x for x in ['#d6d6d6', '#c1c1c1', '#adadad', '#999999', '#7a7a7a', '#5b5b5b', '#3d3d3d',
	                                   '#1e1e1e','#000000']])
	markers = np.array([x for x in {'o', 'd', 's', '*', '+', '8', 'D', 'o', 'v'}])
	Colors = itertools.cycle(scatColors)
	Markers = itertools.cycle(markers)

	#Create Grid Frame for Clustering Representation----------------------------------------
	PCAFrame = pandas.DataFrame(data=XPCA, columns=['Var1', 'Var2'])
	g = sns.JointGrid('Var1', 'Var2', PCAFrame)
	f2 = g.fig
	f2.set_dpi(96)
	f2.set_figheight(1080 / 96)
	f2.set_figwidth(1920 / 96)

	#Create figure for signal representation with clusters-----------------------------------
	f, axes = plt.subplots(n_clusters+1, 1, sharex=True, sharey=True)
	f.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	f.set_dpi(96)
	f.set_figheight(1080/96)
	f.set_figwidth(1920/96)
	f.set_facecolor((face_color_r, face_color_g, face_color_b))

	axes[0].plot(time, OSignal, linewidth=0.5, color='k')
	lines = []
	labels = []

	#Cycle for figure plotting------------------------------------------------------------------------------
	for s in range(0, n_clusters):
		x = XPCA[np.where(y_pred== s)[0], 0]
		y = XPCA[np.where(y_pred== s)[0], 1]
		color = next(Colors)
		marker = next(Markers)
		line = mlines.Line2D([], [], color ='lightgray', marker = marker, markersize=15, markeredgecolor='gray', markerfacecolor='lightgrey', markeredgewidth=2, label='Cluster'+str(s+1))
		lines.append(line)
		cmap = sns.light_palette('#7a7a7a', as_cmap=True, n_colors=120, reverse=True)
		ax = sns.kdeplot(x, y, cmap=cmap, shade=True, shade_lowest=False, alpha=1, n_levels=25, marker=marker, markersize=15, legend=True, label="Cluster"+str(s+1),ax=g.ax_joint)
		# txt = ax.text(max, 'top', "Cluster " + str(s + 1), fontproperties=font0, size=30, color=color)
		# txt = ax.text(np.mean(x), np.mean(y), "Cluster " + str(s + 1), fontproperties=font0, size=30, color=color)
		g.ax_joint.plot(np.mean(x), np.mean(y), marker=marker, markersize=15, markeredgecolor='k', markerfacecolor='lightgrey', markeredgewidth=2)
		# txt.set_path_effects([pte.Stroke(linewidth=2.5, foreground='white'), pte.Normal()])
		sns.distplot(XPCA[np.where(y_pred == s)[0], 0], bins=50, kde=True, kde_kws={"color":'gray', 'shade':True, 'lw':2, 'alpha':0.3},
		             hist=False, rug=False, ax=g.ax_marg_x)

		sns.distplot(XPCA[np.where(y_pred == s)[0], 1], bins=50, kde=True, kde_kws={"color":'gray', 'shade':True, 'lw':2, 'alpha':0.3},
		             hist=False, rug=False, vertical=True, ax=g.ax_marg_y)

		##################
		#Piece of array for line plotting
		##################

		Array1 = np.where(y_pred == s)[0]
		Array11 = np.flip(np.where(y_pred == s)[0], 0)
		DifArray1 = np.diff(Array1)
		DifArray11 = np.diff(Array11)

		Array2 = Array1[np.where(DifArray1 > 1)[0]]
		Array2 = np.insert(Array2, len(Array2), Array1[-1])
		Array22 = np.flip(Array11[np.where(DifArray11 < -1)[0]], 0)
		Array22 = np.insert(Array22, 0, Array1[0])

		#plot signal---------------------------------------------------------------------------------------------
		sns.despine(fig=f)
		# with sns.axes_style("darkgrid"):
		axes[s + 1].plot(time, OSignal, color='gray', alpha=0.2)
		if (len(Array2) == 0):
			axes[s + 1].plot(time[Array1], OSignal[Array1], '-', color="black", linewidth=0.5)
		for i in range(0, len(Array2)):
			axes[s + 1].plot(time[Array22[i]:Array2[i]], OSignal[Array22[i]:Array2[i]], '-', color="black", linewidth=0.5)
		# axes[s + 1].patch.set_facecolor('ivory')
		axes[s + 1].axis('tight')
		axes[s + 1].tick_params(labelsize=15)
		axes[s + 1].set_ylabel(" Amplitude (r.u.) ", fontproperties=font1)
		# axes[s + 1].set_title("Signal representation from Cluster: " + str(s+1), fontproperties=font2)
		axes[s + 1].grid(True, linestyle='-', color = 'white')
		axes[s + 1].axes.get_xaxis().set_visible(False)

	# f.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
	axes[n_clusters].set_xlabel(" Time (s) ", fontproperties=font1)
	axes[n_clusters].axes.get_xaxis().set_visible(True)
	#print(markers[y_pred.tolist()])
	cb_ax = g.fig.add_axes([0.05, 0.12, 0.1, 0.012])
	norm=mpl.colors.Normalize(vmin=0, vmax=1)
	cmapBW = sns.light_palette('#7a7a7a', as_cmap=True, n_colors=120, reverse=True)
	cb1 = mpl.colorbar.ColorbarBase(cb_ax, cmap=cmapBW, norm=norm, orientation='horizontal')

	for c in range(0, n_clusters):
		# xTemp = XPCA[np.where(y_pred == c)[0], 0]
		# yTemp = XPCA[np.where(y_pred == c)[0], 1]
		# XPCATemp = XPCA[np.where(y_pred == c)[0], :]
		#
		# #Find Dots for that Cluster
		# Selection, Tempor = RemoveUglyDots(XPCATemp)
		#
		# #Este e que e o passo muito lento
		# for _X, _Y, _color, _marker in zip(xTemp[Tempor], yTemp[Tempor], scatColors[y_pred.tolist()], markers[y_pred.tolist()]):
		# 	# g.ax_joint.plot(_X, _Y, color=scatColors[c], marker=markers[c], markersize=10, alpha=.5, linewidth=.5)
		# 	g.ax_joint.plot(_X, _Y, color='w', marker=markers[c], markeredgecolor='k', markeredgewidth=0.5, markersize=7.5, alpha=.5, linewidth=.3)
		# # print(ax2.)
		labels = [h.get_label() for h in lines]

	g.fig.legend(handles=lines, labels=labels , labelspacing=1.5, bbox_to_anchor=(0.85, 0.83), loc='lower left', fontsize=10)
		# g.ax_joint.plot(xTemp[Tempor], yTemp[Tempor], color='black', marker='o')


	# for _Data, _color, _marker in zip(XPCA, scatColors[y_pred.tolist()], markers[y_pred.tolist()]):
	# 	g.ax_joint.plot(_Data[0], _Data[1] , color = _color, marker=_marker, alpha=.5, linewidth=.5)
	# g.plot_joint(plt.scatter, s=10, color=scatColors[y_pred.tolist()], alpha=.35, linewidth=.5)
	#g.ax_joint.scatter(XPCA[:,0], XPCA[:,1], color=scatColors[y_pred.tolist()], alpha=.35, linewidth=.5)
	g.set_axis_labels("Feature 1", "Feature 2", fontsize=20, fontproperties=font1)
	# g.fig.suptitle("Clusters after selecting 2 features with PCA ", color="gray", alpha=.9, fontsize=15)
	g.fig.tight_layout()


	pp.savefig(f2)
	pp.savefig(f)

	plt.close('all')



def plotDistanceMetric(time, signal, signalNoise, Prediction, Spec, Sens, CorrectPeriod, sizes, pp):

	#define colors----------------------------------------------------------------------------------------------
	# colors = ['#556270', '#4ECDC4', '#FF6B6B', '#C44D58']
	# colors =['#339900', '#99cc33', '#ff9966', '#cc3300']
	# colors = ['#3c469c', '#b7b3ef', '#f68984', '#f37370']
	# colors = ['#009688','#4db6ac','#f63737','#ff0000']
	# colors = ['#3366CC', '#79BEDB', '#E84150', '#FFB36D']
	colors = ['#3366CC', '#79BEDB', '#E84150', '#FFB36D']

	#Specify Font Properties of text-------------------------------------------------------------------
	font0 = FontProperties()
	font0.set_weight('medium')
	font0.set_family('monospace')
	font0.set_size(25)

	# Specify Font properties of Legends
	font1 = FontProperties()
	font1.set_weight('normal')
	font1.set_family('sans-serif')
	font1.set_style('italic')
	font1.set_size(14)

	# Specify font properties of Titles
	font2 = FontProperties()
	font2.set_size(16)
	font2.set_family('sans-serif')
	font2.set_weight('medium')
	font2.set_style('italic')

	#create grid-------------------------------------------------------------------------------------------------
	fig = plt.figure()
	fig.set_dpi(96)
	fig.set_figheight(1080/96)
	fig.set_figwidth(1920/96)
	gs0 = grid.GridSpec(3, 2)

	ax1 = plt.subplot(gs0[:, 1])
	ax2 = plt.subplot(gs0[0, 0])
	ax3 = plt.subplot(gs0[1, 0])
	ax4 = plt.subplot(gs0[2, 0])

	#plot signal--------------------------------------------------------------------------------------------------
	ax2.plot(time, signal / max(signal), color='#263763')
	ax2.plot(time, signalNoise, color=colors[0], linewidth=2, label="Noise Selection")
	ax2.fill_between(time, signalNoise, where=signalNoise == 1, facecolor=colors[1], alpha=0.6, interpolate=True)
	ax2.tick_params(labelsize=10)
	ax2.set_xlabel(" Time (s) ", fontproperties=font1)
	ax2.set_ylabel(" Amplitude (r.u.)", fontproperties=font1)
	ax2.set_title("True Noise Selection", fontproperties=font2)
	ax2.axis('tight')
	ax3.plot(time, signal / max(signal), color='#263763')
	ax3.plot(time, Prediction, color=colors[2], linewidth=2, label='Noise Predicted Classification')
	ax3.fill_between(time, Prediction, where=Prediction == 1, facecolor=colors[3], alpha=0.6, interpolate=True)
	ax3.tick_params(labelsize=10)
	ax3.set_xlabel(" Time (s) ", fontproperties=font1)
	ax3.set_ylabel(" Amplitude (r.u.)", fontproperties=font1)
	ax3.set_title("Noise Prediction", fontproperties=font2)
	ax3.axis('tight')
	ax4.plot(time, signal / max(signal), color='#263763')
	ax4.plot(time, Prediction, color=colors[2], linewidth=2, label='Correct')
	ax4.plot(time, 1.01 * signalNoise, color=colors[0], linewidth=2, label='Error')
	ax4.fill_between(time, Prediction, where=Prediction == 1, facecolor=colors[3], alpha=0.6, interpolate=True)
	ax4.fill_between(time, 1.01 * signalNoise, where=signalNoise == 1, facecolor=colors[1], alpha=0.6, interpolate=True)
	ax4.tick_params(labelsize=10)
	ax4.set_xlabel(" Time (s) ", fontproperties=font1)
	ax4.set_ylabel(" Amplitude (r.u.)", fontproperties=font1)
	ax4.set_title("Superposition", fontproperties=font2)
	ax4.axis('tight')

	#plot pie chart---------------------------------------------------------------------------------------------
	labels = 'TP', 'TN', 'FP', 'FN'

	TP = sizes[0]/sum(sizes)
	TN = sizes[1]/sum(sizes)
	FP = sizes[2]/sum(sizes)
	FN = sizes[3]/sum(sizes)

	perc = [TP, TN, FP, FN]
	explode = (0, 0, 0.1, 0.1)  # explode a slice if required
	patches, texts, autotexts = ax1.pie(sizes, colors=colors, shadow=False, explode=explode, autopct='%1.1f%%',
	                                    startangle=180)  # draw a circle at the center of pie to make it look like a donut

	for p, at, pc in zip(patches, autotexts, perc):
		p.set_linewidth(3)
		p.set_alpha(0.8)
		p.set_edgecolor('lightgray')
		at.set_fontproperties(font0)
		at.set_size(25 * np.log10(100 * pc + 2))
		at.set_path_effects([pte.Stroke(linewidth=2, foreground='slategray'), pte.Normal()])
		at.set_color('black')

	ax1.legend(patches, labels, loc='best', fontsize=15)
	centre_circle = plt.Circle((0, 0), 0.35, color='lightgray', fc='white', linewidth=3)
	ax1.add_artist(centre_circle)
	text = 'Sensitivity = ' + str(round(Sens * 100.0, 2)) + ' %' + '\n' + 'Specificity = ' + str(
		round(Spec * 100.0, 2)) + ' %' + '\n' + 'Accuracy = ' + str(round(CorrectPeriod * 100.0, 2)) + ' %'
	kwargs = dict(size=15, fontweight='medium', va='center', color='slategray')
	txt = ax1.text(0, 0, text, ha='center', **kwargs)
	txt.set_path_effects([pte.Stroke(linewidth=2.5, foreground='white'), pte.Normal()])

	ax1.axis('equal')
	gs0.tight_layout(fig)

	pp.savefig()

	plt.close('all')
