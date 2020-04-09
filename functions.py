import pandas as pd
import os
from shapely.geometry import *
import matplotlib.pyplot as plt


def plot_polygons_and_linestrings(structure_to_plot, color_for_plot='#00000'):

    if isinstance(structure_to_plot, MultiLineString):
        for bit_to_plot in structure_to_plot:
            x, y = bit_to_plot.xy
            plt.plot(x, y, color=color_for_plot)
    elif isinstance(structure_to_plot, MultiPolygon):
        for bit_to_plot in structure_to_plot:
            x, y = bit_to_plot.boundary.xy
            plt.plot(x, y, color=color_for_plot)
    elif isinstance(structure_to_plot, Polygon):
        x, y = structure_to_plot.boundary.xy
        plt.plot(x, y, color=color_for_plot)
    elif isinstance(structure_to_plot, LineString):
        x, y = structure_to_plot.xy
        plt.plot(x, y, color=color_for_plot)
    else:
        print('Unable to plot structure type: ', type(structure_to_plot))


def addMetricData(dbFileName):
    df = pd.read_excel(dbFileName)
    filtered_col = [col for col in df if col.split(',')[0].endswith('DSC')]
    metricFile = os.path.join(df['basePath'].iloc[0], df['dataFilename'].iloc[0]).replace('data','metric').replace('.mat', '.csv')
    metricDF = pd.read_csv(metricFile)
    columns_to_add = []
    for col in filtered_col:
        for metric in metricDF.columns:
            if metric != 'Organ':
                columns_to_add.append(col.split(',')[0].replace('_DSC', '') + '_' + metric)

    for col in columns_to_add:
        df[col] = ""
    i = 0
    for pID in df['MRN']:
        if isinstance(df['basePath'][i], str):
            metricFile = os.path.join(df['basePath'].iloc[i], df['dataFilename'].iloc[i]).replace('data','metric').replace('.mat','.csv')
            metricDF = pd.read_csv(metricFile)
            j = 0
            for organ in metricDF['Organ']:
                k = 0
                for metric in metricDF.columns:
                    if metric != 'Organ':
                        columnName = organ + "_" + metric
                        df[columnName].iloc[i] = metricDF.iloc[j, k]
                    k = k + 1
                j = j + 1
        i = i + 1

    return df


def main():
    databaseFile = 'G:\\Projects\\AutoQC\\prostateDB\\prostateDB.xlsx'
    df = addMetricData(databaseFile)
    df.to_excel(databaseFile)


if __name__ == '__main__':
    main()