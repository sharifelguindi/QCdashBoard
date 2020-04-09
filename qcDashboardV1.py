import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_table

import os
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
import scipy.io as sio
import SimpleITK as sitk
import time
from imgAug import normalize_equalize_smooth_CT
# Helper functions for datetime parsing ##
epoch = datetime.utcfromtimestamp(0)


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


def smooth_image(arr, t_step=0.125, n_iter=3):
    img = sitk.GetImageFromArray(arr)
    img = sitk.CurvatureFlow(image1=img,
                             timeStep=t_step,
                             numberOfIterations=n_iter)
    arr_smoothed = sitk.GetArrayFromImage(img)
    return arr_smoothed


# Initial DB Pull
df = pd.read_excel('G:\\Projects\\AutoQC\\headNeckDB\\headNeckDB.xlsx')
df.index = df['SCAN_DATETIME']
structureList = [col.split(',')[0].replace('_DSC', '') for col in df if col.split(',')[0].endswith('_DSC')]
contourNames = [col for col in df if col.split(',')[0].endswith('_DSC')]
contourArr = []
i = 0
for contourName in contourNames:
    name, number = contourName.split(',')
    contourNames[i] = name.replace('_DSC', '')
    contourArr.append(int(number))
    i = i + 1
metricSelectionOptions = ['APL', '3D DSC', '95% 2D HD', 'SDSC_1mm', 'SDSC_3mm']

# DASH APP
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[

    html.H1(children='QC Dashboard'),

    html.Div(children='''
    Explore each organ segmentation at MSKCC
    '''),

    html.Div([

        dcc.DatePickerRange(
            id='dateRangeSelection',
            min_date_allowed=df.index.min(),
            max_date_allowed=df.index.max(),
            start_date=df.index.min(),
            end_date=df.index.max(),
            initial_visible_month=df.index.min(),
            start_date_placeholder_text="Start Period",
            end_date_placeholder_text="End Period",
            calendar_orientation='vertical',
            ),

        dcc.Dropdown(
            id='dropdownOrganSelection',
            options=[{'label': i, 'value': i} for i in structureList],
            value=structureList[0]
            ),

        dcc.Dropdown(
            id='dropdownMetricSelection',
            options=[{'label': i, 'value': i} for i in metricSelectionOptions],
            value=metricSelectionOptions[0]
        ),

        dcc.Graph(
            id='quantitative-graph'
            ),

    ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 0'}),

    html.Div([
        dcc.Graph(id='segmentationDisplay')
    ], style={'display': 'inline-block', 'width': '40%'}),

    # Hidden div inside the app that stores processed data (if needed)
    html.Div(id='intermediate-data', style={'display': 'none'})

])


@app.callback(
    Output('quantitative-graph', 'figure'),
    [Input('dateRangeSelection', 'start_date'),
     Input('dateRangeSelection', 'end_date'),
     Input('dropdownOrganSelection', 'value'),
     Input('dropdownMetricSelection', 'value')])
def update_figure(start_date, end_date, organName, comparisonMetric):
    columnName = organName + '_' + comparisonMetric
    fig = go.Figure(go.Scatter(
        x=df['SCAN_DATETIME'],
        y=df[columnName],
        mode='markers'
    ))

    fig.update_xaxes(
        rangeslider_visible=True,
        tickformatstops=[
            dict(dtickrange=[None, 1000], value="%H:%M:%S.%L ms"),
            dict(dtickrange=[1000, 60000], value="%H:%M:%S s"),
            dict(dtickrange=[60000, 3600000], value="%H:%M m"),
            dict(dtickrange=[3600000, 86400000], value="%H:%M h"),
            dict(dtickrange=[86400000, 604800000], value="%e. %b d"),
            dict(dtickrange=[604800000, "M1"], value="%e. %b w"),
            dict(dtickrange=["M1", "M12"], value="%b '%y M"),
            dict(dtickrange=["M12", None], value="%Y Y")
        ]
    )

    # for i in filtered_col:
    #     df_by_structure = filtered_df[i]
    #     xList = df_by_structure.index
    #     # tickLabels = get_marks_from_start_end(df_by_structure.index.min(), df_by_structure.index.max())
    #     #
    #     # tickLabelList = []
    #     # for key, value in tickLabels.items():
    #     #     tickLabelList.append(value)
    #
    # traces.append(dict(
    #     x=filtered_df.index.values.tolist(),
    #     y=filtered_df.values.tolist(),
    #     text=organName,
    #     mode='markers',
    #     opacity=0.6,
    #     marker={
    #         'size': 15,
    #         'line': {'width': 0.5, 'color': 'white'}
    #     },
    #     name=organName
    # ))

    return fig


@app.callback(Output('segmentationDisplay', 'figure'),
              [Input('quantitative-graph', 'clickData'),
               Input('dropdownOrganSelection', 'value'),
               Input('dropdownMetricSelection', 'value')])
def load_data(clickData, organName, comparisonMetric):
    height = width = 128
    if clickData is not None:
        t0 = time.time()
        utc_value = clickData['points'][0]['x']
        dsc_score = clickData['points'][0]['y']
        highlightedColumn = organName + '_' + comparisonMetric
        print(highlightedColumn)
        print(contourArr[structureList.index(organName)])
        contourIndex = contourArr[structureList.index(organName)] - 1
        curData = df[df[highlightedColumn] == dsc_score]
        dataFile = curData['basePath'][0] + os.sep + curData['dataFilename'][0]
        pixels = sio.loadmat(dataFile)
        volume = pixels['data'][contourIndex, 9]
        initContour = pixels['data'][contourIndex, 7]
        subVolume = np.copy(volume)
        np.place(subVolume, subVolume == 1, 0)
        np.place(volume, volume == -1, 0)

        bbox = pixels['data'][contourIndex, 12]
        bbox = bbox[0, :]

        canvas_width = 500
        # planCFile = dataFile.replace('data', 'planC')
        # planCStruct = sio.loadmat(planCFile)
        scan = pixels['data'][1, 15]

        t1 = time.time()
        print('Time to Load Data: ', str(t1 - t0))

        t0 = time.time()

        scan = scan
        mask = np.zeros(np.shape(scan))
        addedVol = np.zeros(np.shape(scan))
        subVol = np.zeros(np.shape(scan))
        autoC = np.zeros(np.shape(scan))

        scan_norm = normalize_equalize_smooth_CT(scan.transpose(2, 1, 0), 1)

        scan = scan_norm.transpose(3, 2, 0, 1)
        addedVol[bbox[0] - 1:bbox[1], bbox[2] - 1:bbox[3], bbox[4] - 1:bbox[5]] = volume
        subVol[bbox[0] - 1:bbox[1], bbox[2] - 1:bbox[3], bbox[4] - 1:bbox[5]] = subVolume
        autoC[bbox[0] - 1:bbox[1], bbox[2] - 1:bbox[3], bbox[4] - 1:bbox[5]] = initContour

        centroid_contour = [int(np.mean([bbox[0], bbox[1] + 1])), int(np.mean([bbox[2], bbox[3] + 1])),
                            int(np.mean([bbox[4], bbox[5] + 1]))]

        scan_arr = scan[centroid_contour[0] - int(height / 2):centroid_contour[0] + int(height / 2),
                   centroid_contour[1] - int(height / 2):centroid_contour[1] + int(height / 2),
                   int(bbox[4]) - 5:int(bbox[5]) + 5, :]

        mask_arr_add = addedVol[centroid_contour[0] - int(height / 2):centroid_contour[0] + int(height / 2),
                       centroid_contour[1] - int(height / 2):centroid_contour[1] + int(height / 2),
                       int(bbox[4]) - 5:int(bbox[5]) + 5]

        mask_arr_sub = subVol[centroid_contour[0] - int(height / 2):centroid_contour[0] + int(height / 2),
                       centroid_contour[1] - int(height / 2):centroid_contour[1] + int(height / 2),
                       int(bbox[4]) - 5:int(bbox[5]) + 5]

        mask_arr_auto = autoC[centroid_contour[0] - int(height / 2):centroid_contour[0] + int(height / 2),
                        centroid_contour[1] - int(height / 2):centroid_contour[1] + int(height / 2),
                        int(bbox[4]) - 5:int(bbox[5]) + 5]

        coloredMask = np.zeros(np.shape(scan_arr))
        h, w, l, c = np.shape(coloredMask)
        coloredMask[:, :, :, 0] = (mask_arr_add * 255).astype('uint8')
        coloredMask[:, :, :, 1] = (mask_arr_auto * 255).astype('uint8')
        coloredMask[:, :, :, 2] = (mask_arr_sub * -255).astype('uint8')
        displayImg = np.zeros(np.shape(scan_arr.astype('uint8')))
        frames = []
        for i in range(0, l):
            blended_img = cv2.addWeighted(coloredMask[:, :, i, :].astype('uint8'), 0.2,
                                          scan_arr[:, :, i, :].astype('uint8'), 0.8, 0)
            displayImg[:, :, i, :] = blended_img.astype('uint8')
            frames.append(go.Frame(data=go.Image(z=blended_img.astype('uint8')), name=str(i)))
        t1 = time.time()
        print('Time to Process Data: ', str(t1 - t0))
    else:
        displayImg = np.zeros((height, width, 1, 3))

    t0 = time.time()
    fig = go.Figure(frames=frames)
    fig.add_trace(go.Image(z=displayImg[:, :, 0, :].astype('uint8')))
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    fig.update_layout(
        title='Slices in volumetric data',
        width=600,
        height=600,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    t1 = time.time()
    print('Time to create figure: ', str(t1 - t0))
    return fig


if __name__ == '__main__':
    PORT = 8000
    ADDRESS = "172.25.132.135"
    app.run_server(debug=True,
                   port=PORT,
                   host=ADDRESS)
