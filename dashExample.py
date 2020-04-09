import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import dash_canvas
from dash_canvas.utils import (image_string_to_PILImage, array_to_data_url,
                               parse_jsonstring_line)
import os
import base64
import json
from textwrap import dedent as d
from dateutil.relativedelta import relativedelta
from datetime import datetime

import pandas as pd
from skimage import io, data
import numpy as np
import cv2
import PIL
import scipy.io as sio
from skimage import exposure
import SimpleITK as sitk

## Helper functions for datetime parsing ##
epoch = datetime.utcfromtimestamp(0)


def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0


def get_marks_from_start_end(start, end):
    ''' Returns dict with one item per month
    {1440080188.1900003: '2015-08',
    '''
    result = []
    current = start
    while current <= end:
        result.append(current)
        current += relativedelta(months=1)
    return {unix_time_millis(m) / 100000: (str(m.strftime('%Y-%m'))) for m in result}


def smooth_image(arr, t_step=0.125, n_iter=3):
    img = sitk.GetImageFromArray(arr)
    img = sitk.CurvatureFlow(image1=img,
                             timeStep=t_step,
                             numberOfIterations=n_iter)
    arr_smoothed = sitk.GetArrayFromImage(img)
    return arr_smoothed


def normalize_equalize_smooth_CT(arr):
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(int(arr.shape[1] / 32), int(arr.shape[2] / 32)))
    clahe = None
    length, width, height = np.shape(arr)
    norm_arr = np.zeros(np.shape(arr), dtype='float')
    norm_arr = cv2.normalize(arr, norm_arr, 0, np.max(arr), cv2.NORM_MINMAX)
    hist, bin_edges = np.histogram(norm_arr[:], int(np.max(arr)))
    # cum_sum = np.cumsum(hist.astype('float') / np.sum(hist).astype('float'))
    # clip_value_t = bin_edges[np.min(np.where(cum_sum > pct_remove_t))]
    # clip_value_b = bin_edges[np.max(np.where(cum_sum < pct_remove_b))]
    norm_arr = np.clip(norm_arr, 700, 1155)
    norm_eq = np.zeros((length, 3, width, height), dtype='uint16')
    if clahe is not None:
        for ii in range(0, length):
            if ii == 0:
                img_0 = clahe.apply(norm_arr[ii + 0, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii + 0, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii + 0, :, :].astype('uint16'))
            elif ii == length - 1:
                img_0 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
            else:
                img_0 = clahe.apply(norm_arr[ii - 0, :, :].astype('uint16'))
                img_1 = clahe.apply(norm_arr[ii + 0, :, :].astype('uint16'))
                img_2 = clahe.apply(norm_arr[ii + 0, :, :].astype('uint16'))

            norm_eq[ii, 0, :, :] = smooth_image(img_0).astype('uint16')
            norm_eq[ii, 1, :, :] = smooth_image(img_1).astype('uint16')
            norm_eq[ii, 2, :, :] = smooth_image(img_2).astype('uint16')
    else:
        for ii in range(0, length):
            if ii == 0:
                img_0 = norm_arr[ii + 0, :, :].astype('uint16')
                img_1 = norm_arr[ii + 0, :, :].astype('uint16')
                img_2 = norm_arr[ii + 0, :, :].astype('uint16')
            elif ii == length - 1:
                img_0 = norm_arr[ii - 0, :, :].astype('uint16')
                img_1 = norm_arr[ii, :, :].astype('uint16')
                img_2 = norm_arr[ii, :, :].astype('uint16')
            else:
                img_0 = norm_arr[ii - 0, :, :].astype('uint16')
                img_1 = norm_arr[ii + 0, :, :].astype('uint16')
                img_2 = norm_arr[ii + 0, :, :].astype('uint16')

            norm_eq[ii, 0, :, :] = img_0.astype('uint16')
            norm_eq[ii, 1, :, :] = img_1.astype('uint16')
            norm_eq[ii, 2, :, :] = img_2.astype('uint16')

    norm_arr_ds = np.zeros(np.shape(norm_eq), dtype='uint8')
    norm_arr_ds = cv2.normalize(norm_eq, norm_arr_ds, 0, 255, cv2.NORM_MINMAX)
    return norm_arr_ds.astype('uint8')


## Initial DB Pull ##
# df = pd.read_excel("D:\\pythonProjects\\autoQC\\headneckDB.xlsx")
df = pd.read_excel("G:\Projects\AutoQC\prostateDB\prostateDB.xlsx")
if not np.dtype('datetime64[ns]') == df.index.dtype:
    i = 0
    for dtObj in df['Datetime']:
        if isinstance(dtObj, str):
            try:
                dtObjConverted = datetime.strptime(dtObj, '%Y%m%d %H%M%S.%f')
                df['Datetime'][i] = dtObjConverted
            except ValueError:
                dtObjConverted = datetime.strptime(dtObj, '%Y%m%d %H%M%S')
                df['Datetime'][i] = dtObjConverted
        i = i + 1
    df.index = df['Datetime']
filtered_col = [col for col in df if col.endswith('DSC')]
df_DCE_only = df[filtered_col]
marksDict = get_marks_from_start_end(df['Datetime'].min(), df['Datetime'].max())

## Dash Canvas, data-display options
height = width = 64
canvas_width = 600
canvas_height = round(height * canvas_width / width)
scale = canvas_width / width

## DASH APP ##
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='graph-with-slider'),
        dcc.RangeSlider(
            id='datetime_RangeSlider',
            updatemode='mouseup',  # don't let it update till mouse released
            min=unix_time_millis(df['Datetime'].min()) / 100000,
            max=unix_time_millis(df['Datetime'].max()) / 100000,
            marks=marksDict,
            value=[unix_time_millis(df['Datetime'].min()) / 100000,
                   unix_time_millis(df['Datetime'].max()) / 100000],
        ),
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        dash_canvas.DashCanvas(
            id='canvas-line',
            width=canvas_width,
            height=canvas_height,
            hide_buttons=['line', 'zoom', 'pan','save','pencil','undo','redo','select','rectangle'],
        ),
        dcc.Slider(
            id='imageSlicer',
            min=0,
            max=10,
            value=0,
        ),
    ], style={'display': 'inline-block', 'width': '49%'}),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='loaded-scan', style={'display': 'none'})
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('datetime_RangeSlider', 'value')])
def update_figure(dateRange):
    print(dateRange)
    filtered_df = df[(df['Datetime'] > datetime.fromtimestamp((dateRange[0] * 100000) / 1000)) &
                     (df['Datetime'] < datetime.fromtimestamp((dateRange[1] * 100000) / 1000))]
    filtered_col = [col for col in filtered_df if col.endswith('DCE')]
    filtered_df = filtered_df[filtered_col]
    traces = []
    for i in filtered_col:
        df_by_structure = filtered_df[i]
        xList = [unix_time_millis(date_obj) / 100000 for date_obj in df_by_structure.index]
        tickLabels = get_marks_from_start_end(df_by_structure.index.min(), df_by_structure.index.max())

        tickLabelList = []
        for key, value in tickLabels.items():
            tickLabelList.append(value)

        traces.append(dict(
            x=xList,
            y=df_by_structure.values.tolist(),
            text=i.strip('_DCE'),
            mode='markers',
            opacity=0.6,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=i.strip('_DCE')
        ))

    return {
        'data': traces,
        'layout': dict(
            xaxis={'type': 'linear', 'title': 'Date Range of Patients',
                   'range': [(unix_time_millis(df_by_structure.index.min()) - (86400 * 3000)) / 100000,
                             (unix_time_millis(df_by_structure.index.max()) + (86400 * 3000)) / 100000], },
            yaxis={'title': 'Volumetric DSC', 'range': [-0.1, 1]},
            margin={'l': 50, 'b': 50, 't': 50, 'r': 50},
            legend={'x': 1, 'y': 1.0},
            hovermode='closest',
            transition={'duration': 250},
            clickmode='event+select',
        )
    }


@app.callback(Output('loaded-scan', 'children'),
              [Input('graph-with-slider', 'clickData')])
def load_data(clickData):
    if clickData is not None:
        utc_value = clickData['points'][0]['x'] * 100000
        dsc_score = clickData['points'][0]['y']
        trace = filtered_col[clickData['points'][0]['curveNumber']]
        curData = df[df[trace] == dsc_score]
        dataFile = curData['basePath'][0] + os.sep + curData['PlanCFileName'][0]
        pixels = sio.loadmat(dataFile)
        volume = pixels['data'][clickData['points'][0]['curveNumber'] + 1, 9]
        initContour = pixels['data'][clickData['points'][0]['curveNumber'] + 1, 7]
        # volume = pixels['data'][1,9]
        subVolume = np.copy(volume)
        np.place(subVolume, subVolume == 1, 0)
        np.place(volume, volume == -1, 0)

        bbox = pixels['data'][clickData['points'][0]['curveNumber'] + 1, 12]
        # bbox = pixels['data'][1, 12]
        bbox = bbox[0, :]
        height = width = 128
        canvas_width = 500
        planCFile = dataFile.replace('data', 'planC')
        planCStruct = sio.loadmat(planCFile)
        scan = planCStruct['planC'][0, 2][0, 0]['scanArray'].astype('uint16')
        scan = scan
        mask = np.zeros(np.shape(scan))
        addedVol = np.zeros(np.shape(scan))
        subVol = np.zeros(np.shape(scan))
        autoC = np.zeros(np.shape(scan))

        scan_norm = normalize_equalize_smooth_CT(scan.transpose(2, 1, 0))
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
        coloredMask[:, :, :, 0] = (mask_arr_add*255).astype('uint8')
        coloredMask[:, :, :, 1] = (mask_arr_auto*255).astype('uint8')
        coloredMask[:, :, :, 2] = (mask_arr_sub*-255).astype('uint8')

        displayImg = np.zeros(np.shape(scan_arr.astype('uint8')))
        for i in range(0, l):

            blended_img = cv2.addWeighted(coloredMask[:, :, i, :].astype('uint8'), 0.2,
                                          scan_arr[:, :, i, :].astype('uint8'), 0.8, 0)
            displayImg[:, :, i, :] = blended_img.astype('uint8')

        serializedData = json.dumps(displayImg.tolist())
    else:
        serializedData = None
    return serializedData


@app.callback(Output('imageSlicer', 'max'),
              [Input('graph-with-slider', 'clickData'),
               Input('loaded-scan', 'children')])
def update_slider_example_max(clickData, serializedData):

    if serializedData is not None:
        reconArray = np.array(json.loads(serializedData))
        print(np.shape(reconArray))
        h, w, l, c = np.shape(reconArray)
        max_value = l
    else:
        max_value = 10
    return max_value


# @app.callback(Output('imageSlicer', 'min'),
#               [Input('graph-with-slider', 'clickData'),
#                Input('loaded-scan', 'children')])
# def update_slider_example_min(clickData, serializedData):
#     if serializedData is not None:
#         reconArray = np.array(json.loads(serializedData))
#         h, w, l, c = np.shape(reconArray)
#         min_value = 0
#     else:
#         min_va
#     return min_value


@app.callback(Output('imageSlicer', 'value'),
              [Input('graph-with-slider', 'clickData'),
               Input('loaded-scan', 'children')])
def update_slider_example_value(clickData, serializedData):
    if serializedData is not None:
        reconArray = np.array(json.loads(serializedData))
        h, w, l, c = np.shape(reconArray)
        value = int(l / 2)
    else:
        value = 5
    return value

#
# @app.callback(Output('imageSlicer', 'marks'),
#               [Input('graph-with-slider', 'clickData'),
#                Input('loaded-scan', 'children')])
# def update_slider_example_marks(clickData, serializedData):
#     reconArray = np.array(json.loads(serializedData))
#     h, w, l, c = np.shape(reconArray)
#     marksDictSlider = {str(slc): str(slc) for slc in range(0, l)}
#     return marksDictSlider


@app.callback(Output('canvas-line', 'image_content'),
              [Input('imageSlicer', 'value'),
               Input('loaded-scan', 'children')])
def update_image_slider(slice_num, serializedData):
    if serializedData is not None:
        reconArray = np.array(json.loads(serializedData))
    else:
        slice_num = 0
        reconArray = np.zeros((128,128,1,3))
    return array_to_data_url(reconArray[:,:,slice_num,:].astype('uint8'))


if __name__ == '__main__':
    PORT = 8000
    ADDRESS = "172.25.132.135"
    app.run_server(debug=True)

                   # port=PORT,
                   # host=ADDRESS)
