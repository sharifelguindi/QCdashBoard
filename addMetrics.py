from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from shapely.ops import split
import numpy as np
from scipy import stats as spstats
import matplotlib.pyplot as plt
from functions import plot_polygons_and_linestrings
import csv
import pandas as pd
import scipy.io as sio
import os
from skimage import measure
import surface_distance


def compute_metrics(mask_gt, mask_pred, spacing_mm=(3, 0.6, 0.6), surface_tolerances=[1, 3]):
    surface_DSC = []
    DSC = []
    HD_95 = []
    surface_distances = surface_distance.compute_surface_distances(mask_gt.astype(int),
                                                                   mask_pred.astype(int),
                                                                   spacing_mm=spacing_mm)
    DSC.append(surface_distance.compute_dice_coefficient(mask_gt.astype(int), mask_pred.astype(int)))

    for surface_tolerance in surface_tolerances:
        surface_DSC.append(surface_distance.compute_surface_dice_at_tolerance(surface_distances, surface_tolerance))

    HD_95.append(surface_distance.compute_robust_hausdorff(surface_distances, 95))

    return DSC, surface_DSC, HD_95


def mask2poly(mask, slice, pxSize, threshold=0.5):
    contours = measure.find_contours(mask, threshold)
    poly = Polygon()

    for contour in contours:
        pList = []
        if len(contour) > 2 and poly.is_empty == True:
            for point in contour:
                pList.append([point[0] * pxSize[0], point[1] * pxSize[1], slice * pxSize[2]])
            poly = Polygon(pList)
        elif len(contour) > 2 and poly.is_empty == False:
            for point in contour:
                pList.append([point[0] * pxSize[0], point[1] * pxSize[1], slice * pxSize[2]])
            this_ref_polygon = Polygon(pList[:-1])
            if not this_ref_polygon.is_valid:
                this_ref_polygon = this_ref_polygon.buffer(0)

            poly = poly.union(this_ref_polygon)
        else:
            print("mask contained no contour")
    return poly


def get_distance_measures(ref_poly, test_poly, stepsize=1.0, warningsize=1.0):
    # Hausdorff is trivial to compute with Shapely, but average distance requires stepping along each polygon.
    # This is the 'stepsize' in mm. At each point the minimum distance to the other contour is calculated to
    # create a list of distances. From this list the HD can be calculated from this, but it is inaccurate. Therefore,
    # we compare it to the Shapely one and report a problem if the error is greater that 'warningsize' in mm.

    reference_line = ref_poly.boundary
    test_line = test_poly.boundary

    distance_ref_to_test = []
    for distance_along_contour in np.arange(0, reference_line.length, stepsize):
        distance_to_other = reference_line.interpolate(distance_along_contour).distance(test_line)
        distance_ref_to_test.append(distance_to_other)

    distance_test_to_ref = []
    for distance_along_contour in np.arange(0, test_line.length, stepsize):
        distance_to_other = test_line.interpolate(distance_along_contour).distance(reference_line)
        distance_test_to_ref.append(distance_to_other)

    my_hd = np.max([np.max(distance_ref_to_test), np.max(distance_test_to_ref)])
    shapely_hd = test_poly.hausdorff_distance(ref_poly)

    if (my_hd + warningsize < shapely_hd) | (my_hd - warningsize > shapely_hd):
        print('There is a discrepancy between the Hausdorff distance and the list used to calculate the 95% HD')
        print('You may wish to consider a smaller stepsize')

    return distance_ref_to_test, distance_test_to_ref


def get_added_path_length(ref_poly, contracted_poly, expanded_poly, debug=False):
    total_path_length = 0

    reference_boundary = ref_poly.boundary
    if contracted_poly.area > 0:
        contracted_boundary = contracted_poly.boundary
    else:
        contracted_boundary = None
    expanded_boundary = expanded_poly.boundary

    if debug:
        plot_polygons_and_linestrings(reference_boundary, '#000000')
        if contracted_boundary is not None:
            plot_polygons_and_linestrings(contracted_boundary, '#0000ff')
        plot_polygons_and_linestrings(expanded_boundary, '#0000ff')

    if contracted_boundary is not None:
        ref_split_inside = split(reference_boundary, contracted_boundary)
        for line_segment in ref_split_inside:
            # check it the centre of the line is within the contracted polygon
            mid_point = line_segment.interpolate(0.5, True)
            if contracted_poly.contains(mid_point):
                total_path_length = total_path_length + line_segment.length
                if debug:
                    plot_polygons_and_linestrings(line_segment, '#00ff00')
            else:
                if debug:
                    plot_polygons_and_linestrings(line_segment, '#ff0000')

    ref_split_outside = split(reference_boundary, expanded_boundary)
    for line_segment in ref_split_outside:
        # check it the centre of the line is outside the expanded polygon
        mid_point = line_segment.interpolate(0.5, True)
        if not expanded_poly.contains(mid_point):
            total_path_length = total_path_length + line_segment.length
            if debug:
                plot_polygons_and_linestrings(line_segment, '#00ff00')
        else:
            if debug:
                plot_polygons_and_linestrings(line_segment, '#ff0000')

    return total_path_length


def find_and_score_slice_matches(pixels, contourNames, contourArr, tolerance=1):
    result_list = []
    ii = 0
    for contourName in contourNames:
        print(contourName)
        structNum = contourArr[ii] - 1
        gt = pixels['data'][structNum, 8]
        msk = pixels['data'][structNum, 7]
        bbox = pixels['data'][structNum, 12][0, :]
        pxShape = pixels['data'][structNum, 13][0, :]
        pxSize = pixels['data'][structNum, 14][0, :] * 10

        if pxShape.size != 0:

            if bbox[0] <= 0:
                bbox[0] = 1

            if bbox[1] > pxShape[0]:
                bbox[1] = pxShape[0]

            if bbox[2] <= 0:
                bbox[2] = 1

            if bbox[3] > pxShape[1]:
                bbox[3] = pxShape[1]

            if bbox[4] <= 0:
                bbox[4] = 1

            if bbox[5] > pxShape[2]:
                bbox[5] = pxShape[2]

            ref_mask = np.zeros(pxShape)
            ref_mask[bbox[0] - 1:bbox[1], bbox[2] - 1:bbox[3], bbox[4] - 1:bbox[5]] = gt

            test_mask = np.zeros(pxShape)
            test_mask[bbox[0] - 1:bbox[1], bbox[2] - 1:bbox[3], bbox[4] - 1:bbox[5]] = msk

            if np.size(pxSize) == 0:
                pxSize = [0.1171875, 0.1171875, 0.2] * 10
                print("Did not find pixel size information")
            slice_thickness = pxSize[2]

            if bbox[5] > pxShape[2]:
                bbox[5] = pxShape[2]

            total_added_path_length = 0
            total_true_positive_area = 0
            total_false_positive_area = 0
            total_false_negative_area = 0
            total_test_area = 0
            total_ref_area = 0
            distance_ref_to_test = []
            distance_test_to_ref = []
            ref_weighted_centroid_sum = np.array([0, 0, 0])
            test_weighted_centroid_sum = np.array([0, 0, 0])
            structure_name = ''
            ref_polygon_dictionary = {}
            test_polygon_dictionary = {}

            for z_value in range(bbox[4], bbox[5]):
                ref_polygon = None
                if (np.max(ref_mask[:, :, z_value]) > 0) or (np.max(test_mask[:, :, z_value]) > 0):
                    if ref_polygon is None:
                        # Make points into Polygon
                        ref_polygon = mask2poly(ref_mask[:, :, z_value], z_value, pxSize)
                    else:
                        # Turn next set of points into a Polygon
                        this_ref_polygon = mask2poly(ref_mask[:, :, z_value], z_value, pxSize)
                        # Attempt to fix any self-intersections in the resulting polygon
                        if not this_ref_polygon.is_valid:
                            this_ref_polygon = this_ref_polygon.buffer(0)
                        if ref_polygon.contains(this_ref_polygon):
                            # if the new polygon is inside the old one, chop it out
                            ref_polygon = ref_polygon.difference(this_ref_polygon)
                        elif ref_polygon.within(this_ref_polygon):
                            # if the new and vice versa
                            ref_polygon = this_ref_polygon.difference(ref_polygon)
                        else:
                            # otherwise it is a floating blob to add
                            ref_polygon = ref_polygon.union(this_ref_polygon)
                    # Attempt to fix any self-intersections in the resulting polygon
                    if ref_polygon is not None:
                        if not ref_polygon.is_valid:
                            ref_polygon = ref_polygon.buffer(0)
                    ref_polygon_dictionary[z_value] = ref_polygon

            for z_value in range(bbox[4], bbox[5]):
                test_polygon = None
                if (np.max(ref_mask[:, :, z_value]) > 0) | (np.max(test_mask[:, :, z_value]) > 0):
                    if test_polygon is None:
                        # Make points into Polygon
                        test_polygon = mask2poly(test_mask[:, :, z_value], z_value, pxSize)
                    else:
                        # Turn next set of points into a Polygon
                        this_test_polygon = mask2poly(test_mask[:, :, z_value], z_value, pxSize)
                        # Attempt to fix any self-intersections in the resulting polygon
                        if not this_test_polygon.is_valid:
                            this_test_polygon = this_test_polygon.buffer(0)
                        if test_polygon.contains(this_test_polygon):
                            # if the new polygon is inside the old one, chop it out
                            test_polygon = test_polygon.difference(this_test_polygon)
                        elif test_polygon.within(this_test_polygon):
                            # if the new and vice versa
                            test_polygon = this_test_polygon.difference(test_polygon)
                        else:
                            # otherwise it is a floating blob to add
                            test_polygon = test_polygon.union(this_test_polygon)
                    # Attempt to fix any self-intersections in the resulting polygon
                    if test_polygon is not None:
                        if not test_polygon.is_valid:
                            test_polygon = test_polygon.buffer(0)
                    test_polygon_dictionary[z_value] = test_polygon

            # for each slice in ref find corresponding slice in test
            for z_value, refpolygon in ref_polygon_dictionary.items():
                testpolygon = test_polygon_dictionary[z_value]

                if refpolygon.is_empty is False and testpolygon.is_empty is False:
                    debug = False

                    if debug:
                        plot_polygons_and_linestrings(refpolygon, '#00ff00')
                        plt.show()
                        plt.pause(0.1)
                        plt.close()

                    # go get some distance measures
                    # these get added to a big list so that we can calculate the 95% HD
                    [ref_to_test, test_to_ref] = get_distance_measures(refpolygon, testpolygon, 0.05)
                    distance_ref_to_test.extend(ref_to_test)
                    distance_test_to_ref.extend(test_to_ref)

                    # apply tolerance ring margin to test with added path length
                    expanded_poly = cascaded_union(testpolygon.buffer(tolerance, 32, 1, 1))
                    contracted_poly = cascaded_union(testpolygon.buffer(-tolerance, 32, 1, 1))

                    # add intersection of contours
                    contour_intersection = refpolygon.intersection(testpolygon)
                    total_true_positive_area = total_true_positive_area + contour_intersection.area
                    total_false_negative_area = total_false_negative_area + \
                                                (refpolygon.difference(contour_intersection)).area
                    total_false_positive_area = total_false_positive_area + \
                                                (testpolygon.difference(contour_intersection)).area
                    total_test_area = total_test_area + testpolygon.area
                    total_ref_area = total_ref_area + refpolygon.area
                    centroid_point = refpolygon.centroid
                    centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                    ref_weighted_centroid_sum = ref_weighted_centroid_sum + (refpolygon.area * centroid_point_np)
                    centroid_point = testpolygon.centroid
                    centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                    test_weighted_centroid_sum = test_weighted_centroid_sum + (testpolygon.area * centroid_point_np)

                    # add length of remain contours

                    added_path = get_added_path_length(refpolygon, contracted_poly, expanded_poly)
                    total_added_path_length = total_added_path_length + added_path

                elif refpolygon.is_empty is False and testpolygon.is_empty is True:
                    # if no corresponding slice, then add the whole ref length
                    # print('Adding path for whole contour')
                    path_length = refpolygon.length
                    total_added_path_length = total_added_path_length + path_length
                    # also the whole slice is false negative
                    total_false_negative_area = total_false_negative_area + refpolygon.area
                    total_ref_area = total_ref_area + refpolygon.area
                    centroid_point = refpolygon.centroid
                    centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                    ref_weighted_centroid_sum = ref_weighted_centroid_sum + (refpolygon.area * centroid_point_np)

                else:
                    total_false_positive_area = total_false_positive_area + testpolygon.area
                    total_test_area = total_test_area + testpolygon.area
                    centroid_point = testpolygon.centroid
                    centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                    test_weighted_centroid_sum = test_weighted_centroid_sum + (testpolygon.area * centroid_point_np)

                # now we need to deal with the distance lists to work out the various distance measures
                # NOTE: these are different calculations to those used in plastimatch. The book chapter will explain all...

            ref_centroid = ref_weighted_centroid_sum / total_ref_area
            test_centroid = test_weighted_centroid_sum / total_ref_area

            if len(distance_ref_to_test) > 1:
                hd = np.max([np.max(distance_ref_to_test), np.max(distance_test_to_ref)])
                ninety_five_hd = np.max(
                    [np.percentile(distance_ref_to_test, 95), np.percentile(distance_test_to_ref, 95)])
                ave_dist = (np.mean(distance_ref_to_test) + np.mean(distance_test_to_ref)) / 2
                median_dist = (np.median(distance_ref_to_test) + np.median(distance_test_to_ref)) / 2
            else:
                hd = 0
                ninety_five_hd = 0
                ave_dist = 0
                median_dist = 0

            tau = [1, 3]
            DSC, surface_DSC, HD_95 = compute_metrics(ref_mask, test_mask, spacing_mm=pxSize, surface_tolerances=tau)

            result_list.append((contourName,
                                [total_added_path_length, total_true_positive_area * slice_thickness,
                                 total_false_negative_area * slice_thickness,
                                 total_false_positive_area * slice_thickness, total_ref_area * slice_thickness,
                                 total_test_area * slice_thickness,
                                 hd, ninety_five_hd, ave_dist, median_dist, ref_centroid, test_centroid, DSC,
                                 surface_DSC[0], surface_DSC[1], HD_95]))
            ii = ii + 1
        else:
            ii = ii + 1

    return result_list


def estimate_slice_thickness(contour_data_set):
    # this is a crude attempt to estimate the slice thickness without loading the image
    # we assume that the slices are equally spaced, and if we collect unique slice positions
    # for enough slices with contours then the modal difference will represent the slice thickness

    z_list = []
    z_diff_list = []

    for contour_set in contour_data_set.ROIContourSequence:
        for contour_slice in contour_set.ContourSequence:
            contour_points = contour_slice.ContourData
            z_list.append(contour_points[2])

    z_list = np.unique(z_list)
    z_list = np.sort(z_list)

    old_z_val = z_list[0]
    for z_val in z_list:
        z_diff = z_val - old_z_val
        old_z_val = z_val
        z_diff_list.append(z_diff)

    slice_thickness = spstats.mode(z_diff_list)[0][0]

    print('slice thickness: ', slice_thickness)

    return slice_thickness


def score_case(pixels, contourNames, contourArr, output_filename=''):
    resultlist = find_and_score_slice_matches(pixels, contourNames, contourArr, 1)

    auto_contour_measures = []

    for result in resultlist:
        organname, scores = result
        # scores[0] APL
        # scores[1] TP volume
        # scores[2] FN volume
        # scores[3] FP volume
        # scores[4] Ref volume
        # scores[5] Test volume
        # scores[6] Hausdorff
        # scores[7] 95% Hausdorff
        # scores[8] Average Distance
        # scores[9] Median Distance
        # scores[10] Reference Centroid
        # scores[11] Test Centroid
        # scores[12] VDSC: from Nikolav
        # scores[13] SDSC: 1mm tolerance
        # scores[14] SDSC: 3mm tolerance
        # scores[15] Robust 95% Hausdorff

        results_structure = {'Organ': organname, 'APL': scores[0], 'TPVol': scores[1], 'FNVol': scores[2],
                             'FPVol': scores[3], 'SEN': scores[1] / scores[4], 'SFP': scores[3] / scores[5],
                             'three_D_DSC': 2 * scores[1] / (scores[4] + scores[5]), 'HD': scores[6],
                             'ninety_five_HD': scores[7], 'AD': scores[8], 'MD': scores[9], 'ref_cent': scores[10],
                             'test_cent': scores[11], 'VDSC': scores[12], 'SDSC_1mm': scores[13],
                             'SDSC_3mm': scores[14], 'RobustHD_95': scores[15]}
        auto_contour_measures.append(results_structure)

    if output_filename != '':
        print('Writing results to: ', output_filename)
        with open(output_filename, mode='w', newline='\n', encoding='utf-8') as out_file:
            result_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(
                ['Organ', 'APL', 'TP volume', 'FN volume', 'FP volume', 'SEN', '%FP', '3D DSC', '2D HD',
                 '95% 2D HD', 'Ave 2D Dist', 'Median 2D Dist', 'Reference Centroid', 'Test Centroid',
                 'VDSC', 'SDSC_1mm', 'SDSC_3mm', 'RobustHD_95'])
            for results_structure in auto_contour_measures:
                result_writer.writerow([results_structure['Organ'],
                                        results_structure['APL'],
                                        results_structure['TPVol'],
                                        results_structure['FNVol'],
                                        results_structure['FPVol'],
                                        results_structure['SEN'],
                                        results_structure['SFP'],
                                        results_structure['three_D_DSC'],
                                        results_structure['HD'],
                                        results_structure['ninety_five_HD'],
                                        results_structure['AD'],
                                        results_structure['MD'],
                                        results_structure['ref_cent'],
                                        results_structure['test_cent'],
                                        results_structure['VDSC'],
                                        results_structure['SDSC_1mm'],
                                        results_structure['SDSC_3mm'],
                                        results_structure['RobustHD_95'],
                                        ])
    else:
        # TODO function could take list of parameters of measures we want to return.
        return auto_contour_measures


def main():
    # Find measures contour comparison. First is "ground truth", second is "test"
    databaseFile = 'G:\\Projects\\AutoQC\\prostateDB\\prostateDB.xlsx'
    df = pd.read_excel(databaseFile)

    for k in range(8, len(df)):
        print(k)
        if isinstance(df['basePath'][k], str):
            dataFile = df['basePath'][k] + os.sep + df['dataFilename'][k]
            output_file = dataFile.replace('data', 'metric').replace('.mat', '.csv')
            contourNames = [col for col in df if col.split(',')[0].endswith('_DSC')]
            contourArr = []
            i = 0
            for contourName in contourNames:
                name, number = contourName.split(',')
                contourNames[i] = name.replace('_DSC', '')
                contourArr.append(int(number))
                i = i + 1
            pixels = sio.loadmat(dataFile)
            score_case(pixels, contourNames, contourArr, output_file)


if __name__ == '__main__':
    main()
