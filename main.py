# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import cv2 as cv2
import imutils
import numpy as np

# refs
# networkx https://networkx.org/documentation/stable/tutorial.html#
# osmnx https://geoffboeing.com/publications/osmnx-complex-street-networks/
# https://www.geeksforgeeks.org/template-matching-using-opencv-in-python/
import osmnx.io

import connection


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def unpack_gpx(gpx_file) -> (list[float], list[float]):
    gpx = gpxpy.parse(gpx_file)

    print("Getting name: " + gpx.tracks[0].name)

    lat = []
    long = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                # print('Point at {0},{1}) -> {2}'.format(point.latitude, point.longitude, point.elevation))
                lat.append(point.latitude)
                long.append(point.longitude)
    return lat, long


def unpack_gpx_generic(gpx_file) -> (list[float], list[float]):
    gpx = gpxpy.parse(gpx_file)

    #print("Getting name: " + gpx.tracks[0].name)

    lat = []
    long = []
    for waypoint in gpx.waypoints:
        #print(waypoint.latitude)
        #print(waypoint.longitude)
        #for segment in track.segments:
        #    for point in segment.points:
                # print('Point at {0},{1}) -> {2}'.format(point.latitude, point.longitude, point.elevation))
        lat.append(waypoint.latitude)
        long.append(waypoint.longitude)
    return lat, long



def save_gpx_file_as_png(gpx_filepath: str, output_png_filepath: str):
    gpx_file = open(gpx_filepath, 'r')
    (lat, long) = unpack_gpx(gpx_file)
    plt.scatter(x=long, y=lat, s=2) # s is width
    plt.axis('off')
    plt.savefig(output_png_filepath)


def save_gpx_file_as_png_generic(gpx_filepath: str, output_png_filepath: str):
    gpx_file = open(gpx_filepath, 'r')
    (lat, long) = unpack_gpx_generic(gpx_file)
    plt.scatter(x=long, y=lat, s=2) # s is width
    plt.axis('off')
    plt.savefig(output_png_filepath)

def image_bw_to_cleaned_np_array(image_in):
    image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image)
    #print("Datatype: ",image.dtype)
    normalized_image = np.where(image>0, 1, image)
    #print("Datatype (norm): ", normalized_image.dtype)
    return normalized_image  #/ 255 # [num / 255 for num in image]


def calculate_lower_and_upper_bounds_scale(template_width, image_width_miles, run_distance_miles, orig_image_width):
    # figure out how much we should scale stuff
    #  > the original code kept the template constant. i'm fine with that.
    #  > we want to try to scale the width_template to match width_image
    #  > template only gives us total distance.
    #     > this can be a straight line (largest width) to a convoluted ball of yarn (smallest width)
    #     > if dist is the entire distance, the width can be:
    #         > width_template_miles (upper bound): dist
    #         > width_template_miles (lower bound): if it were perfect circle and 2 laps: --> dist/(2*pi) ~ dist/6.28
    #             (because pi*(2*radius) = distance)

    # now:
    #   we want width_template / width_template_miles ~= width_image / image_width_miles
    #
    #   width_image is the one that we would actually change.
    #     (width_template * image_width_miles) / width_template_miles ~= width_image
    #     (width_template * image_width_miles) / run_dist <= width_image <= 7 (width_template * image_width*miles) / run_dist
    lowerbound = template_width * image_width_miles / run_distance_miles
    lowerbound_scale = lowerbound / orig_image_width
    upperbound_scale = lowerbound_scale * 8
    return (lowerbound_scale, upperbound_scale)

    # input file should be white background, black foreground
def run_template_matching_for_map(image_filepath: str, template_filepath: str, image_width_miles: float, run_distance_miles: float):
    # derived from: https://www.tutorialspoint.com/template-matching-using-opencv-in-python#:~:text=Template%20matching%20using%20OpenCV%20in%20Python%201%20Steps,...%206%20Example%20code%20...%207%20Output%20

    # Open template and get canny
    template_raw = cv2.imread(template_filepath)
    template = image_bw_to_cleaned_np_array(template_raw)
    template_height, template_width = template.shape

    # open the main image and convert it to gray scale image
    main_image = cv2.imread(image_filepath)
    gray_image = image_bw_to_cleaned_np_array(main_image)
    orig_image_height, orig_image_width = gray_image.shape

    # calculate lower and upper bounds scale
    (lower_bound_scale, upper_bound_scale) = calculate_lower_and_upper_bounds_scale(template_width, image_width_miles, run_distance_miles, orig_image_width)

    #lower_bound_scale=0.3
    #upper_bound_scale=.6
    print("Lower scale: ", lower_bound_scale)
    print("Upper scale: ", upper_bound_scale)



    temp_found = None
    for scale in np.linspace(lower_bound_scale, upper_bound_scale, 200)[::-1]:


        # resize the image and store the ratio
        resized_img = imutils.resize(gray_image, width=int(orig_image_width * scale))
        resized_height, resized_width = resized_img.shape
        ratio = orig_image_width / float(resized_width)
        if resized_height < template_height or resized_width < template_width:
            print("Resized height: ", resized_height, "Resized width: ", resized_width, " Scale: ", scale)
            print("Breaking")
            break

        # Convert to edged image for checking
        #e = cv2.Canny(resized_img, 10, 25)

        #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        match = cv2.matchTemplate(resized_img, template, cv2.TM_CCOEFF)
        (_, val_max, _, loc_max) = cv2.minMaxLoc(match)
        print("Scale: ", scale)
        #val_max, loc_max = compare_map_matching_test(resized_img, template)

        if temp_found is None or val_max > temp_found[0]:
            print("Scale: ", scale, " Val: ", val_max, " Ratio: ", ratio, "Width: ", orig_image_width * scale,
                  " Loc: ", loc_max)
            temp_found = (val_max, loc_max, ratio, scale)


    # Get information from temp_found to compute x,y coordinate
    (_, loc_max, r, scale) = temp_found
    (x_start, y_start) = (int(loc_max[0]), int(loc_max[1]))
    (x_end, y_end) = (int((loc_max[0] + template_width)), int((loc_max[1] + template_height)))

    # stack the
    print("Final scale (of main image): ", scale)
    print("Tempalte height: ", template_height)
    print("Tempalte width: ", template_width)
    print("Y: ", y_start, " to ", y_end)
    print("X: ", x_start, " to ", x_end)
    resized_img = imutils.resize(gray_image, width=int(orig_image_width * scale))

    layered_img = resized_img
    layered_img[y_start:y_end, x_start:x_end] += (template*6)

    print("Layered max: ", np.amax(layered_img))

    # factor up so max is 255
    layered_img = (layered_img*32)

    # Draw rectangle around the template
    cv2.rectangle(layered_img, (x_start, y_start), (x_end, y_end), (153, 22, 0), 5)

    window_name = 'Template found'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 600)

    cv2.imshow(window_name, layered_img)
    cv2.waitKey(0)



def test_filter():
    image = np.array([[0, 1, 0, 1],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]], np.uint8)

    filter = np.array([[0,1],
                       [1,0]], np.uint8)

    compare_map_matching_test(image,filter)

def test_filter_on_imgs(image_filepath, template_filepath):
    # Open template and get canny
    template_raw = cv2.imread(template_filepath)
    template = image_bw_to_cleaned_np_array(template_raw)

    # open the main image and convert it to gray scale image
    main_image = cv2.imread(image_filepath)
    gray_image = image_bw_to_cleaned_np_array(main_image)

    #cv2.matchTemplate(gray_image, template, cv2.TM_CCORR)
    compare_map_matching_test(gray_image, template)


def transform_stravarun(source_image_filepath,output_filepath: str):
    # these may need to be changed, depending on snapshot
    channel_to_use = 1
    threshold = 40



    image = cv2.imread(source_image_filepath)
    image_channel = image[:,:,channel_to_use]
    scrubbed_map = np.where(image_channel < threshold, 1, image_channel)
    scrubbed_map = np.where(scrubbed_map >= threshold, 0, scrubbed_map)

    inverted_image = cv2.bitwise_not(scrubbed_map * 255)
    cv2.imshow("img", inverted_image)
    cv2.waitKey(0)

    cv2.imwrite(output_filepath,inverted_image)



def transform_heatmap(source_image_filepath, output_filepath):
    threshold = 50
    image = cv2.imread(source_image_filepath, 0)

    normalized_image = np.where(image < threshold, 0, image)
    normalized_image = np.where(normalized_image >= threshold, 1, normalized_image)

    print(normalized_image)
    print(normalized_image.shape)
    print(np.amax(normalized_image))
    cv2.imshow("image", normalized_image*255)
    cv2.waitKey(0)

    inverted_image = cv2.bitwise_not(normalized_image*255)
    cv2.imwrite(output_filepath, inverted_image)

    '''cv2.imshow("r", image[:,:,0])
    cv2.waitKey(0)
    cv2.imshow("g", image[:, :, 1])
    cv2.waitKey(0)
    cv2.imshow("b", image[:, :, 2])
    cv2.waitKey(0)'''


def compare_map_matching_test(source_image_in, template_image_in):

    # we can assume these are all np arrays.
    # we can also
    height_source, width_source = source_image_in.shape
    height_template, width_template = template_image_in.shape

    print("Source image shape: ", source_image_in.shape)
    print("Template image shape: ", template_image_in.shape)

    # all we need to do is get the output of us applying the template as a filter on the source image
    #out = cv2.filter2D(src=source_image_in, kernel=template_image_in, ddepth=-1, borderType=cv2.BORDER_CONSTANT)
    out = cv2.matchTemplate(source_image_in, template_image_in, cv2.TM_CCORR)
    (_, val_max, _, loc_max) = cv2.minMaxLoc(out)
    print("Out:")
    print(out)
    print("Max out: ", np.amax(out))
    cv2.imshow("out", out / np.amax(out))
    cv2.waitKey(0)

    #(_, val_max, _, loc_max) = cv2.minMaxLoc(small_out)
    x_start, y_start = loc_max


    #do a straight addition of the arrays
    layered_img = source_image_in
    layered_img[y_start:y_start+height_template, x_start:x_start+width_template] += template_image_in
    # factor up so max is 255
    layered_img = layered_img * 128
    cv2.imshow("temp", layered_img)
    cv2.waitKey(0)


    print("Val max: ", val_max, " Loc max: ", loc_max)

    # layer on top


## GENERIC IMAGE PROCESSING FNS ##

def invert_image(input_filepath: str, output_filepath: str):
    image = cv2.imread(input_filepath, 0)
    inverted_image = cv2.bitwise_not(image)
    cv2.imwrite(output_filepath, inverted_image)

def clean_image_for_matching(input_filepath: str, output_filepath: str):
    # https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
    image = cv2.imread(input_filepath)

    # gray it
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white_image) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite("temp_img.png", black_and_white_image)

    remove_white_from_image("temp_img.png", output_filepath)


def remove_white_from_image(in_filepath: str, out_filepath: str):
    # https://stackoverflow.com/questions/55673060/how-to-set-white-pixels-to-transparent-using-opencv
    # read image in
    img = cv2.imread(in_filepath)

    # get image dimensions
    h, w, c = img.shape

    # append alpha channel (require for BGRA)
    img_bgra = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)

    # create mask
    white = np.all(img == [255, 255, 255], axis=-1)

    img_bgra[white, -1] = 0

    cv2.imwrite(out_filepath, img_bgra)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # connection.initial_connect("5ae79f48779ac81af597f5e84c9819e85bc6b203")
    # connection.test_pull_activities()
    # connection.connectToStrava()

    # G = ox.graph_from_place("Madison, WI", network_type='drive')
    # ox.plot_graph(G)

    # osmnx.io.save_graphml(G, "madison_wi.graphml")

    # G = osmnx.io.load_graphml("madison_wi.graphml")
    # ox.plot_graph(G, node_size=0, edge_linewidth=0.2, save=True, filepath="madison_map.png", dpi=600)

    # print("DOne with OX")

    # Viewing the GPX file
    #save_gpx_file_as_png('testgpx/Easy_run.gpx', 'easyrun.png')
    # gpx_file = open('testgpx/Easy_run.gpx', 'r')
    # (lat, long) = unpack_gpx(gpx_file)
    # plt.scatter(x=long, y=lat, s=2) # s is width
    # plt.axis('off')
    # plt.savefig("easyrun.png")

    #save_gpx_file_as_png_generic("testgpx/run2.gpx", "run2.png")
    #gpx_file = open('testgpx/run.gpx', 'r')
    #(lat, long) = unpack_gpx(gpx_file)
    #plt.scatter(x=long, y=lat)
    #plt.waitforbuttonpress()

    # plt.rcParams["figure.figsize"] = (50, 35)

    # plt.show()

    # remove_white_from_image("easyrun.png", "easyrun_out.png")
    run_template_matching_for_map("madison_heatmap_strava_bw.png", "lunchrun_8dec22_bw.png", 2, 4)
    #test_filter_on_imgs("madison_map_cropped_bw.png", "easyrun_bw.png")
    '''template_filepath = "easyrun_bw.png"
    template_raw = cv2.imread(template_filepath)
    template = image_bw_to_cleaned_np_array(template_raw)
    print(template)
    '''
    #test_filter()

    #transform_heatmap("madison_heatmap_strava.png","madison_heatmap_strava_bw.png")
    transform_stravarun("run_22nov22.png", "run_22nov22_bw.png")

    # test filter
    #test_filter()


    #template = cv2.imread("madison_map_cropped_bw.png")
    #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #inverted_image = cv2.bitwise_not(template)
    ##print(inverted_image)
    #inverted_image_np = np.array(inverted_image)
    #inverted_image_np = inverted_image_np/255
    #print(np.amax(inverted_image_np))
    #print(template.shape)

    #for y in range(0,template.shape[0]):
    #    print(template[y,10])




    '''
    main_image = cv2.imread("madison_map_cropped_bw.png")
    main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    scale = 0.4555
    y_start = 48
    y_end = 528
    x_start = 570
    x_end = 1210
    template = cv2.imread("madison_map_cropped_bw.png")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    resized_img = imutils.resize(main_image, width=int(main_image.shape[1] * scale))

    cv2.rectangle(main_image, (x_start, y_start), (x_end, y_end), (153, 22, 0), 5)

    window_name = 'Template found'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 600)

    cv2.imshow(window_name, main_image)
    cv2.waitKey(0)
    '''


    # temp stuff
    #clean_image_for_matching("madison_map_cropped_inverted.png", "madison_map_cropped_bw.png")
    #clean_image_for_matching("easyrun_out.png", "easyrun_bw.png")
    print("stopping")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
