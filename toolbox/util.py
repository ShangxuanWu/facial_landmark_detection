import os, pdb, glob, cv2
import numpy as np

dpi = 80
color_set = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w', 'lime', 'cyan', 'aqua']
color_set_big = ['aqua', 'azure', 'red', 'black', 'blue', 'brown', 'cyan', 'darkblue', 'fuchsia', 'gold', 'green', 'grey', 'indigo', 'magenta', 'lime', 'yellow', 'white', 'tomato', 'salmon']
marker_set = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
hatch_set = [None, 'o', '/', '\\', '|', '-', '+', '*', 'x', 'O', '.']
linestyle_set = ['-', '--', '-.', ':', None, ' ', 'solid', 'dashed']

def isstring(string_test):
	return isinstance(string_test, basestring)

def fileparts(pathname, debug=True):
	'''
	this function return a tuple, which contains (directory, filename, extension)
	if the file has multiple extension, only last one will be displayed
	'''
	pathname = safepath(pathname)
	if len(pathname) == 0:
		return ('', '', '')
	if pathname[-1] == '/':
		if len(pathname) > 1:
			return (pathname[:-1], '', '')	# ignore the final '/'
		else:
			return (pathname, '', '')	# ignore the final '/'
	directory = os.path.dirname(os.path.abspath(pathname))
	filename = os.path.splitext(os.path.basename(pathname))[0]
	ext = os.path.splitext(pathname)[1]
	return (directory, filename, ext)


def is_path_valid(pathname):
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isstring(pathname) or not pathname:
            return False
    except TypeError:
        return False
    else:
        return True


def safepath(pathname):
    '''
    convert path to a normal representation
    '''
    assert is_path_valid(pathname), 'path is not valid: %s' % pathname
    return os.path.normpath(pathname)


def is_path_exists(pathname):
    '''
	this function is to justify is given path existing or not
    '''
    try:
        return is_path_valid(pathname) and os.path.exists(pathname)
    except OSError:
        return False


def load_txt_file(file_path, debug=True):
    '''
    load data or string from text file
    '''
    file_path = safepath(file_path)
    if debug:
        assert is_path_exists(file_path), 'text file is not existing at path: %s!' % file_path

    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    num_lines = len(data)
    file.close()
    return data, num_lines


def is_path_creatable(pathname):
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.

    For folder, it needs the previous level of folder existing
    for file, it needs the folder existing
    '''
    if is_path_valid(pathname) is False:
    	return False
    
    pathname = safepath(pathname)
    pathname = os.path.dirname(os.path.abspath(pathname))
    
    # recursively to find the root existing
    while not is_path_exists(pathname):     
        pathname_new = os.path.dirname(os.path.abspath(pathname))
        if pathname_new == pathname:
            return False
        pathname = pathname_new
    return os.access(pathname, os.W_OK)

def is_path_exists_or_creatable(pathname):
    '''
	this function is to justify is given path existing or creatable
    '''
    try:
        return is_path_valid(pathname) and (os.path.exists(pathname) or is_path_creatable(pathname))
    except OSError:
        return False

def isfolder(pathname):
    if is_path_valid(pathname):
        pathname = safepath(pathname)
        if pathname == './':
            return True
        name = os.path.splitext(os.path.basename(pathname))[0]
        ext = os.path.splitext(pathname)[1]
        return len(name) > 0 and len(ext) == 0
    else:
        return False

def mkdir_if_missing(pathname):
	pathname = safepath(pathname)
	assert is_path_exists_or_creatable(pathname), 'input path is not valid or creatable: %s' % pathname
	dirname, _, _ = fileparts(pathname)

	if not is_path_exists(dirname):
		mkdir_if_missing(dirname)

	if isfolder(pathname) and not is_path_exists(pathname):
		os.mkdir(pathname)

def fullfile(*args):
	# result_path = "."
	# for a in args:
	# 	result_path =  result_path + '/' + a
	# return result_path
    result_path = ""
    for a in args:
        result_path =  result_path + '/' + a
    return result_path[1:] 

def isInteger(x):
	return isinstance(x, (int, long))

# TODO
def isIntegerImage(x):
	return True

# TODO
def isFloatImage(x):
	return True

def isext(ext_test):
    return isstring(ext_test) and ext_test[0] == '.'

def isdict(dict_test):
    return isinstance(dict_test, dict)

def islist(list_test):
    return isinstance(list_test, list)

def islogical(logical_test):
    return isinstance(logical_test, bool)

def isinteger(integer_test):
    return isinstance(integer_test, int)

def isnparray(nparray_test):
    return isinstance(nparray_test, np.ndarray)

def is2dptsarray(pts_test):
    return isnparray(pts_test) and pts_test.shape[0] == 2 and len(pts_test.shape) == 2 and pts_test.shape[1] >= 0

def is2dptsarray_occlusion(pts_test):
    return isnparray(pts_test) and pts_test.shape[0] == 3 and len(pts_test.shape) == 2 and pts_test.shape[1] >= 0

def load_list_from_file(file_path):
    '''
    this function reads list from a txt file
    '''
    file_path = safepath(file_path)
    _, _, extension = fileparts(file_path)
    assert extension == '.txt', 'File doesn''t have valid extension.'
    file = open(file_path, 'r')
    assert file != -1, 'datalist not found'

    fulllist = file.read().splitlines()
    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)
    file.close()

    return fulllist, num_elem

def string2ext_filter(string, debug=True):
    '''
    convert a string to an extension filter
    '''
    if debug:
        assert isstring(string), 'input should be a string'

    if isext(string):
        return string
    else:
        return '.' + string

def load_list_from_folder(folder_path, ext_filter=None, depth=1, recursive=False, sort=True, save_path=None):
    '''
    load a list of files or folders from a system path

    parameter:
        folder_path: root to search 
        ext_filter: a string to represent the extension of files interested
        depth: maximum depth of folder to search, when it's None, all levels of folders will be searched
        recursive: 
            False: only return current level
            True: return all levels till to the depth
    '''
    folder_path = safepath(folder_path)
    assert isfolder(folder_path), 'input folder path is not correct: %s' % folder_path
    if not is_path_exists(folder_path):
        return [], 0
    assert islogical(recursive), 'recursive should be a logical variable'
    assert (isinteger(depth) and depth >= 1) or depth is None, 'input depth is not correct {}'.format(depth)
    #pdb.set_trace()
    assert ext_filter is None or (islist(ext_filter) and all(isstring(ext_tmp) for ext_tmp in ext_filter)) or isstring(ext_filter), 'extension filter is not correct'
    if isstring(ext_filter):    # convert to a list
        ext_filter = [ext_filter]

    fulllist = list()
    if depth is None:        # find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                wildcard = os.path.join(wildcard_prefix, '*' + string2ext_filter(ext_tmp))
                curlist = glob2.glob(os.path.join(folder_path, wildcard))
                if sort:
                    curlist = sorted(curlist)
                fulllist += curlist

        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort:
                curlist = sorted(curlist)
            fulllist += curlist
    else:                    # find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth-1):
            wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                wildcard = wildcard_prefix + string2ext_filter(ext_tmp)
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort:
                    curlist = sorted(curlist)
                fulllist += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            if sort:
                curlist = sorted(curlist)
            fulllist += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth-1, recursive=True)
            fulllist += newlist

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)

    # save list to a path
    if save_path is not None:
        save_path = safepath(save_path)
        assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in fulllist:
                file.write('%s\n' % item)
        file.close()

    return fulllist, num_elem

def file_exist(path):
    return os.path.isfile(path)

def matlab_style_find(array, func):
    return [i for (i, val) in enumerate(array) if func(val)]

def is_numpy_int_array(array):
    return array.dtype == 'int64' or array.dtype == 'int32' or array.dtype == 'int16' or array.dtype == 'int8'

def is_list_int_array(lst):
    return all(isinstance(n, int) for n in lst)

def remove_str_from_str(src_str, substr, debug_mode):
    valid = (src_str.find(substr) != -1)
    removed = src_str.replace(substr, '')
    pre_part = src_str[0:valid-1] if (valid > -1) else ''
    pos_part = src_str[valid+len(substr):] if (valid < len(src_str) -1) else '' 
    return removed, valid, pre_part, pos_part

def rotate_bound(image, angle):
    # angle is counter_clockwise
    if angle == -90:
        # rotate clockwise
        return np.rot90(image, 3)
    else:
        return np.rot90(image)
        # rotate counter-clockwise

# bicubic
def resize_portion(img, portion):
    height, width = img.shape[:2]
    img_ = cv2.resize(img,(int(portion*width), int(portion*height)), interpolation = cv2.INTER_CUBIC)
    return img_

def resize_portion_bilinear(img, portion):
    height, width = img.shape[:2]
    img_ = cv2.resize(img,(int(portion*width), int(portion*height)), interpolation = cv2.INTER_LINEAR)
    return img_

def check_imageorPath(img, debug_mode):
    if isstring(img):
        return cv2.imread(img).astype(float)
    return img

def is2dPtsInside(pts_test, im_size, debug_mode = False):

    isValid = True

    im_h = im_size[0]
    im_w = im_size[1]


    x = pts_test[0];
    y = pts_test[1];
    if x >= 0 and x < im_w and y >= 0 and y < im_h:
        isValid = True
    else:
        isValid = False

    return isValid

def preprocess_image_caffe(img, mean_value, debug_mode=True):
    # input is h w c
    # output is c h w
    if debug_mode:
        img = check_imageorPath(img);
        assert isIntegerImage(img), 'image should be in integer format.'
        assert length(mean_value) == 1 or length(mean_value) or 3, 'mean value should be length 1 or 3!'
        assert all(mean_value <= 1.0 and mean_value >= 0.0), 'mean value should be in range [0, 1].'


    img_out = img/255.0
    img_out = img_out - mean_value
    #img_out = np.transpose(img_out, [1,0,2])    # permute to width x height x channel
    img_out = np.transpose(img_out, [2,0,1])    # permute to channel x height x width
    
    if img_out.shape[0] == 1:
        img_out[1, :, :] = img_out[0, :, :]    # broadcast to color image
        img_out[2, :, :] = img_out[0, :, :]   
    
    # python + opencv is BGR itself
    #img_out = img_out[[2,1,0], :, :]       # swap channel to bgr
    return img_out

def load_2dmatrix_from_file(src_path, delimiter=' ', dtype='float32', debug=True):
    src_path = safepath(src_path)
    if debug:
        assert is_path_exists(src_path), 'txt path is not correct at %s' % src_path

    data = np.loadtxt(src_path, delimiter=delimiter, dtype=dtype)
    return data

def save_2dmatrix_to_file(data, save_path, formatting='%.1f', debug=True):
    save_path = safepath(save_path)
    if debug:
        assert isnparray(data) and len(data.shape) == 2, 'input data is not 2d numpy array'
        assert is_path_exists_or_creatable(save_path), 'save path is not correct'
        mkdir_if_missing(save_path)
        # assert isnparray(data) and len(data.shape) <= 2, 'the data is not correct'

    # if len(data.shape) == 2:
        # num_elem = data.size
        # data = np.reshape(data, (1, num_elem))
    # if 
    np.savetxt(save_path, data, delimiter=' ', fmt=formatting)

def saferotation_angle(rotation_angle):
    '''
    ensure the rotation is in [-180, 180] in degree
    '''
    while rotation_angle > 180:
        rotation_angle -= 360

    while rotation_angle < -180:
        rotation_angle += 360

    return rotation_angle

def pts_rotate2D(pts_array, rotation_angle, im_height, im_width, debug=True):
    '''
    rotate the point array in 2D plane counter-clockwise

    parameters:
        pts_array:          2 x num_pts
        rotation_angle:     e.g. 90

    return
        pts_array:          2 x num_pts
    '''
    if debug:
        assert is2dptsarray(pts_array), 'the input point array does not have a good shape'

    rotation_angle = saferotation_angle(rotation_angle)             # ensure to be in [-180, 180]

    if rotation_angle > 0:
        cols2rotated = im_width
        rows2rotated = im_width
    else:
        cols2rotated = im_height
        rows2rotated = im_height
    rotation_matrix = cv2.getRotationMatrix2D((cols2rotated/2, rows2rotated/2), rotation_angle, 1)         # 2 x 3
    
    num_pts = pts_array.shape[1]
    pts_rotate = np.ones((3, num_pts), dtype='float32')             # 3 x num_pts
    pts_rotate[0:2, :] = pts_array         

    return np.dot(rotation_matrix, pts_rotate)         # 2 x num_pts

def save_vis_close_helper(fig=None, ax=None, vis=False, save_path=None, debug=True, closefig=True):
    # save and visualization
    if save_path is not None:
        if debug:
            assert is_path_exists_or_creatable(save_path) and isfile(save_path), 'save path is not valid: %s' % save_path
            mkdir_if_missing(save_path)
        fig.savefig(save_path, dpi=dpi, transparent=True)
    if vis:
        plt.show()

    if closefig:
        plt.close(fig)
        return None, None
    else:
        return fig, ax

def save_vis_close_helper(fig=None, ax=None, vis=False, save_path=None, debug=True, closefig=True):
    # save and visualization
    if save_path is not None:
        if debug:
            assert is_path_exists_or_creatable(save_path) and isfile(save_path), 'save path is not valid: %s' % save_path
            mkdir_if_missing(save_path)
        fig.savefig(save_path, dpi=dpi, transparent=True)
    if vis:
        plt.show()

    if closefig:
        plt.close(fig)
        return None, None
    else:
        return fig, ax

def visualize_image(image_path, vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize various images

    parameters:
        image_path:         a path to an image / an image / a list of images / a list of image paths
    '''
    if debug:
        if isstring(image_path):
            assert is_path_exists(image_path), 'image is not existing at %s' % image_path
        else:
            assert islist(image_path) or isimage(image_path, debug=debug), 'the input is not a list or an good image'

    if isstring(image_path):            # a path to an image
        try:
            image = imread(image_path)
        except IOError:
            print('path is not a valid image path. Please check: %s' % image_path)
            return
    elif islist(image_path):            # a list of images / a list of image paths
        imagelist = image_path
        save_path_list = save_path
        if vis:
            print('visualizing a list of images:')
        if save:
            print('saving a list of images')
            if debug:
                assert islist(save_path_list), 'for saving a list of images, please provide a list of saving path'
                assert all(is_path_exists_or_creatable(save_path_tmp) and isfile(save_path_tmp) for save_path_tmp in save_path_list), 'save path is not valid'
                assert len(save_path_list) == len(imagelist), 'length of list for saving path and data is not equal'
        index = 0
        for image_tmp in imagelist:
            print('processing %d/%d' % (index+1, len(imagelist)))
            if save:
                visualize_image(image_tmp, vis=vis, save_path=save_path[i], save=save, debug=debug)
            else:
                visualize_image(image_tmp, vis=vis, debug=debug)
            index += 1
        return
    else:                               # an image
        if ispilimage(image_path):
            image = np.array(image_path)
        else:
            image = image_path

    width = image.shape[1]
    height = image.shape[0]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # display image
    if iscolorimage(image, debug=debug):
        if debug:
            print 'visualizing color image'
        ax.imshow(image, interpolation='nearest')
    elif isgrayimage(image, debug=debug):
        if debug:
            print 'visualizing grayscale image'
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.reshape(image, image.shape[:-1])

        if isfloatimage(image, debug=debug) and all(item == 1.0 for item in image.flatten().tolist()):
            if debug:
                print('all elements in image are 1. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 0.00001
        elif isuintimage(image, debug=debug) and all(item == 255 for item in image.flatten().tolist()):
            if debug:
                print('all elements in image are 255. For visualizing, we subtract the top left with an epsilon value')
            image[0, 0] -= 1
        ax.imshow(image, interpolation='nearest', cmap='gray')
    else:
        assert False, 'image is not correct'
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_pts_array(pts_array, covariance=False, color_index=0, fig=None, ax=None, pts_size=20, label=False, label_list=None, occlusion=True, vis_threshold=-10000, save_path=None, vis=False, debug=True, closefig=True):
    '''
    plot keypoints

    parameters:
        pts_array:      2 or 3 x num_pts, the third channel could be confidence or occlusion
    '''

    fontsize = 20
    std = None
    conf = 0.95
    if islist(color_index):
        if debug:
            assert not occlusion, 'the occlusion is not compatible with plotting different colors during scattering'
            assert not covariance, 'the covariance is not compatible with plotting different colors during scattering'
        color_tmp = [color_set_big[index_tmp] for index_tmp in color_index]
    else:
        color_tmp = color_set_big[color_index]
    num_pts = pts_array.shape[1]

    if is2dptsarray(pts_array):    
        ax.scatter(pts_array[0, :], pts_array[1, :], color=color_tmp, s=pts_size)

        if debug and islist(color_tmp):
            assert len(color_tmp) == pts_array.shape[1], 'number of points to plot is not equal to number of colors provided'
        pts_visible_index = range(pts_array.shape[1])
        pts_ignore_index = []
        pts_invisible_index = []
    else:
        num_float_elements = np.where(np.logical_and(pts_array[2, :] > 0, pts_array[2, :] < 1))[0].tolist()
        if len(num_float_elements) > 0:
            type_3row = 'conf'
            if debug:
                print('third row is confidence')
        else:
            type_3row = 'occu'
            if debug:
                print('third row is occlusion')

        if type_3row == 'occu':
            pts_visible_index   = np.where(pts_array[2, :] == 1)[0].tolist()              # plot visible points in red color
            pts_invisible_index = np.where(pts_array[2, :] == 0)[0].tolist()              # plot invisible points in blue color
            pts_ignore_index    = np.where(pts_array[2, :] == -1)[0].tolist()             # do not plot points with annotation
        else:
            pts_visible_index   = np.where(pts_array[2, :] > vis_threshold)[0].tolist()
            pts_ignore_index    = np.where(pts_array[2, :] <= vis_threshold)[0].tolist()
            pts_invisible_index = []

        if debug and islist(color_tmp):
            assert len(color_tmp) == len(pts_visible_index), 'number of points to plot is not equal to number of colors provided'

        ax.scatter(pts_array[0, pts_visible_index], pts_array[1, pts_visible_index], color=color_tmp, s=pts_size)
        if occlusion:
            ax.scatter(pts_array[0, pts_invisible_index], pts_array[1, pts_invisible_index], color=color_set_big[(color_index+1) % len(color_set_big)], s=pts_size)
        # else:
            # ax.scatter(pts_array[0, pts_invisible_index], pts_array[1, pts_invisible_index], color=color_tmp, s=pts_size)
        if covariance:
            visualize_pts_covariance(pts_array[0:2, :], std=std, conf=conf, fig=fig, ax=ax, debug=False, color=color_tmp)

    if label:
        for pts_index in xrange(num_pts):
            label_tmp = label_list[pts_index]
            if pts_index in pts_ignore_index:
                continue
            else:
                # note that the annotation is based on the coordinate instead of the order of plotting the points, so the orider in pts_index does not matter
                if islist(color_index):
                    plt.annotate(label_tmp, xy=(pts_array[0, pts_index], pts_array[1, pts_index]), xytext=(-1, 1), color=color_set_big[(color_index[pts_index]+5) % len(color_set_big)], textcoords='offset points', ha='right', va='bottom', fontsize=fontsize)
                else:
                    plt.annotate(label_tmp, xy=(pts_array[0, pts_index], pts_array[1, pts_index]), xytext=(-1, 1), color=color_set_big[(color_index+5) % len(color_set_big)], textcoords='offset points', ha='right', va='bottom', fontsize=fontsize)
                # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                # arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_image_with_pts(image_path, pts, pts_size=20, label=False, label_list=None, color_index=0, vis=False, save_path=None, debug=True, closefig=True):
    '''
    visualize image and plot keypoints on top of it

    parameter:
        image_path:     a path to an image / an image
        pts:            a dictionary or 2 or 3 x num_pts numpy array
                        when there are 3 channels in pts, the third one denotes the occlusion flag
                        occlusion: 0 -> invisible, 1 -> visible, -1 -> not existing
        label:          determine to add text label for each point
        label_list:     label string for all points
        color_index:    a scalar or a list of color indexes
    '''
    fig, ax = visualize_image(image_path, vis=False, debug=debug, closefig=False)

    if label and (label_list is None):
        if not isdict(pts):
            num_pts = pts.shape[1]
        else:
            pts_tmp = pts.values()[0]
            num_pts = np.asarray(pts_tmp).shape[1] if islist(pts_tmp) else pts_tmp.shape[1]
        label_list = [str(i+1) for i in xrange(num_pts)];

    if debug:
        assert not islist(image_path), 'this function only support to plot points on one image'
        if isdict(pts):
            for pts_tmp in pts.values():
                if islist(pts_tmp):
                    pts_tmp = np.asarray(pts_tmp)
                assert is2dptsarray(pts_tmp) or is2dptsarray_occlusion(pts_tmp), 'input points within dictionary are not correct: (2 (3), num_pts) vs %s' % print_np_shape(pts_tmp)
        else:
            assert is2dptsarray(pts) or is2dptsarray_occlusion(pts), 'input points are not correct'
        assert islogical(label), 'label flag is not correct'
        if label:
            assert islist(label_list) and all(isstring(label_tmp) for label_tmp in label_list), 'labels are not correct'

    if isdict(pts):
        color_index = color_index
        for pts_id, pts_array in pts.items():
            if islist(pts_array):
                pts_array = np.asarray(pts_array)
            visualize_pts_array(pts_array, fig=fig, ax=ax, color_index=color_index, pts_size=pts_size, label=label, label_list=label_list, occlusion=False, debug=debug, closefig=False)
            color_index += 1
    else:   
        visualize_pts_array(pts, fig=fig, ax=ax, color_index=color_index, pts_size=pts_size, label=label, label_list=label_list, occlusion=False, debug=debug, closefig=False)

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

if __name__ == "__main__":
	# for debugging
	#print(fullfile('aa', 'bb'))
	print(isInteger(1))
	pass