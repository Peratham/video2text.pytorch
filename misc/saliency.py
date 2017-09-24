import math
import numpy as np
import networkx as nx
import scipy.spatial.distance
import scipy.signal
import skimage
import skimage.io
from skimage.util import img_as_float
import scipy.spatial.distance
import scipy.signal
import skimage
import skimage.io
from skimage.segmentation import slic


def raster_scan(img, L, U, D):
    n_rows = len(img)
    n_cols = len(img[0])

    for x in range(1, n_rows - 1):
        for y in range(1, n_cols - 1):
            ix = img[x][y]
            d = D[x][y]

            u1 = U[x - 1][y]
            l1 = L[x - 1][y]

            u2 = U[x][y - 1]
            l2 = L[x][y - 1]

            b1 = max(u1, ix) - min(l1, ix)
            b2 = max(u2, ix) - min(l2, ix)

            if d <= b1 and d <= b2:
                continue
            elif b1 < d and b1 <= b2:
                D[x][y] = b1
                U[x][y] = max(u1, ix)
                L[x][y] = min(l1, ix)
            else:
                D[x][y] = b2
                U[x][y] = max(u2, ix)
                L[x][y] = min(l2, ix)

    return True


def raster_scan_inv(img, L, U, D):
    n_rows = len(img)
    n_cols = len(img[0])

    for x in range(n_rows - 2, 1, -1):
        for y in range(n_cols - 2, 1, -1):

            ix = img[x][y]
            d = D[x][y]

            u1 = U[x + 1][y]
            l1 = L[x + 1][y]

            u2 = U[x][y + 1]
            l2 = L[x][y + 1]

            b1 = max(u1, ix) - min(l1, ix)
            b2 = max(u2, ix) - min(l2, ix)

            if d <= b1 and d <= b2:
                continue
            elif b1 < d and b1 <= b2:
                D[x][y] = b1
                U[x][y] = max(u1, ix)
                L[x][y] = min(l1, ix)
            else:
                D[x][y] = b2
                U[x][y] = max(u2, ix)
                L[x][y] = min(l2, ix)

    return True


def mbd(img, num_iters):

    if len(img.shape) != 2:
        print('did not get 2d np array to fast mbd')
        return None
    if (img.shape[0] <= 3 or img.shape[1] <= 3):
        print('image is too small')
        return None

    L = np.copy(img)
    U = np.copy(img)
    D = float('Inf') * np.ones(img.shape)
    D[0, :] = 0
    D[-1, :] = 0
    D[:, 0] = 0
    D[:, -1] = 0

    # unfortunately, iterating over numpy arrays is very slow
    img_list = img.tolist()
    L_list = L.tolist()
    U_list = U.tolist()
    D_list = D.tolist()

    for x in range(0, num_iters):
        if x % 2 == 1:
            raster_scan(img_list, L_list, U_list, D_list)
        else:
            raster_scan_inv(img_list, L_list, U_list, D_list)

    return np.array(D_list)


def get_saliency_mbd(img, method='b'):

    # Saliency map calculation based on: Minimum Barrier Salient Object
    # Detection at 80 FPS

    if isinstance(img, str):
        img = skimage.io.imread(img)
    img_mean = np.mean(img, axis=(2))
    sal = mbd(img_mean, 3)

    if method == 'b':
        # get the background map

        # paper uses 30px for an image of size 300px, so we use 10%
        (n_rows, n_cols, n_channels) = img.shape
        img_size = math.sqrt(n_rows * n_cols)
        border_thickness = int(math.floor(0.1 * img_size))

        img_lab = img_as_float(skimage.color.rgb2lab(img))

        px_left = img_lab[0:border_thickness, :, :]
        px_right = img_lab[n_rows - border_thickness - 1:-1, :, :]

        px_top = img_lab[:, 0:border_thickness, :]
        px_bottom = img_lab[:, n_cols - border_thickness - 1:-1, :]

        px_mean_left = np.mean(px_left, axis=(0, 1))
        px_mean_right = np.mean(px_right, axis=(0, 1))
        px_mean_top = np.mean(px_top, axis=(0, 1))
        px_mean_bottom = np.mean(px_bottom, axis=(0, 1))

        px_left = px_left.reshape((n_cols * border_thickness, 3))
        px_right = px_right.reshape((n_cols * border_thickness, 3))

        px_top = px_top.reshape((n_rows * border_thickness, 3))
        px_bottom = px_bottom.reshape((n_rows * border_thickness, 3))

        cov_left = np.cov(px_left.T)
        cov_right = np.cov(px_right.T)

        cov_top = np.cov(px_top.T)
        cov_bottom = np.cov(px_bottom.T)

        cov_left = np.linalg.inv(cov_left)
        cov_right = np.linalg.inv(cov_right)

        cov_top = np.linalg.inv(cov_top)
        cov_bottom = np.linalg.inv(cov_bottom)

        u_left = np.zeros(sal.shape)
        u_right = np.zeros(sal.shape)
        u_top = np.zeros(sal.shape)
        u_bottom = np.zeros(sal.shape)

        u_final = np.zeros(sal.shape)
        img_lab_unrolled = img_lab.reshape(img_lab.shape[0] * img_lab.shape[1], 3)

        px_mean_left_2 = np.zeros((1, 3))
        px_mean_left_2[0, :] = px_mean_left

        u_left = scipy.spatial.distance.cdist(
            img_lab_unrolled, px_mean_left_2, 'mahalanobis', VI=cov_left)
        u_left = u_left.reshape((img_lab.shape[0], img_lab.shape[1]))

        px_mean_right_2 = np.zeros((1, 3))
        px_mean_right_2[0, :] = px_mean_right

        u_right = scipy.spatial.distance.cdist(
            img_lab_unrolled, px_mean_right_2, 'mahalanobis', VI=cov_right)
        u_right = u_right.reshape((img_lab.shape[0], img_lab.shape[1]))

        px_mean_top_2 = np.zeros((1, 3))
        px_mean_top_2[0, :] = px_mean_top

        u_top = scipy.spatial.distance.cdist(
            img_lab_unrolled, px_mean_top_2, 'mahalanobis', VI=cov_top)
        u_top = u_top.reshape((img_lab.shape[0], img_lab.shape[1]))

        px_mean_bottom_2 = np.zeros((1, 3))
        px_mean_bottom_2[0, :] = px_mean_bottom

        u_bottom = scipy.spatial.distance.cdist(
            img_lab_unrolled, px_mean_bottom_2, 'mahalanobis', VI=cov_bottom)
        u_bottom = u_bottom.reshape((img_lab.shape[0], img_lab.shape[1]))

        max_u_left = np.max(u_left)
        max_u_right = np.max(u_right)
        max_u_top = np.max(u_top)
        max_u_bottom = np.max(u_bottom)

        u_left = u_left / max_u_left
        u_right = u_right / max_u_right
        u_top = u_top / max_u_top
        u_bottom = u_bottom / max_u_bottom

        u_max = np.maximum(np.maximum(np.maximum(u_left, u_right), u_top), u_bottom)

        u_final = (u_left + u_right + u_top + u_bottom) - u_max

        u_max_final = np.max(u_final)
        sal_max = np.max(sal)
        sal = sal / sal_max + u_final / u_max_final

        # postprocessing

        # apply centredness map
        sal = sal / np.max(sal)

        s = np.mean(sal)
        alpha = 50.0
        delta = alpha * math.sqrt(s)

        xv, yv = np.meshgrid(np.arange(sal.shape[1]), np.arange(sal.shape[0]))
        (w, h) = sal.shape
        w2 = w / 2.0
        h2 = h / 2.0

        C = 1 - np.sqrt(np.power(xv - h2, 2) + np.power(yv - w2, 2)) / \
            math.sqrt(np.power(w2, 2) + np.power(h2, 2))

        sal = sal * C

        # increase bg/fg contrast

        def f(x):
            b = 10.0
            return 1.0 / (1.0 + math.exp(-b * (x - 0.5)))

        fv = np.vectorize(f)

        sal = sal / np.max(sal)

        sal = fv(sal)

        return sal * 255.0


def get_saliency_ft(img):

    # Saliency map calculation based on:

    if isinstance(img, str):
        img = skimage.io.imread(img)

    img_rgb = img_as_float(img)

    img_lab = skimage.color.rgb2lab(img_rgb)

    mean_val = np.mean(img_rgb, axis=(0, 1))

    kernel_h = (1.0 / 16.0) * np.array([[1, 4, 6, 4, 1]])
    kernel_w = kernel_h.transpose()

    blurred_l = scipy.signal.convolve2d(img_lab[:, :, 0], kernel_h, mode='same')
    blurred_a = scipy.signal.convolve2d(img_lab[:, :, 1], kernel_h, mode='same')
    blurred_b = scipy.signal.convolve2d(img_lab[:, :, 2], kernel_h, mode='same')

    blurred_l = scipy.signal.convolve2d(blurred_l, kernel_w, mode='same')
    blurred_a = scipy.signal.convolve2d(blurred_a, kernel_w, mode='same')
    blurred_b = scipy.signal.convolve2d(blurred_b, kernel_w, mode='same')

    im_blurred = np.dstack([blurred_l, blurred_a, blurred_b])

    sal = np.linalg.norm(mean_val - im_blurred, axis=2)
    sal_max = np.max(sal)
    sal_min = np.min(sal)
    sal = 255 * ((sal - sal_min) / (sal_max - sal_min))
    return sal


def S(x1, x2, geodesic, sigma_clr=10):
    return math.exp(-pow(geodesic[x1, x2], 2) / (2 * sigma_clr * sigma_clr))


def compute_saliency_cost(smoothness, w_bg, wCtr):
    n = len(w_bg)
    A = np.zeros((n, n))
    b = np.zeros((n))

    for x in range(0, n):
        A[x, x] = 2 * w_bg[x] + 2 * (wCtr[x])
        b[x] = 2 * wCtr[x]
        for y in range(0, n):
            A[x, x] += 2 * smoothness[x, y]
            A[x, y] -= 2 * smoothness[x, y]

    x = np.linalg.solve(A, b)

    return x


def path_length(path, G):
    dist = 0.0
    for i in range(1, len(path)):
        dist += G[path[i - 1]][path[i]]['weight']
    return dist


def make_graph(grid):
    # get unique labels
    vertices = np.unique(grid)

    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices, np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)

    # create edges
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges, axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:, 0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x % num_vertices],
              vertices[int(x / num_vertices)]] for x in edges]

    return vertices, edges


def get_saliency_rbd(img):

    # Saliency map calculation based on:
    # Saliency Optimization from Robust Background Detection, Wangjiang Zhu,
    # Shuang Liang, Yichen Wei and Jian Sun, IEEE Conference on Computer
    # Vision and Pattern Recognition (CVPR), 2014

    if isinstance(img, str):
        img = skimage.io.imread(img)

    if len(img.shape) != 3:  # got a grayscale image
        img = skimage.color.gray2rgb(img)

    img_lab = img_as_float(skimage.color.rgb2lab(img))

    img_rgb = img_as_float(img)

    img_gray = img_as_float(skimage.color.rgb2gray(img))

    segments_slic = slic(img_rgb, n_segments=250, compactness=10,
                         sigma=1, enforce_connectivity=False)

    num_segments = len(np.unique(segments_slic))

    nrows, ncols = segments_slic.shape
    max_dist = math.sqrt(nrows * nrows + ncols * ncols)

    grid = segments_slic

    (vertices, edges) = make_graph(grid)

    gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]

    centers = dict()
    colors = dict()
    distances = dict()
    boundary = dict()

    for v in vertices:
        centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]
        colors[v] = np.mean(img_lab[grid == v], axis=0)

        x_pix = gridx[grid == v]
        y_pix = gridy[grid == v]

        if np.any(x_pix == 0) or np.any(y_pix == 0) or np.any(x_pix == nrows - 1) or np.any(y_pix == ncols - 1):
            boundary[v] = 1
        else:
            boundary[v] = 0

    G = nx.Graph()

    # buid the graph
    for edge in edges:
        pt1 = edge[0]
        pt2 = edge[1]
        color_distance = scipy.spatial.distance.euclidean(colors[pt1], colors[pt2])
        G.add_edge(pt1, pt2, weight=color_distance)

    # add a new edge in graph if edges are both on boundary
    for v1 in vertices:
        if boundary[v1] == 1:
            for v2 in vertices:
                if boundary[v2] == 1:
                    color_distance = scipy.spatial.distance.euclidean(
                        colors[v1], colors[v2])
                    G.add_edge(v1, v2, weight=color_distance)

    geodesic = np.zeros((len(vertices), len(vertices)), dtype=float)
    spatial = np.zeros((len(vertices), len(vertices)), dtype=float)
    smoothness = np.zeros((len(vertices), len(vertices)), dtype=float)
    adjacency = np.zeros((len(vertices), len(vertices)), dtype=float)

    sigma_clr = 10.0
    sigma_bndcon = 1.0
    sigma_spa = 0.25
    mu = 0.1

    all_shortest_paths_color = nx.shortest_path(
        G, source=None, target=None, weight='weight')

    for v1 in vertices:
        for v2 in vertices:
            if v1 == v2:
                geodesic[v1, v2] = 0
                spatial[v1, v2] = 0
                smoothness[v1, v2] = 0
            else:
                geodesic[v1, v2] = path_length(all_shortest_paths_color[v1][v2], G)
                spatial[v1, v2] = scipy.spatial.distance.euclidean(
                    centers[v1], centers[v2]) / max_dist
                smoothness[
                    v1, v2] = math.exp(- (geodesic[v1, v2] * geodesic[v1, v2]) / (2.0 * sigma_clr * sigma_clr)) + mu

    for edge in edges:
        pt1 = edge[0]
        pt2 = edge[1]
        adjacency[pt1, pt2] = 1
        adjacency[pt2, pt1] = 1

    for v1 in vertices:
        for v2 in vertices:
            smoothness[v1, v2] = adjacency[v1, v2] * smoothness[v1, v2]

    area = dict()
    len_bnd = dict()
    bnd_con = dict()
    w_bg = dict()
    ctr = dict()
    wCtr = dict()

    for v1 in vertices:
        area[v1] = 0
        len_bnd[v1] = 0
        ctr[v1] = 0
        for v2 in vertices:
            d_app = geodesic[v1, v2]
            d_spa = spatial[v1, v2]
            w_spa = math.exp(- ((d_spa) * (d_spa)) / (2.0 * sigma_spa * sigma_spa))
            area_i = S(v1, v2, geodesic)
            area[v1] += area_i
            len_bnd[v1] += area_i * boundary[v2]
            ctr[v1] += d_app * w_spa
        bnd_con[v1] = len_bnd[v1] / math.sqrt(area[v1])
        w_bg[v1] = 1.0 - math.exp(- (bnd_con[v1] * bnd_con[v1]) /
                                  (2 * sigma_bndcon * sigma_bndcon))

    for v1 in vertices:
        wCtr[v1] = 0
        for v2 in vertices:
            d_app = geodesic[v1, v2]
            d_spa = spatial[v1, v2]
            w_spa = math.exp(- (d_spa * d_spa) / (2.0 * sigma_spa * sigma_spa))
            wCtr[v1] += d_app * w_spa * w_bg[v2]

    # normalise value for wCtr

    min_value = min(wCtr.values())
    max_value = max(wCtr.values())

    minVal = [key for key, value in wCtr.items() if value == min_value]
    maxVal = [key for key, value in wCtr.items() if value == max_value]

    for v in vertices:
        wCtr[v] = (wCtr[v] - min_value) / (max_value - min_value)

    img_disp1 = img_gray.copy()
    img_disp2 = img_gray.copy()

    x = compute_saliency_cost(smoothness, w_bg, wCtr)

    for v in vertices:
        img_disp1[grid == v] = x[v]

    img_disp2 = img_disp1.copy()
    sal = np.zeros((img_disp1.shape[0], img_disp1.shape[1], 3))

    sal = img_disp2
    sal_max = np.max(sal)
    sal_min = np.min(sal)
    sal = 255 * ((sal - sal_min) / (sal_max - sal_min))

    return sal


def binarise_saliency_map(saliency_map, method='adaptive', threshold=0.5):

    # check if input is a numpy array
    if type(saliency_map).__module__ != np.__name__:
        print('Expected numpy array')
        return None

    # check if input is 2D
    if len(saliency_map.shape) != 2:
        print('Saliency map must be 2D')
        return None

    if method == 'fixed':
        return (saliency_map > threshold)

    elif method == 'adaptive':
        adaptive_threshold = 1.0 * saliency_map.mean()
        return (saliency_map > adaptive_threshold)

    elif method == 'clustering':
        print('Not yet implemented')
        return None

    else:
        print("Method must be one of fixed, adaptive or clustering")
        return None
