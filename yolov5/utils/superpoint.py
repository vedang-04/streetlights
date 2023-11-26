import cv2
import numpy as np

def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_preprocessed = img

    return img_preprocessed, img_orig

def band_func(im0):
    imc = im0.copy()
    M = imc.shape[1] // 3
    print(imc.shape)
    x0 = M
    x1 = M + M
    x2 = M + M + M
    N = imc.shape[0] // 4
    shapes = np.zeros_like(imc, np.uint8)
    cv2.rectangle(shapes, (M, imc.shape[0]), (M + M, 0), (255, 255, 255), cv2.FILLED)
    alpha = 0.5
    mask = shapes.astype(bool)
    im0[mask] = cv2.addWeighted(im0, alpha, shapes, 1 - alpha, 0)[mask]
    cb = [(M, imc.shape[0]), (M + M, 0)]
    im0 = np.expand_dims(im0, 2)
    im0 = im0.astype(np.float32)
    im0 = im0 / 255.
    return im0, cb


def band_func_orig(im0):
    imc = im0.copy()
    M = imc.shape[1] // 3
    x0 = M
    x1 = M + M
    x2 = M + M + M
    N = imc.shape[0] // 4
    shapes = np.zeros_like(imc, np.uint8)
    cv2.rectangle(shapes, (M, imc.shape[0]), (M + M, 0), (255, 255, 255), cv2.FILLED)
    alpha = 0.5
    mask = shapes.astype(bool)
    im0[mask] = cv2.addWeighted(im0, alpha, shapes, 1 - alpha, 0)[mask]
    return im0

def preprocess_coord(box, img0):
    # print(img0.shape)
    W, H = img0.shape[0:2]

    c = []
    
    x, y, w, h = map(float, box[0][0:4])
    l = int((x - w / 2) * W)
    r = int((x + w / 2) * W)
    t = int((y - h / 2) * H)
    b = int((y + h / 2) * H)
    
    if l < 0:
        l = 0
    if r > W - 1:
        r = W - 1
    if t < 0:
        t = 0
    if b > H - 1:
        b = H - 1
    c.append([l, t, r, b])
    
    c = np.array(c, dtype='float')
    return c

def preprocess_file(lab_file, img0):
    print(img0.shape)
    W, H = img0.shape[0:2]
    with open(lab_file, 'r') as f:
        lf = f.readlines()
        c = []
        for line in lf:
            x, y, w, h = map(float, line.strip().split(' ')[1:5])
            l = int((x - w / 2) * W)
            r = int((x + w / 2) * W)
            t = int((y - h / 2) * H)
            b = int((y + h / 2) * H)
            
            if l < 0:
                l = 0
            if r > W - 1:
                r = W - 1
            if t < 0:
                t = 0
            if b > H - 1:
                b = H - 1
            c.append([l, t, r, b])
    
        c = np.array(c, dtype='float')
    return c

def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map, cb, coord, img0,
                                                 keep_k_points=10000):
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    # keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(float(p[1]), float(p[0]), 1) for p in keypoints]

    keypoints1 = []
    keypoints2 = []
    for cord in coord:
        for p in keypoints:
            # print(p.pt)
            # print(cb)
            # print(cord)
            if (p.pt[0] >= cb[0][0] / 3) & (p.pt[0] <= 3 * cb[1][0]) & (p.pt[1] >= cb[1][1]/3) & (p.pt[1] <= 3 * cb[0][1]):
                # print(p.pt)
                # print(cb)
                # print(cord)
                # if (p.pt[0] >= cord[0]/3) & (p.pt[0] <= 3 * cord[2]) & (p.pt[1] >= cord[1]/3) & (
                #         p.pt[1] <= 3 * cord[3]):
                keypoints1.append(p)
                keypoints2.append(list(p.pt)[::-1])


    

    # print(keypoints1)
    keypoints2 = np.array(keypoints2, dtype='int')
    # print(keypoints2)
    if len(keypoints2) == 0:
        return None, None
    desc = descriptor_map[keypoints2[:, 0], keypoints2[:, 1]]
    keypoints = keypoints1

    # img00 = img0.copy()
    # img00 = cv2.drawKeypoints(img0, keypoints, img00, (255, 0, 0))
    # cv2.imwrite('img0.jpg', img00)
    return keypoints, desc

def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    if type(desc1) != None and type(desc2) != None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches_idx = np.array([m.queryIdx for m in matches])
        m_kp1 = [kp1[idx] for idx in matches_idx]
        matches_idx = np.array([m.trainIdx for m in matches])
        m_kp2 = [kp2[idx] for idx in matches_idx]

        return m_kp1, m_kp2, matches
    else:
        raise AssertionError

def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers

