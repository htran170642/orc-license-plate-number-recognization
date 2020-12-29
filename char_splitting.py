import cv2


def remove_boundary(img_bin):
    img_bin[:10, :10] = 0
    contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x in (0, img_bin.shape[1]-w) or y in (0, img_bin.shape[0]-h):
            cv2.drawContours(img_bin, [contour], 0, (0, 0, 0), -1)

    return img_bin


def get_box(img_bin):
    img = cv2.morphologyEx(img_bin,
                           cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box = []
    min_w, max_w = 1/15, 1/7
    min_h, max_h = 0.25, 0.45
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_w <= (w / img_bin.shape[1]) <= max_w and min_h <= (h / img_bin.shape[0]) <= max_h:
            tl = (x-1, y-1)
            br = (x+w-1, y+h-1)
            bounding_box.append((tl, br))
    
    return bounding_box


def sort_bounding_box(bounding_box):
    tl_list = [b[0] for b in bounding_box]
    y_list = [p[1] for p in tl_list]
    ymean = sum(y_list) / len(y_list)
    row = [0 if y < ymean else 1 for y in y_list]
    col = [p[0] for p in tl_list]
    
    sort_key = [r*1000 + c for r, c in zip(row, col)]
    sort_idx = sorted(range(len(bounding_box)), key=lambda i: sort_key[i])
    bounding_box_sorted = [bounding_box[i] for i in sort_idx]
    
    row = [row[i] for i in sort_idx]
    col = [col[i] for i in sort_idx]
    numbers = [r != 0 or c <= 40 or c >= 55 for r, c in zip(row, col)]
    
    return bounding_box_sorted, numbers


def preprocess(img):
    img_resized = cv2.resize(img, (0, 0), fx=80/img.shape[1], fy=80/img.shape[1])
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    img_bin = (img_bin != 0).astype('uint8')
    edge = (cv2.Canny(img_gray, 100, 200) != 0).astype('uint8')
    img_rmbound = remove_boundary(img_bin & (1-edge))
    bounding_box = get_box(img_rmbound | edge)
    bounding_box, numbers = sort_bounding_box(bounding_box)
    
    return  img_resized, img_bin, edge, img_rmbound, bounding_box, numbers