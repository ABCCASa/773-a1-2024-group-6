import concurrent.futures

import numpy as np

def neighbor_circle(x,y):
    return[(x, y-3), (x+1, y-3), (x+2, y-2), (x+3, y-1), (x+3, y), (x+3, y+1), (x+2, y+2), (x+1, y+3), (x, y+3), (x-1, y+3), (x-2, y+2), (x-3, y+1), (x-3, y), (x-3, y-1), (x-2, y-2), (x-1, y-3)]



def detect_corner(circle_value,n=9):
    count=0
    t=30
    state_counter = []
    state="similar"
    for value in circle_value:
        if t <= value:
            if state == "brighter":
                count += 1
            else:
                state_counter.append((state,count))
                state = "brighter"
                count = 1
        elif -t >= value:
            if state == "darker":
                count += 1
            else:
                state_counter.append((state,count))
                state = "darker"
                count = 1
        else:
            if state=="similar":
                count+=1
            else:
                state_counter.append((state,count))
                state="similar"
                count=1
        if(count>=n and state!="similar"):
            return True
    state_counter.append((state, count))
    if state_counter[1][0]==state_counter[-1][0] and state_counter[1][1]+state_counter[-1][1]>=n:
        return True
    else:
        return False

def detect_candidate(angle_point_list,t):
    count_candidate=0
    for node in angle_point_list:
        if abs(node)>t:
            count_candidate+=1
    if count_candidate>=3:
        return True
    else:return False
def candidate_circle(x,y):
    return[(x,y-3),(x+3,y),(x,y+3),(x-3,y)]
def convolution_2d(image, convolution_kernel, pad_mode="edge"):
    img_shape = image.shape
    result = np.zeros(img_shape)
    kernel_size = convolution_kernel.shape[0]
    image = np.pad(image, kernel_size//2, mode=pad_mode)\

    for r in range(kernel_size):
        for c in range(kernel_size):
            result +=image[r:img_shape[0] + r, c:img_shape[1] + c]*convolution_kernel[r, c]
    return result


def fast_corner_detection(image):
    result=[]
    corners=[]
    height, weight = image.shape
    image=np.pad(image,3,mode="edge")
    Score_matrix = np.zeros(image.shape)
    num_process=4
    step=height//num_process
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures=[]
        for i in range(num_process):
            start_y=i*step
            end_y=start_y+ step if i<num_process-1 else height+3
            future=executor.submit(iterate_image,image,start_y,end_y,weight,Score_matrix)
            futures.append(future)
        for future in futures:
            corners_frag,Score_frag=future.result()
            corners+=corners_frag
            Score_matrix+=Score_frag

    window_size = 5
    center_mask = np.ones((window_size, window_size), dtype=bool)
    nms_offset = window_size // 2
    for set in corners:
        row = set[1] + 3
        column = set[0] + 3
        nums_window = Score_matrix[row - nms_offset:row + 1 + nms_offset, column - nms_offset:column + 1 + nms_offset]
        if np.max(nums_window[center_mask]) <= Score_matrix[row][column]:
            result.append((Score_matrix[row][column], (column - 3, row - 3)))

    result.sort(key=lambda x: x[0], reverse=True)
    top_x_corners = [x[1] for x in result[:min(len(result), 1000)]]
    return top_x_corners

def iterate_image(image,start_y,end_y,weight,Score_matrix):
    result=[]
    if start_y<3: start_y=3
    for row in range(start_y,end_y):
        for column in range(3,weight+3):
            value=image[row][column]
            candidate_value=[image[offset[1], offset[0]]-value for offset in candidate_circle(column,row)]
            if detect_candidate(candidate_value,30):
                circle_value=[image[offset[1], offset[0]]-value for offset in neighbor_circle(column,row)]
                if detect_corner(circle_value):
                    result.append((column-3,row-3))
                    Score = np.sum(np.abs(circle_value))
                    Score_matrix[row][column]=Score
    return result,Score_matrix