import numpy as np


def detect_corner(image,x,y,n=9):
    count=0
    t=30
    Score=0
    state_counter = {"darker": 0, "brighter": 0,"similar":0}
    state="similar"
    neighbor=neighbor_circle(x,y)
    end_index=len(neighbor)-1
    state_change=2
    if detect_candidate(image,x,y,[neighbor[0],neighbor[4],neighbor[8],neighbor[12]],t):
        i=0
        while True:
            node=neighbor[i]
            Score += abs(image[x][y] - image[node[0]][node[1]])
            if image[x][y] + t < image[node[0]][node[1]]:
                if state == "brighter":
                    count += 1
                else:
                    state_counter[state] = max(state_counter[state], count)
                    state = "brighter"
                    count = 0
                    state_change-=1
            elif image[x][y] - t > image[node[0]][node[1]]:
                if state == "darker":
                    count += 1
                else:
                    state_counter[state] = max(state_counter[state], count)
                    state = "darker"
                    count = 0
                    state_change-=1
            if (state_counter[state] >= n) or end_index == i:
                break
            if state_change==0:
                end_index=i
            i=(i+1)%len(neighbor)

        if(max(state_counter.values())>=n):
            return True,Score
        else:return False,0
    else:

        return False,0


def detect_candidate(image,x,y,angle_point_list,t):
    count_candidate=0
    for node in angle_point_list:
        if abs(image[x][y]-image[node[0]][node[1]])>t:
            count_candidate+=1
    if count_candidate>=2:
        return True
    else:return False

def neighbor_circle(x,y):
    return[(x, y-3), (x+1, y-3), (x+2, y-2), (x+3, y-1), (x+3, y), (x+3, y+1), (x+2, y+2), (x+1, y+3), (x, y+3), (x-1, y+3), (x-2, y+2), (x-3, y+1), (x-3, y), (x-3, y-1), (x-2, y-2), (x-1, y-3)]
def fast_corner_detection(image):
    corners=[]
    result=[]
    image=np.pad(image,3,mode="edge")

    Score_matrix = np.zeros(image.shape)
    for row in range(3,image.shape[0]-3):
        for column in range(3,image.shape[1]-3):
            is_corner,Score=detect_corner(image,row,column)
            if is_corner:
                corners.append((column-3,row-3))
                Score_matrix[row][column]=Score
    window_size=5
    center_mask = np.ones((window_size, window_size), dtype=bool)
    for set in corners:
        nums_window=Score_matrix[set[1]+3-window_size//2:set[1]+4+window_size//2,set[0]-window_size//2+3:set[0]+4+window_size//2]
        if np.max(nums_window[center_mask])<=Score_matrix[set[1]+3][set[0]+3]:
            result.append((Score_matrix[set[1]+3][set[0]+3],set))
    result.sort(key=lambda x: x[0], reverse=True)
    top_x_corners = [x[1] for x in result[:min(len(result), 1000)]]
    return top_x_corners
