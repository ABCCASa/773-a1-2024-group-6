import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

def select_line(img, corners):
    root = tk.Tk()
    root.resizable(False, False)
    window_width = 1000

    img = Image.fromarray(img)
    img_width, img_height = img.width, img.height
    ratio = window_width / img_width
    img_width, img_height = int(img_width * ratio), int(img_height * ratio)
    img = ImageTk.PhotoImage(img.resize((img_width, img_height)))
    canvas = tk.Canvas(root, width=img_width, height=img_height, bg="black")
    canvas.place(x=0, y=0, anchor="nw")
    canvas.create_image(0, 0, image=img, anchor="nw")
    root.geometry(f"{img_width+200}x{img_height}")

    corner_groups = list()
    temp_corner_group = dict()
    corner_dict = dict()
    used_corner = list()

    def on_corner_click(event):
        item_id = canvas.find_withtag(tk.CURRENT)[0]
        if item_id not in used_corner:
            if item_id in temp_corner_group.keys():
                temp_corner_group.pop(item_id)
                canvas.itemconfig(item_id, fill="red")
            else:
                temp_corner_group[item_id] = corner_dict[item_id]
                canvas.itemconfig(item_id, fill="blue")




    for corner in corners:
        r = 2
        x, y = corner[0]*ratio, corner[1]*ratio
        corner_id = canvas.create_oval(x-r, y-r, x+r, y+r, fill="red", outline="")
        canvas.tag_bind(corner_id, "<Button-1>", on_corner_click)
        corner_dict[corner_id] = corner

    lines_id = []
    def add_group():
        if len(temp_corner_group) > 1:
            temp_corners = list(temp_corner_group.values())
            corner_groups.append(temp_corners)
            for key in temp_corner_group.keys():
                used_corner.append(key)
                canvas.itemconfig(key, fill='green')
            temp_corner_group.clear()
            for i in range(1, len(temp_corners)):
                x1, y1 = temp_corners[i-1]
                x2, y2 = temp_corners[i]
                line_id = canvas.create_line(x1*ratio, y1*ratio, x2*ratio, y2*ratio, fill="green", width=2)
                lines_id.append(line_id)

    tk.Button(root, text="Add Group", command=add_group).place(x=img_width+100, y=40, anchor=CENTER)

    def clear_all():
        for key in temp_corner_group.keys():
            canvas.itemconfig(key, fill='red')
        for id in used_corner:
            canvas.itemconfig(id, fill='red')
        for id in lines_id:
            canvas.delete(id)
        lines_id.clear()
        used_corner.clear()
        temp_corner_group.clear()
        corner_groups.clear()

    tk.Button(root, text="Clear All", command=clear_all).place(x=img_width+100, y=200, anchor=CENTER)
    tk.Button(root, text="Apply", command=lambda:root.destroy()).place(x=img_width + 100, y=80, anchor=CENTER)
    root.mainloop()

    return corner_groups