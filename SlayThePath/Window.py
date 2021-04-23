import tkinter as tk
import PathNN as NN


def buttonTrain():
    NN.train()

def getNewInput():
    NN.predict()

def constructWindow():
    root = tk.Tk()

    canvas = tk.Canvas(root, height = 400, width = 300)

    frame = tk.Frame(root, bg = "white")
    frame.place(relwidth = 0.8, relheight = 0.8, relx = 0.1, rely = 0.8)

    tk.Button(root, text = "Train", command = buttonTrain, height = 5, width = 10).pack(anchor = "w")
    tk.Button(root, text = "getOutput", command = getNewInput, height = 5, width = 10).pack(anchor = "w")

    canvas.pack()
    root.mainloop()


