import tkinter as tk

def TestFunction():

    print("Button PRESSED")

def constructWindow():
    root = tk.Tk()

    canvas = tk.Canvas(root, height = 400, width = 300)

    frame = tk.Frame(root, bg = "white")
    frame.place(relwidth = 0.8, relheight = 0.8, relx = 0.1, rely = 0.8)

    tk.Button(root, text = "TestKnapp", command = TestFunction, height = 5, width = 10).pack(anchor = "w")

    canvas.pack()
    root.mainloop()


