import tkinter as tk
import PathNN as NN


def buttonTrain():
    NN.train()

def getNewInput():
    #Get input
    #put input into list
    
    Result = NN.getSuggestedPath()

    T.insert(tk.END, Result)


def constructWindow():

    


    root = tk.Tk()

    # Create text widget and specify size.
    T = tk.Text(root, height = 5, width = 52, font = 25)
    
    # Create label
    #l = tk.Label(root, text = "Fact of the Day")
    #l.config(font =("Courier", 14))
  
    Fact = """A man can be arrested in
    Italy for wearing a skirt in public."""

    def updatetext():       
         Result = NN.getSuggestedPath()
         T.config(state = tk.NORMAL)
         T.delete('1.0', tk.END)
         T.insert(tk.END, Result)


    canvas = tk.Canvas(root, height = 200, width = 300)

    #frame = tk.Frame(root, bg = "white")
    #frame.place(relwidth = 0.8, relheight = 0.8, relx = 0.1, rely = 0.8)

    #tk.Button(root, text = "Train", command = buttonTrain, height = 5, width = 10).pack(anchor = "w")
    tk.Button(root, text = "getOutput", command = updatetext, height = 5, width = 10).pack(anchor = "w")

   
    canvas.pack()
    #l.pack()
    T.pack()
    T.insert(tk.END, Fact)

    root.mainloop()


