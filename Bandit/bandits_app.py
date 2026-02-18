import tkinter as tk
from bandit_gui import BanditGUI

def main():
    root = tk.Tk()
    BanditGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()