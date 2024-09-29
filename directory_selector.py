
import tkinter as tk
from tkinter import filedialog

def select_directory():
    """Megnyit egy fájlkezelőt, ahol a felhasználó kiválaszthat egy könyvtárat."""
    root = tk.Tk()
    root.withdraw()  # Elrejti a fő ablakot
    directory = filedialog.askdirectory(title="Válaszd ki a képek könyvtárát")
    return directory
