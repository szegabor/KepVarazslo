
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

class PhotoManagerGUI:
    def __init__(self, app):
        """A grafikus felület inicializálása."""
        self.app = app
        self.root = tk.Tk()
        self.root.title("Photo Manager")

        # Gomb a könyvtár kiválasztásához
        self.select_button = tk.Button(self.root, text="Képek kiválasztása", command=self.select_images)
        self.select_button.pack()

        # Lista a képek megjelenítésére és kiválasztására
        self.image_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE)
        self.image_listbox.pack(fill=tk.BOTH, expand=True)

        # Képjavító gomb
        self.enhance_button = tk.Button(self.root, text="Kiválasztott képek javítása", command=self.enhance_image)
        self.enhance_button.pack()

        # Képek kombinálása gomb
        self.combine_button = tk.Button(self.root, text="Kiválasztott képek kombinálása", command=self.combine_images)
        self.combine_button.pack()

        # Képek mentése gomb
        self.save_button = tk.Button(self.root, text="Képek mentése", command=self.save_images)
        self.save_button.pack()

        # Listaelem kiválasztás esemény
        self.image_listbox.bind('<<ListboxSelect>>', self.show_preview)

    def select_images(self):
        """Képek kiválasztása egy könyvtárból."""
        directory = filedialog.askdirectory(title="Válaszd ki a képek könyvtárát")
        if directory:
            self.app.album = self.app.load_album(directory)
            print(f"Képek betöltve a könyvtárból: {directory}")
            self.update_image_listbox()  # Frissíti a listboxot a képekkel
        else:
            print("Nem választottál könyvtárat.")

    def update_image_listbox(self):
        """Frissíti a listboxot a betöltött képekkel."""
        if self.app.album:
            self.image_listbox.delete(0, tk.END)  # Törli a régi elemeket
            for photo_name in self.app.album.get_photos().keys():
                self.image_listbox.insert(tk.END, photo_name)  # Hozzáadja a fájlneveket

    def get_selected_images(self):
        """Lekérdezi a kiválasztott képek nevét."""
        selected_indices = self.image_listbox.curselection()  # Lekéri a kiválasztott indexeket
        selected_names = [self.image_listbox.get(i) for i in selected_indices]
        return selected_names

    def enhance_image(self):
        """A kiválasztott képek javítása."""
        selected_images = self.get_selected_images()
        if not selected_images:
            messagebox.showwarning("Nincs kiválasztott kép", "Válassz ki egy vagy több képet a javításhoz!")
            return

        for photo_name in selected_images:
            image = self.app.album.get_photos()[photo_name]
            enhanced_image = self.app.enhancer.enhance_contrast(image)
            print(f"{photo_name} kontrasztja javítva.")

    def combine_images(self):
        """A kiválasztott képek kombinálása."""
        selected_images = self.get_selected_images()
        if not selected_images:
            messagebox.showwarning("Nincs kiválasztott kép", "Válassz ki egy vagy több képet a kombináláshoz!")
            return

        photos = self.app.album.get_photos()
        images = [photos[name] for name in selected_images]
        combined_image = self.app.processor.combine_images(images)
        if combined_image is not None:
            print(f"Kombinált kép létrehozva a kiválasztott képekből: {selected_images}")

    def save_images(self):
        """A kiválasztott képek mentése."""
        selected_images = self.get_selected_images()
        if not selected_images:
            messagebox.showwarning("Nincs kiválasztott kép", "Válassz ki egy vagy több képet a mentéshez!")
            return

        save_directory = filedialog.askdirectory(title="Válaszd ki a mentési könyvtárat")
        if not save_directory:
            messagebox.showwarning("Nincs kiválasztott könyvtár", "Válassz ki egy könyvtárat a mentéshez!")
            return

        for photo_name in selected_images:
            image = self.app.album.get_photos()[photo_name]
            save_path = os.path.join(save_directory, photo_name)
            cv2.imwrite(save_path, image)
            print(f"{photo_name} elmentve a {save_path} helyre.")

    def show_preview(self, event):
        """Előnézet megjelenítése a kiválasztott képekről."""
        selected_images = self.get_selected_images()
        if selected_images:
            photo_name = selected_images[0]  # Az első kiválasztott kép előnézete
            image = self.app.album.get_photos()[photo_name]
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img.thumbnail((400, 400))  # Előnézet méretének beállítása
            img_tk = ImageTk.PhotoImage(img)

            if hasattr(self, 'preview_label'):
                self.preview_label.config(image=img_tk)
                self.preview_label.image = img_tk  # Frissíti a képet
            else:
                self.preview_label = tk.Label(self.root, image=img_tk)
                self.preview_label.image = img_tk  # Meg kell tartani a referenciát
                self.preview_label.pack()
