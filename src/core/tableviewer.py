import tkinter as tk
import json

from tkinter import ttk
from loguru import logger


def update_table_data():
    try:
        with open("src/data/table.json", encoding="utf-8") as f:
            table = json.load(f)
        new_data = [i for i in table["data"]]
        for item in tree.get_children():
            tree.delete(item)

        for row in new_data:
            tree.insert("", "end", values=row)

        root.after(1000, update_table_data)
    except Exception as e:
        logger.error(e)
        root.after(1000, update_table_data)


def startTable():
    global root
    global tree
    root = tk.Tk()
    root.geometry("1000x500")
    root.title("GUI Table Update")

    headers = (
        "Channel",
        "Coin",
        "Diraction",
        "Deposit*L",
        "Order Price",
        "Current Price",
        "Profit",
        "Procent",
    )
    tree = ttk.Treeview(root, columns=headers, show="headings")
    for col in headers:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=100)
    tree.pack(expand=1, fill=tk.BOTH)
    logger.success("Table View started")
    update_table_data()
    root.mainloop()
