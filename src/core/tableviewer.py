import tkinter as tk
import json

from tkinter import ttk
from loguru import logger


def update_table_data():
    try:
        with open("src/data/table.json", "r", encoding="utf-8") as f:
            table = json.load(f)
        
        new_data = table.get("data", [])
        
        # Clear existing treeview items
        for item in tree.get_children():
            tree.delete(item)

        # Insert new data
        for row in new_data:
            tree.insert("", "end", values=row)

    except FileNotFoundError:
        logger.warning("table.json not found. Waiting for it to be created.")
    except json.JSONDecodeError:
        logger.error("Error decoding table.json. The file might be corrupted or empty.")
    except Exception as e:
        logger.error(f"An error occurred in update_table_data: {e}")
    finally:
        # Schedule the next update
        if 'root' in globals() and root.winfo_exists():
            root.after(1000, update_table_data)


def startTable():
    global root
    global tree
    try:
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
        tree.pack(expand=True, fill=tk.BOTH)
        
        logger.success("Table View started")
        update_table_data()
        root.mainloop()
    except Exception as e:
        logger.error(f"Failed to start Table View: {e}")

