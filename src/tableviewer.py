import json
import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from PyQt6.QtCore import QTimer
from loguru import logger


class TableViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI Table Update")
        self.setGeometry(100, 100, 1000, 500)

        self.table_widget = QTableWidget()
        self.setCentralWidget(self.table_widget)

        self.headers = [
            "Channel",
            "Coin",
            "Direction",
            "Deposit*L",
            "Order Price",
            "Current Price",
            "Profit",
            "Percent",
        ]
        self.table_widget.setColumnCount(len(self.headers))
        self.table_widget.setHorizontalHeaderLabels(self.headers)
        header = self.table_widget.horizontalHeader()
        if header:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_table_data)
        self.timer.start(100)

        self.update_table_data()

    def update_table_data(self):
        try:
            with open("data/table.json", "r", encoding="utf-8") as f:
                table_data = json.load(f)

            new_data = table_data.get("data", [])
            self.table_widget.setRowCount(len(new_data))

            for row_idx, row_data in enumerate(new_data):
                for col_idx, cell_data in enumerate(row_data):
                    self.table_widget.setItem(
                        row_idx, col_idx, QTableWidgetItem(str(cell_data))
                    )
            # logger.info("Table data updated successfully.")

        except FileNotFoundError:
            logger.warning("table.json not found. Waiting for it to be created.")
        except json.JSONDecodeError:
            logger.error(
                "Error decoding table.json. The file might be corrupted or empty."
            )
        except Exception as e:
            logger.error(f"An error occurred in update_table_data: {e}")


def startTable():
    try:
        app = QApplication(sys.argv)
        viewer = TableViewer()
        viewer.show()
        logger.success("Table View started")
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Failed to start Table View: {e}")
