"""
This module provides a real-time GUI display for the trading bot's activity
using the PyQt6 framework.
"""

import sys
import json
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLabel,
)
from PyQt6.QtCore import QTimer
from loguru import logger


class TableViewer(QMainWindow):
    """
    The main window for the application, displaying real-time trading data in a table
    and summary labels for balance and winrate.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot Dashboard")
        self.setGeometry(100, 100, 1200, 600)

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Info bar
        info_layout = self._create_info_layout()
        main_layout.addLayout(info_layout)

        # Table
        self.table_widget = self._create_table_widget()
        main_layout.addWidget(self.table_widget)

        # Setup a timer to refresh data periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)  # Refresh every 1 second

        logger.success("GUI Initialized.")

    def _create_info_layout(self) -> QHBoxLayout:
        """Creates the top layout with labels for balance and winrate."""
        info_layout = QHBoxLayout()
        self.balance_label = QLabel("Balance: N/A")
        self.available_balance_label = QLabel("Available Balance: N/A")
        self.winrate_label = QLabel("Winrate: N/A")

        # Apply some styling
        for label in [
            self.balance_label,
            self.available_balance_label,
            self.winrate_label,
        ]:
            label.setStyleSheet("font-size: 14px; padding: 5px;")
            info_layout.addWidget(label)

        return info_layout

    def _create_table_widget(self) -> QTableWidget:
        """Creates and configures the main table for displaying orders."""
        table = QTableWidget()
        headers = [
            "Channel",
            "Coin",
            "Side",
            "Margin ($)",
            "Entry Price",
            "Current Price",
            "PnL ($)",
            "PnL (%)",
        ]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )  # Make table read-only
        return table

    def update_data(self):
        """
        Loads data from table.json and updates the UI elements.
        This method is called by the QTimer.
        """
        try:
            with open("data/table.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            # Update info labels
            self.balance_label.setText(f"<b>Balance:</b> ${data.get('balance', 0):.2f}")
            self.available_balance_label.setText(
                f"<b>Available:</b> ${data.get('available_balance', 0):.2f}"
            )
            self.winrate_label.setText(f"<b>Winrate:</b> {data.get('winrate', 0)}%")

            # Update table content
            orders = data.get("orders", [])
            self.table_widget.setRowCount(len(orders))
            for row_idx, row_data in enumerate(orders):
                for col_idx, cell_data in enumerate(row_data):
                    self.table_widget.setItem(
                        row_idx, col_idx, QTableWidgetItem(str(cell_data))
                    )

        except FileNotFoundError:
            # This is expected if the updater hasn't run yet
            pass
        except json.JSONDecodeError:
            logger.warning(
                "Could not decode table.json. It may be empty or being written to."
            )
        except Exception as e:
            logger.error(f"An error occurred while updating GUI data: {e}")


def start_gui():
    """
    Initializes and runs the PyQt6 application. This should be called
    from a separate thread to not block the main application logic.
    """
    try:
        app = QApplication(sys.argv)
        viewer = TableViewer()
        viewer.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Failed to start the GUI: {e}")
