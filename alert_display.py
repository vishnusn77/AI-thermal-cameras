import tkinter as tk
from tkinter import ttk
import queue


class AlertSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Public Safety Alerts")
        self.root.geometry("1000x600")
        self.root.configure(bg="#1a1a1a")

        self.header_frame = tk.Frame(self.root, bg="#1a1a1a", height=100)
        self.header_frame.pack(fill="x", side="top")

        self.header_label = tk.Label(
            self.header_frame,
            text="🚨 Public Safety Monitoring System 🚨",
            font=("Helvetica", 28, "bold"),
            fg="#f9ed69",
            bg="#1a1a1a"
        )
        self.header_label.pack(pady=10)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=10)

        self.main_frame = tk.Frame(self.root, bg="#1a1a1a")
        self.main_frame.pack(expand=True, fill="both")

        self.message_label = tk.Label(
            self.main_frame,
            text="System Initializing...",
            font=("Helvetica", 24, "bold"),
            fg="#e63946",
            bg="#1a1a1a",
            wraplength=800,
            justify="center"
        )
        self.message_label.pack(pady=50)

    def update_alert(self, message, color="#e63946"):
        self.message_label.config(text=message, fg=color)


def run_alert_system(alert_queue):
    root = tk.Tk()
    alert_system = AlertSystem(root)

    def check_alerts():
        while not alert_queue.empty():
            alert = alert_queue.get()
            alert_system.update_alert(alert["message"], alert["color"])
        root.after(200, check_alerts)

    check_alerts()
    root.mainloop()
