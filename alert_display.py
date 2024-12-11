import tkinter as tk
from tkinter import ttk


class AlertSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Public Safety Alerts")
        self.root.geometry("1000x600")
        self.root.configure(bg="#1a1a1a")

        # Header Frame
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

        # Crowd Alert Frame
        self.crowd_frame = tk.Frame(self.root, bg="#1a1a1a")
        self.crowd_frame.pack(expand=True, fill="both", side="top", padx=10, pady=10)

        self.crowd_label = tk.Label(
            self.crowd_frame,
            text="Crowd Alert:\nInitializing...",
            font=("Helvetica", 20, "bold"),
            fg="#e63946",
            bg="#1a1a1a",
            wraplength=800,
            justify="left"
        )
        self.crowd_label.pack(pady=10, anchor="w")

        self.crowd_alert_label = tk.Label(
            self.crowd_frame,
            text="",
            font=("Helvetica", 18),
            fg="#ffdf00",  # Yellow for alert message
            bg="#1a1a1a",
            wraplength=800,
            justify="left"
        )
        self.crowd_alert_label.pack(pady=5, anchor="w")

        # Temperature Alert Frame
        self.temp_frame = tk.Frame(self.root, bg="#1a1a1a")
        self.temp_frame.pack(expand=True, fill="both", side="top", padx=10, pady=10)

        self.temp_label = tk.Label(
            self.temp_frame,
            text="Temperature Alert:\nInitializing...",
            font=("Helvetica", 20, "bold"),
            fg="#e63946",
            bg="#1a1a1a",
            wraplength=800,
            justify="left"
        )
        self.temp_label.pack(pady=10, anchor="w")

        self.temp_alert_label = tk.Label(
            self.temp_frame,
            text="",
            font=("Helvetica", 18),
            fg="#ffdf00",  # Yellow for alert message
            bg="#1a1a1a",
            wraplength=800,
            justify="left"
        )
        self.temp_alert_label.pack(pady=5, anchor="w")

    def update_crowd_alert(self, message, color="#e63946", alert_message=None):
        """
        Update the crowd alert section with an optional yellow alert message.
        :param message: The main message to display in the crowd alert section.
        :param color: The color of the main text.
        :param alert_message: The alert message to display in yellow, if applicable.
        """
        self.crowd_label.config(text=f"Crowd Alert:\n{message}", fg=color)
        if alert_message:
            self.crowd_alert_label.config(text=f"⚠️ {alert_message}")
        else:
            self.crowd_alert_label.config(text="")

    def update_temp_alert(self, message, color="#e63946", alert_message=None):
        """
        Update the temperature alert section with an optional yellow alert message.
        :param message: The main message to display in the temperature alert section.
        :param color: The color of the main text.
        :param alert_message: The alert message to display in yellow, if applicable.
        """
        self.temp_label.config(text=f"Temperature Alert:\n{message}", fg=color)
        if alert_message:
            self.temp_alert_label.config(text=f"⚠️ {alert_message}")
        else:
            self.temp_alert_label.config(text="")


def run_alert_system(alert_queue):
    root = tk.Tk()
    alert_system = AlertSystem(root)

    def check_alerts():
        while not alert_queue.empty():
            alert = alert_queue.get()

            # Check the type of alert and update the corresponding section
            if alert["type"] == "crowd":
                alert_system.update_crowd_alert(
                    alert["message"],
                    alert["color"],
                    alert.get("alert_message")
                )
            elif alert["type"] == "temperature":
                alert_system.update_temp_alert(
                    alert["message"],
                    alert["color"],
                    alert.get("alert_message")
                )
        
        root.after(200, check_alerts)

    check_alerts()
    root.mainloop()
