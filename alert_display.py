import tkinter as tk
from tkinter import ttk
import threading
import queue
import time


class AlertSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Public Safety Alerts")
        self.root.geometry("1000x600")
        self.root.configure(bg="#1a1a1a")  # Dark background for a professional look

        # Header Frame with Gradient Background
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

        # Divider
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=10)

        # Main Content Frame
        self.main_frame = tk.Frame(self.root, bg="#1a1a1a")
        self.main_frame.pack(expand=True, fill="both")

        # Alert Message Area
        self.message_frame = tk.Frame(self.main_frame, bg="#1a1a1a")
        self.message_frame.pack(expand=True, side="left", fill="both", padx=20)

        self.message_label = tk.Label(
            self.message_frame,
            text="System Initializing...",
            font=("Helvetica", 24, "bold"),
            fg="#e63946",  # Vibrant red for attention
            bg="#1a1a1a",
            wraplength=800,
            justify="center"
        )
        self.message_label.pack(pady=50)

        # Footer Frame
        self.footer_frame = tk.Frame(self.root, bg="#1a1a1a", height=50)
        self.footer_frame.pack(fill="x", side="bottom")

        self.footer_label = tk.Label(
            self.footer_frame,
            text="Stay Safe | Follow Health Guidelines | Contact Authorities in Case of Emergency",
            font=("Helvetica", 14, "italic"),
            fg="#a8dadc",
            bg="#1a1a1a"
        )
        self.footer_label.pack(pady=10)

        # Bind "q" key to quit the program
        self.root.bind('q', self.quit_program)

    def quit_program(self, event=None):
        """Quit the Tkinter application when "q" is pressed."""
        self.root.quit()

    def update_alert(self, message, color="#e63946"):
        """
        Update the alert message dynamically.
        :param message: The alert message to display.
        :param color: Color of the message text.
        """
        self.message_label.config(text=message, fg=color)


def update_alert_message(alert_queue, people_count, temperatures):
    """
    Create and push an alert message based on the crowd and temperature information.
    :param alert_queue: Queue to send the alert message.
    :param people_count: The number of people detected.
    :param temperatures: Dictionary with person IDs and their temperatures.
    """
    temperature_info = ", ".join([f"Person {id}: {temp}°C" for id, temp in temperatures.items()])
    alert_message = f"Total People: {people_count}\nTemperatures: {temperature_info}"
    
    alert_queue.put({"message": alert_message, "color": "#ff6347"})


def detect_and_send_alerts(alert_queue):
    """
    Simulate detection of people and temperature, then send alerts.
    :param alert_queue: Queue for sending alert messages.
    """
    # Simulated detection data for two people
    detected_people = [
        {"id": 1, "bbox": (100, 150, 200, 250)},  # Person 1
        {"id": 2, "bbox": (300, 200, 400, 300)}   # Person 2
    ]
    
    # Simulated temperatures for the detected people
    temperatures = {1: 36.5, 2: 37.2}  # Simulated temperatures for detected people
    
    # Get the number of detected people
    people_count = len(detected_people)
    
    # Send the alert message with updated people count and their temperatures
    update_alert_message(alert_queue, people_count, temperatures)


def run_alert_system(alert_queue):
    """
    Run the alert display system.
    :param alert_queue: Queue to hold incoming alert messages.
    """
    root = tk.Tk()
    alert_system = AlertSystem(root)

    def check_alerts():
        if not alert_queue.empty():
            alert = alert_queue.get()
            alert_system.update_alert(alert["message"], alert["color"])
        root.after(1000, check_alerts)  # Check every second for new alerts

    check_alerts()
    root.mainloop()


def main():
    # Queue to communicate between processes (detection system and alert system)
    alert_queue = queue.Queue()

    # Start the alert system in a separate thread
    alert_thread = threading.Thread(target=run_alert_system, args=(alert_queue,), daemon=True)
    alert_thread.start()

    # Simulate detection and send alerts
    detect_and_send_alerts(alert_queue)
    
    # Keep the main thread running for Tkinter's event loop
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
