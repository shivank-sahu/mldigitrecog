import tkinter as tk
import pyttsx3 

def speech():
    # Get text from entry widget
    text = text_entry.get()
    
    # Initialize the engine
    engine = pyttsx3.init()
    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)   # Volume (0.0 to 1.0)
    # Convert text to speech
    engine.say(text)
    # Wait for speech to finish
    engine.runAndWait()

# Create a tkinter window
window = tk.Tk()
window.title("Text to Speech")

# Create a frame to hold the widgets
frame = tk.Frame(window, padx=20, pady=20)
frame.grid(row=0, column=0)

# Create a label and entry widget for text input
text_label = tk.Label(frame, text="Enter text:")
text_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

text_entry = tk.Entry(frame, width=40)
text_entry.grid(row=0, column=1, padx=5, pady=5)

# Create a button to trigger text-to-speech conversion
convert_button = tk.Button(frame, text="Convert to Speech", command=speech)
convert_button.grid(row=1, column=0, columnspan=2, pady=10)

# Run the tkinter event loop
window.mainloop()
