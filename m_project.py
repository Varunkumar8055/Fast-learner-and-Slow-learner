import os
import pandas as pd
import numpy as np
from scipy.signal import welch, coherence
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns

# Feature Extraction Functions
def calculate_band_power(data, fs, band, window_sec=2):
    if len(data) < window_sec * fs:
        nperseg = len(data)
    else:
        nperseg = window_sec * fs
    freqs, psd = welch(data, fs, nperseg=nperseg)
    band_freqs = (freqs >= band[0]) & (freqs <= band[1])
    return np.sum(psd[band_freqs])

def calculate_signal_entropy(data):
    from scipy.stats import entropy
    return entropy(data)

def calculate_channel_coherence(channel1, channel2, fs):
    freqs, coh = coherence(channel1, channel2, fs)
    return np.mean(coh)

def extract_features(student_data, fs, bands):
    band_power = []
    entropy_values = []
    coherence_values = []

    for channel in range(student_data.shape[1]):
        channel_band_power = {
            band_name: calculate_band_power(student_data.iloc[:, channel], fs, band_range)
            for band_name, band_range in bands.items()
        }
        band_power.append(channel_band_power)

        entropy_values.append(calculate_signal_entropy(student_data.iloc[:, channel]))

        if channel > 0:
            channel_coherence = calculate_channel_coherence(
                student_data.iloc[:, 0], student_data.iloc[:, channel], fs
            )
            coherence_values.append(channel_coherence)

    return band_power, entropy_values, coherence_values

# Load EEG Data
def load_eeg_data(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            data.append(pd.read_csv(file_path, header=None))
    return data

# Prepare Features
def prepare_features(data, fs, bands, label_mapping=None):
    band_power_list = []
    features = []
    labels = []

    for student_data in data:
        band_power, _, _ = extract_features(student_data, fs, bands)
        band_power_list.append(band_power)

    for band_power in band_power_list:
        for channel_band_power in band_power:
            features.append(list(channel_band_power.values()))
            if label_mapping:
                labels.append(label_mapping.get(max(channel_band_power, key=channel_band_power.get), 'slow'))
            else:
                labels.append(max(channel_band_power, key=channel_band_power.get))

    return pd.DataFrame(features, columns=list(bands.keys())), labels

# Build CNN-LSTM Model
def build_cnn_lstm_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Visualization Functions
def plot_bar_chart(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(data.keys(), data.values(), color='skyblue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    return fig

# Tkinter GUI
class EEGAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Cognitive Load Analysis")
        self.root.geometry("900x700")
        self.fs = 500
        self.bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 100),
        }
        self.label_mapping = {"alpha": 1, "beta": 1, "gamma": 1, "delta": 0, "theta": 0}
        self.model_path = "eeg_cnn_lstm_model.h5"
        self.model = self.load_or_train_model()

        # GUI Elements
        header_label = tk.Label(root, text="EEG Cognitive Load Analysis", font=("Helvetica", 18, "bold"))
        header_label.pack(pady=20)

        self.info_frame = tk.Frame(root)
        self.info_frame.pack(pady=20)

        self.predict_button = tk.Button(self.info_frame, text="Predict Cognitive Load", command=self.predict_data, width=20)
        self.predict_button.grid(row=0, column=0, padx=20)

        self.compare_button = tk.Button(self.info_frame, text="Compare Groups", command=self.compare_groups, width=20)
        self.compare_button.grid(row=0, column=1, padx=20)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill="both", expand=True)

        # Status message area
        self.status_message = tk.Label(root, text="Status: Ready", font=("Helvetica", 12), fg="green")
        self.status_message.pack(pady=10)

    def display_plot(self, fig):
        # Clear the canvas frame
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # Add the plot to the canvas frame
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_or_train_model(self):
        if os.path.exists(self.model_path):
            return load_model(self.model_path)
        else:
            return self.train_model()

    def train_model(self):
        train_folder = "Active state"  # Specify your path here
        train_data = load_eeg_data(train_folder)
        features_df, labels = prepare_features(train_data, self.fs, self.bands, self.label_mapping)
        scaler = StandardScaler()
        X = scaler.fit_transform(features_df)
        y = pd.Categorical(labels).codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        model = build_cnn_lstm_model(X_train.shape[1:], 2)
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
        model.save(self.model_path)

        # Display Metrics
        loss, accuracy = model.evaluate(X_test, y_test)
        messagebox.showinfo("Training Complete", f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        # Display Confusion Matrix
        y_pred = model.predict(X_test).argmax(axis=1)
        fig = plot_confusion_matrix(y_test, y_pred, ['Slow', 'Fast'])
        self.display_plot(fig)

        return model

    def predict_data(self):
        file_path = filedialog.askopenfilename(title="Select Test Data File")
        if file_path and self.model:
            test_data = pd.read_csv(file_path, header=None)
            features_df, _ = prepare_features([test_data], self.fs, self.bands, self.label_mapping)
            scaler = StandardScaler()
            features = scaler.fit_transform(features_df)
            features = np.expand_dims(features, axis=-1)

            predictions = self.model.predict(features).argmax(axis=1)
            prediction_counts = {"Slow": (predictions == 0).sum(), "Fast": (predictions == 1).sum()}
            fig = plot_bar_chart(prediction_counts, "Predicted Cognitive Load", "Class", "Count")
            self.display_plot(fig)

            # Update status
            self.status_message.config(text="Status: Prediction Complete", fg="blue")

    def compare_groups(self):
        group1_path = filedialog.askdirectory(title="Select Group 1 Folder")
        group2_path = filedialog.askdirectory(title="Select Group 2 Folder")
        
        if group1_path and group2_path and self.model:
            # Predict for Group 1
            group1_data = load_eeg_data(group1_path)
            group1_features_df, _ = prepare_features(group1_data, self.fs, self.bands, self.label_mapping)
            scaler = StandardScaler()
            group1_features = scaler.fit_transform(group1_features_df)
            group1_features = np.expand_dims(group1_features, axis=-1)
            group1_predictions = self.model.predict(group1_features).argmax(axis=1)
            group1_counts = {
                "Slow": (group1_predictions == 0).sum(),
                "Fast": (group1_predictions == 1).sum(),
            }
            
            # Predict for Group 2
            group2_data = load_eeg_data(group2_path)
            group2_features_df, _ = prepare_features(group2_data, self.fs, self.bands, self.label_mapping)
            group2_features = scaler.fit_transform(group2_features_df)
            group2_features = np.expand_dims(group2_features, axis=-1)
            group2_predictions = self.model.predict(group2_features).argmax(axis=1)
            group2_counts = {
                "Slow": (group2_predictions == 0).sum(),
                "Fast": (group2_predictions == 1).sum(),
            }
            
            # Visualization for comparison
            labels = ["Slow", "Fast"]
            group1_values = [group1_counts[label] for label in labels]
            group2_values = [group2_counts[label] for label in labels]
            
            x = np.arange(len(labels))  # Labels for the bars
            width = 0.35  # Width of the bars
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(x - width / 2, group1_values, width, label='Group 1', color='blue')
            ax.bar(x + width / 2, group2_values, width, label='Group 2', color='orange')
            
            ax.set_xlabel('Cognitive Load Class')
            ax.set_ylabel('Count')
            ax.set_title('Comparison of Cognitive Load Between Groups')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            
            self.display_plot(fig)
        else:
            messagebox.showerror("Error", "Please select valid directories for both groups.")


# Start the GUI Application
if __name__ == "__main__":
    root = tk.Tk()
    app = EEGAnalyzerApp(root)
    root.mainloop()
