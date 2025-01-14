import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from query_data import query_rag
import logging
import shutil

# Directory to store user-uploaded data
USER_DATA_FOLDER = "user_data"
if not os.path.exists(USER_DATA_FOLDER):
    os.makedirs(USER_DATA_FOLDER)

# Directory for chat history
HISTORY_FOLDER = "history"
if not os.path.exists(HISTORY_FOLDER):
    os.makedirs(HISTORY_FOLDER)


# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Set app title and default theme
        self.title("Aegis RAG")
        self.geometry("900x600")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize state
        self.current_screen = "chat"
        self.current_chat = None
        self.chat_history = {}
        self.selected_model = "mock"

        # Configure the main layout
        self.columnconfigure(0, weight=1)  # Full width
        self.rowconfigure(1, weight=1)  # Remaining content area


        # Navbar with clickable text
        self.navbar = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.navbar.grid(row=0, column=0, sticky="ew")
        self.navbar.grid_propagate(False)
        self.navbar.columnconfigure([0, 1, 2], weight=0)  # Distribute evenly

        # Clickable text as navbar
        self.create_navbar_item("Chat", self.load_chat_screen, 0)
        self.create_navbar_item("Data", self.load_data_screen, 1)
        self.create_navbar_item("Settings", self.load_settings_screen, 2)

        # Create Main Frame for dynamic content
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, sticky="nsew")
        self.rowconfigure(1, weight=3)
        self.columnconfigure(0, weight=1)



        # Load Default Screen
        self.load_chat_screen()

        logging.info("Application initialized.")

    def create_navbar_item(self, text, command, column):
        """Create a clickable navbar item."""
        label = ctk.CTkLabel(
            self.navbar,
            text=text,
            font=("Arial", 12, "bold"),
            cursor="hand2",  # Changes cursor to a hand pointer
            anchor="center",
        )
        label.grid(row=0, column=column, sticky="ew", padx=5)
        label.bind("<Button-1>", lambda event: command())  # Bind left mouse click

    def load_chat_screen(self):
        """Load the Chat screen."""
        self.clear_main_frame()
        self.current_screen = "chat"

        # Configure the main frame to distribute space
        self.main_frame.columnconfigure(0, weight=0)  # Navbar or empty space
        self.main_frame.columnconfigure(1, weight=1)  # Left frame (chat history)
        self.main_frame.columnconfigure(2, weight=3)  # Right frame (chat interface)
        self.main_frame.rowconfigure(0, weight=1)  # Allow rows to expand

        # Left Frame: Chat history
        left_frame = ctk.CTkFrame(self.main_frame)
        left_frame.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)  # No padx
        left_frame.columnconfigure(0, weight=1)

        history_label = ctk.CTkLabel(left_frame, text="Chat History", font=("Arial", 16, "bold"))
        history_label.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)  # No padx

        self.history_listbox = ctk.CTkScrollableFrame(left_frame)
        self.history_listbox.grid(row=1, column=0, sticky="nsew", padx=3, pady=3)
        self.load_history()

        # New Chat Button
        ctk.CTkButton(left_frame, text="New Chat", command=self.new_chat).grid(
            row=6, column=0, sticky="ew", padx=5, pady=5  # No padx
        )

        self.model_selector = ctk.CTkOptionMenu(left_frame,
            values=["mock", "gpt-neo", "rag", "minilm", "distilgpt2"],
            command=self.change_model,
        ).grid(row=7, column=0, sticky="ew", padx=5, pady=5)


        # Right Frame: Chat interface
        chat_frame = ctk.CTkFrame(self.main_frame)
        chat_frame.grid(row=0, column=2, sticky="nsew")  # No padx or pady
        chat_frame.rowconfigure(0, weight=4)  # Chat display row
        chat_frame.rowconfigure(1, weight=1)  # Input frame row
        chat_frame.columnconfigure(0, weight=1)  # Single column layout for chat

        self.chat_display = ctk.CTkTextbox(chat_frame, state="disabled", wrap="word")
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        input_frame = ctk.CTkFrame(chat_frame)
        input_frame.grid(row=1, column=0, sticky="ew", padx=2, pady=2)  # No padx
        input_frame.columnconfigure(0, weight=1)

        self.input_box = ctk.CTkEntry(input_frame, placeholder_text="Type your message...")
        self.input_box.grid(row=0, column=0, sticky="ew", padx=2)  # No padx
        self.input_box.bind("<Return>", self.send_message_with_event)

        ctk.CTkButton(input_frame, text="Send", command=self.send_message).grid(
            row=0, column=1, padx=2, sticky="e"  # No padx
        )

        logging.info("Chat screen loaded.")

    def load_data_screen(self):
        """Load the Data screen for the active chat."""
        if not self.current_chat:
            messagebox.showerror("Error", "No active chat to load data for.")
            return

        self.clear_main_frame()
        self.current_screen = "data"

        # Ensure the chat folder exists
        chat_folder = self.create_chat_folder(self.current_chat)

        # Main data frame to occupy the entire space under the navbar
        data_frame = ctk.CTkFrame(self.main_frame)
        data_frame.grid(row=0, column=0, sticky="nsew")
        data_frame.columnconfigure(0, weight=3)
        data_frame.rowconfigure([1], weight=3)  # Row for file list expands
        data_frame.rowconfigure([0, 2], weight=0)  # Title and upload options don't expand

        # Title label
        title_label = ctk.CTkLabel(
            data_frame,
            text=f"Data for {self.current_chat}",
            font=("Arial", 20, "bold")
        )
        title_label.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        # Scrollable frame for displaying files
        self.file_listbox = ctk.CTkScrollableFrame(data_frame)
        self.file_listbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.load_file_list(chat_folder)

        # Upload options frame
        upload_frame = ctk.CTkFrame(data_frame)
        upload_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkButton(upload_frame, text="Upload Single File", command=self.upload_single_file).pack(
            fill="x", padx=5, pady=5
        )
        ctk.CTkButton(upload_frame, text="Upload Bulk Files", command=self.upload_bulk_files).pack(
            fill="x", padx=5, pady=5
        )

        # Back to chat button
        back_button = ctk.CTkButton(
            data_frame,
            text="Back to Chat",
            command=self.load_chat_screen
        )
        back_button.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

        logging.info(f"Data screen loaded for {self.current_chat}.")

    def load_settings_screen(self):
        """Load the Settings screen."""
        self.clear_main_frame()
        self.current_screen = "settings"

        settings_frame = ctk.CTkFrame(self.main_frame)
        settings_frame.grid(row=0, column=0, sticky="nsew")
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.rowconfigure([0, 1, 2], weight=1)  # Allow rows to stretch evenly

        ctk.CTkLabel(settings_frame, text="Settings", font=("Arial", 20, "bold")).grid(
            row=0, column=0, columnspan=2, pady=20, sticky="n"
        )

        # Dark Mode Checkbox
        self.theme_checkbox = ctk.CTkCheckBox(
            settings_frame,
            text="Enable Dark Mode",
            command=self.toggle_theme,
            onvalue="dark",
            offvalue="light",
        )

        self.theme_checkbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Dark Mode Checkbox
        self.theme_checkbox = ctk.CTkCheckBox(
            settings_frame,
            text="Enable Dark Mode",
            command=self.toggle_theme,
            onvalue="dark",
            offvalue="light",
        )
        self.theme_checkbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Explicitly set initial checkbox state based on current theme
        current_mode = ctk.get_appearance_mode()
        if current_mode == "Dark":
            self.theme_checkbox.select()
        else:
            self.theme_checkbox.deselect()

        logging.info(f"Settings screen loaded with theme: {current_mode}.")

    def clear_main_frame(self):
        """Clear all widgets from the main frame."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def toggle_theme(self):
        """Toggle the application theme."""
        new_mode = self.theme_checkbox.get()
        ctk.set_appearance_mode(new_mode)
        logging.info(f"Theme toggled to {new_mode}.")

    def change_model(self, model: str):
        """Change the selected model."""
        self.selected_model = model
        self._update_chat(f"System: Switched to model '{model}'.")

    def load_file_list(self, chat_folder):
        """Load the list of files in the given chat folder."""
        for widget in self.file_listbox.winfo_children():
            widget.destroy()

        files = os.listdir(chat_folder)
        if not files:
            no_files_label = ctk.CTkLabel(self.file_listbox, text="No files uploaded yet.")
            no_files_label.pack(pady=10)
        else:
            for file in files:
                file_path = os.path.join(chat_folder, file)

                # Frame for file and delete button
                file_frame = ctk.CTkFrame(self.file_listbox)
                file_frame.pack(fill="x", padx=5, pady=5)

                # File label
                file_label = ctk.CTkLabel(file_frame, text=file, anchor="w")
                file_label.pack(side="left", fill="x", expand=True)

                # Delete button
                delete_button = ctk.CTkButton(
                    file_frame,
                    text="Delete",
                    command=lambda fp=file_path, fn=file: self.safe_delete(fp, fn)
                )
                delete_button.pack(side="right")

    def change_model(self, model: str):
        """Change the selected model."""
        self.selected_model = model
        self._update_chat(f"System: Switched to model '{model}'.")

    def send_message_with_event(self, event=None):
        """Wrapper to allow sending messages with Enter key."""
        self.send_message()

    def prevent_button_enter(self, event):
        """Prevent Enter from triggering the button when focused."""
        return "break"

    def send_message(self):
        """Send a message and process it using the selected model."""
        message = self.input_box.get()
        if not message.strip():
            return

        # Automatically create a new chat if no active chat exists
        if self.current_chat is None:
            self.new_chat()

        self.input_box.delete(0, "end")
        self._update_chat(f"You: {message}")

        # Query the RAG pipeline
        self.after(500, lambda: self.query_model(message))

    def query_model(self, message: str):
        """Query the selected model and display the response."""
        try:
            response = query_rag(message, self.selected_model)
            if response["error"]:
                self._update_chat(f"Error: {response['error']}")
            else:
                self._update_chat(f"Assistant: {response['content']}")
                # Append the message and response to the current chat history
                self.chat_history[self.current_chat].append(
                    {"user": message, "assistant": response["content"]}
                )
                self.save_chat_history(self.current_chat)  # Save chat
        except Exception as e:
            self._update_chat(f"Error: {str(e)}")

    def _update_chat(self, message: str):
        """Update the chat display."""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{message}\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def create_chat_folder(self, chat_id):
        """Create a folder for the given chat."""
        chat_folder = os.path.join(USER_DATA_FOLDER, chat_id)
        if not os.path.exists(chat_folder):
            os.makedirs(chat_folder)
            logging.info(f"Created folder for chat: {chat_folder}")
        return chat_folder

    def new_chat(self):
        """Start a new chat session with a unique ID."""
        existing_chats = self.list_history()

        # Generate the next available unique ID
        next_chat_number = 1
        while f"Chat_{next_chat_number}" in existing_chats:
            next_chat_number += 1

        chat_id = f"Chat_{next_chat_number}"
        self.chat_history[chat_id] = []
        self.current_chat = chat_id

        # Create a folder for the new chat
        self.create_chat_folder(chat_id)

        self.save_chat_history(chat_id)
        self.update_history_list()
        self.clear_chat_display()
        logging.info(f"New chat '{chat_id}' started.")

    def clear_chat_display(self):
        """Clear the chat display."""
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

    def list_history(self):
        """List all chat files in the chat folder."""
        try:
            files = os.listdir(HISTORY_FOLDER)
            chat_ids = [os.path.splitext(f)[0] for f in files if f.endswith(".json")]
            logging.info(f"Available chats: {chat_ids}")
            return chat_ids
        except Exception as e:
            logging.error(f"Error listing chats: {e}")
            return []

    def update_history_list(self):
        """Update the chat history list on the left."""
        for widget in self.history_listbox.winfo_children():
            widget.destroy()

        chat_ids = self.list_history()
        logging.debug(f"Updating history list with: {chat_ids}")
        for chat_id in chat_ids:
            frame = ctk.CTkFrame(self.history_listbox)
            frame.pack(fill="x", padx=5, pady=2)

            button = ctk.CTkButton(
                frame, text=chat_id, command=lambda cid=chat_id: self.load_chat(cid)
            )
            button.pack(side="left", fill="x", expand=True)

            delete_button = ctk.CTkButton(
                frame, text="Delete", command=lambda cid=chat_id: self.delete_chat(cid)
            )
            delete_button.pack(side="right")

    def load_chat(self, chat_id: str):
        """Load a chat session."""
        logging.info(f"Loading chat with ID: {chat_id}")
        logging.debug(f"Available chats in history: {list(self.chat_history.keys())}")

        file_path = os.path.join(HISTORY_FOLDER, f"{chat_id}.json")
        try:
            with open(file_path, "r") as file:
                self.chat_history[chat_id] = json.load(file)
            self.current_chat = chat_id
            self.clear_chat_display()
            for entry in self.chat_history[chat_id]:
                self._update_chat(f"You: {entry['user']}")
                self._update_chat(f"Assistant: {entry['assistant']}")
            logging.info(f"Chat '{chat_id}' loaded successfully from {file_path}.")
        except FileNotFoundError:
            logging.error(f"Chat file '{file_path}' not found.")
            messagebox.showerror("Error", f"Chat session '{chat_id}' does not exist.")
        except Exception as e:
            logging.error(f"Error loading chat '{chat_id}': {e}")

    def save_chat_history(self, chat_id):
        """Save a specific chat session to a file."""
        file_path = os.path.join(HISTORY_FOLDER, f"{chat_id}.json")
        try:
            with open(file_path, "w") as file:
                json.dump(self.chat_history[chat_id], file)
            logging.info(f"Chat '{chat_id}' saved successfully to {file_path}.")
        except Exception as e:
            logging.error(f"Error saving chat '{chat_id}': {e}")

    def load_history(self):
        """Populate the chat history list without loading individual chats."""
        try:
            chat_ids = self.list_history()
            if chat_ids:
                logging.info(f"Found {len(chat_ids)} chat(s).")
            else:
                logging.info("No chats found in history.")
            self.update_history_list()
        except Exception as e:
            logging.error(f"Error loading history: {e}")

    def delete_chat(self, chat_id):
        """Delete a specific chat."""
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete '{chat_id}'?"):
            file_path = os.path.join(HISTORY_FOLDER, f"{chat_id}.json")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Chat '{chat_id}' deleted successfully.")
                    self.chat_history.pop(chat_id, None)  # Remove from in-memory history
                    self.update_history_list()
                    self.clear_chat_display()
                else:
                    logging.warning(f"Chat file '{file_path}' does not exist.")
                    messagebox.showwarning("Warning", f"Chat '{chat_id}' does not exist.")
            except Exception as e:
                logging.error(f"Error deleting chat '{chat_id}': {e}")

    def upload_single_file(self):
        """Upload a single file to the folder for the active chat."""
        if not self.current_chat:
            messagebox.showerror("Error", "No active chat to upload files for.")
            return

        chat_folder = self.create_chat_folder(self.current_chat)
        file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("PDF files", "*.pdf")])
        if file_path:
            filename = os.path.basename(file_path)
            destination = os.path.join(chat_folder, filename)
            try:
                shutil.copy2(file_path, destination)  # Copy the file instead of moving
                logging.info(f"File '{filename}' copied successfully to {chat_folder}.")
                self.load_file_list(chat_folder)  # Refresh file list
            except Exception as e:
                logging.error(f"Error copying file '{filename}': {e}")
                messagebox.showerror("Error", f"Failed to copy file: {e}")

    def upload_bulk_files(self):
        """Upload multiple files to the folder for the active chat."""
        if not self.current_chat:
            messagebox.showerror("Error", "No active chat to upload files for.")
            return

        chat_folder = self.create_chat_folder(self.current_chat)
        file_paths = filedialog.askopenfilenames(title="Select Files", filetypes=[("PDF files", "*.pdf")])
        if file_paths:
            try:
                for file_path in file_paths:
                    filename = os.path.basename(file_path)
                    destination = os.path.join(chat_folder, filename)
                    shutil.copy2(file_path, destination)  # Copy the file instead of moving
                    logging.info(f"File '{filename}' copied successfully to {chat_folder}.")
                self.load_file_list(chat_folder)  # Refresh file list
            except Exception as e:
                logging.error(f"Error copying files: {e}")
                messagebox.showerror("Error", f"Failed to copy files: {e}")

    def safe_delete(self, file_path, file_name):
        """Safely delete a file with confirmation."""
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete '{file_name}'?"):
            try:
                os.remove(file_path)
                logging.info(f"File '{file_name}' deleted successfully.")
                self.load_file_list(os.path.dirname(file_path))  # Refresh file list
                messagebox.showinfo("Success", f"'{file_name}' has been deleted.")
            except Exception as e:
                logging.error(f"Error deleting file '{file_name}': {e}")
                messagebox.showerror("Error", f"Could not delete '{file_name}'. Please try again.")


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
