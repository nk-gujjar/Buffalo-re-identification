import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests
import os
import json
from PIL import Image, ImageTk
import io

class CattleManagementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cattle Management System")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # API endpoint (change this to your actual backend API)
        self.api_url = "http://localhost:8000/api/cattle"
        
        # Variables
        self.selected_image_path = None
        self.image_data = None
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill="both", expand=True)
        
        # Title
        ttk.Label(self.main_frame, text="Cattle Management System", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Action buttons frame
        self.action_frame = ttk.Frame(self.main_frame)
        self.action_frame.pack(pady=10)
        
        # Add and Fetch buttons
        ttk.Button(self.action_frame, text="Add Cattle", command=self.show_add_form, width=20).grid(row=0, column=0, padx=10)
        ttk.Button(self.action_frame, text="Fetch Cattle Info", command=self.show_fetch_form, width=20).grid(row=0, column=1, padx=10)
        
        # Content frame (will contain either add form or fetch form)
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill="both", expand=True, pady=10)
        
        # Image preview frame
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Image Preview")
        self.image_frame.pack(fill="both", expand=True, pady=10)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Show initial view
        self.show_add_form()
    
    def clear_content_frame(self):
        """Clear the content frame to prepare for new content"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def show_add_form(self):
        """Show the form for adding cattle information"""
        self.clear_content_frame()
        
        # Create form elements
        ttk.Label(self.content_frame, text="Add New Cattle", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)
        
        # ID
        ttk.Label(self.content_frame, text="ID:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.id_var = tk.StringVar()
        ttk.Entry(self.content_frame, textvariable=self.id_var, width=30).grid(row=1, column=1, sticky=tk.W)
        
        # Gender
        ttk.Label(self.content_frame, text="Gender:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.gender_var = tk.StringVar()
        gender_combo = ttk.Combobox(self.content_frame, textvariable=self.gender_var, width=27)
        gender_combo['values'] = ('Male', 'Female')
        gender_combo.grid(row=2, column=1, sticky=tk.W)
        
        # Age
        ttk.Label(self.content_frame, text="Age:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.age_var = tk.StringVar()
        ttk.Entry(self.content_frame, textvariable=self.age_var, width=30).grid(row=3, column=1, sticky=tk.W)
        
        # Type
        ttk.Label(self.content_frame, text="Type:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.type_var = tk.StringVar()
        ttk.Entry(self.content_frame, textvariable=self.type_var, width=30).grid(row=4, column=1, sticky=tk.W)
        
        # Description
        ttk.Label(self.content_frame, text="Description:").grid(row=5, column=0, sticky=tk.NW, pady=5)
        self.description_text = tk.Text(self.content_frame, width=30, height=5)
        self.description_text.grid(row=5, column=1, sticky=tk.W)
        
        # Image Selection
        ttk.Label(self.content_frame, text="Image:").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Button(self.content_frame, text="Select Image", command=self.select_image).grid(row=6, column=1, sticky=tk.W)
        
        # Submit Button
        ttk.Button(self.content_frame, text="Submit", command=self.submit_add_form).grid(row=7, column=0, columnspan=2, pady=20)
    
    def show_fetch_form(self):
        """Show the form for fetching cattle information"""
        self.clear_content_frame()
        
        ttk.Label(self.content_frame, text="Fetch Cattle Information", font=("Arial", 12, "bold")).pack(pady=(0, 20), anchor=tk.W)
        
        ttk.Label(self.content_frame, text="Please select an image to fetch cattle information:").pack(anchor=tk.W, pady=5)
        
        ttk.Button(self.content_frame, text="Select Image", command=self.select_image).pack(anchor=tk.W, pady=5)
        
        ttk.Button(self.content_frame, text="Fetch Information", command=self.submit_fetch_form).pack(anchor=tk.W, pady=20)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(self.content_frame, text="Results")
        self.results_frame.pack(fill="both", expand=True, pady=10)
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.status_var.set(f"Image selected: {os.path.basename(file_path)}")
            self.load_image_preview(file_path)
    
    def load_image_preview(self, image_path):
        """Load and display image preview"""
        try:
            # Open and resize image for preview
            img = Image.open(image_path)
            # Calculate new dimensions while maintaining aspect ratio
            width, height = img.size
            max_size = 300
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Update image preview
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference to prevent garbage collection
            
            # Store image data for submission
            with open(image_path, "rb") as img_file:
                self.image_data = img_file.read()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.image_label.config(image=None)
            self.image_label.image = None
    
    def submit_add_form(self):
        """Submit the add form data to the API"""
        if not self.selected_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        # Get form data
        try:
            cattle_id = self.id_var.get()
            gender = self.gender_var.get()
            age = self.age_var.get()
            cattle_type = self.type_var.get()
            description = self.description_text.get("1.0", tk.END).strip()
            
            # Validate required fields
            if not cattle_id or not gender or not age or not cattle_type:
                messagebox.showwarning("Warning", "Please fill all required fields")
                return
            
            # Prepare form data
            form_data = {
                'id': cattle_id,
                'gender': gender,
                'age': age,
                'type': cattle_type,
                'description': description
            }
            
            # Prepare files data
            files = {
                'image': (os.path.basename(self.selected_image_path), self.image_data, 
                          f"image/{os.path.splitext(self.selected_image_path)[1][1:]}")
            }
            
            self.status_var.set("Submitting data...")
            
            # Make API request
            try:
                response = requests.post(f"{self.api_url}/add", 
                                         data=form_data,
                                         files=files)
                
                if response.status_code == 200:
                    messagebox.showinfo("Success", "Cattle information added successfully")
                    self.status_var.set("Data submitted successfully")
                    # Clear form
                    self.clear_form()
                else:
                    messagebox.showerror("Error", f"Failed to submit data: {response.text}")
                    self.status_var.set("Failed to submit data")
            
            except requests.RequestException as e:
                messagebox.showerror("Connection Error", f"Failed to connect to server: {str(e)}")
                self.status_var.set("Connection error")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred")
    
    def submit_fetch_form(self):
        """Submit the fetch form data to the API"""
        if not self.selected_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        self.status_var.set("Fetching information...")
        
        # Prepare files data
        files = {
            'image': (os.path.basename(self.selected_image_path), self.image_data, 
                      f"image/{os.path.splitext(self.selected_image_path)[1][1:]}")
        }
        
        # Make API request
        try:
            response = requests.post(f"{self.api_url}/fetch", files=files)
            
            if response.status_code == 200:
                # Process and display results
                self.display_results(response.json())
                self.status_var.set("Information fetched successfully")
            else:
                messagebox.showerror("Error", f"Failed to fetch data: {response.text}")
                self.status_var.set("Failed to fetch data")
        
        except requests.RequestException as e:
            messagebox.showerror("Connection Error", f"Failed to connect to server: {str(e)}")
            self.status_var.set("Connection error")
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid response from server")
            self.status_var.set("Invalid response")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred")
    
    def display_results(self, data):
        """Display the results from the fetch operation"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if not data or len(data) == 0:
            ttk.Label(self.results_frame, text="No matching cattle found").pack(pady=10)
            return
        
        # Create a table to display results
        columns = ('ID', 'Gender', 'Age', 'Type', 'Description')
        tree = ttk.Treeview(self.results_frame, columns=columns, show='headings')
        
        # Define headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # Add data to the table
        for item in data:
            values = (
                item.get('id', ''),
                item.get('gender', ''),
                item.get('age', ''),
                item.get('type', ''),
                item.get('description', '')
            )
            tree.insert('', tk.END, values=values)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        # Pack the table and scrollbar
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def clear_form(self):
        """Clear all form fields"""
        self.id_var.set("")
        self.gender_var.set("")
        self.age_var.set("")
        self.type_var.set("")
        self.description_text.delete("1.0", tk.END)
        self.selected_image_path = None
        self.image_data = None
        self.image_label.config(image=None)
        self.image_label.image = None

if __name__ == "__main__":
    root = tk.Tk()
    app = CattleManagementApp(root)
    root.mainloop()