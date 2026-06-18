import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Task1 import EOGAnalyzer

class EOGTaskGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EOG Signal Analysis - HCI Task")
        self.root.geometry("1000x700")
        
        self.analyzer = EOGAnalyzer()
        self.horizontal_file = None
        self.test_file = None
        self.results = None
        
        # Colors for clam style
        self.bg_color = '#dcdad5'
        self.primary_color = '#4a7a9c'
        
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TLabel', background=self.bg_color, font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=6)
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabelframe', background=self.bg_color, font=('Arial', 10, 'bold'))
        style.configure('TLabelframe.Label', background=self.bg_color, font=('Arial', 10, 'bold'))
        style.map('TButton', background=[('active', '#c0c0c0')])
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="EOG Signal Analysis", 
                                font=('Arial', 22, 'bold'), foreground=self.primary_color)
        title_label.pack(pady=(0, 15))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="15")
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Horizontal signals
        horiz_frame = ttk.Frame(file_frame)
        horiz_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(horiz_frame, text="Browse Horizontal Signals (Excel)", 
                  command=self.load_horizontal).pack(side=tk.LEFT, padx=5)
        self.horizontal_label = ttk.Label(horiz_frame, text="No file selected", 
                                         foreground='gray')
        self.horizontal_label.pack(side=tk.LEFT, padx=10)
        
        # Test signal
        test_frame = ttk.Frame(file_frame)
        test_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(test_frame, text="Browse Test Signal (Text)", 
                  command=self.load_test).pack(side=tk.LEFT, padx=5)
        self.test_label = ttk.Label(test_frame, text="No file selected", 
                                   foreground='gray')
        self.test_label.pack(side=tk.LEFT, padx=10)
        
        # Process button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.process_btn = ttk.Button(button_frame, text="Process and Classify", 
                                      command=self.process_data, state='disabled')
        self.process_btn.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(button_frame, mode='indeterminate', length=400)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="15")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left side - Classification table
        table_frame = ttk.Frame(results_frame)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Treeview for results
        columns = ('rank', 'name', 'distance')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', 
                                 height=20, selectmode='browse')
        
        self.tree.heading('rank', text='Rank', anchor='center')
        self.tree.heading('name', text='Signal Name', anchor='center')
        self.tree.heading('distance', text='Euclidean Distance', anchor='center')
        
        self.tree.column('rank', width=60, anchor='center')
        self.tree.column('name', width=150, anchor='center')
        self.tree.column('distance', width=150, anchor='center')
        
        # Scrollbar
        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right side - Plot
        plot_frame = ttk.Frame(results_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Test signal plot
        test_plot = ttk.LabelFrame(plot_frame, text="Test Signal", padding="10")
        test_plot.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.test_fig, self.test_ax = plt.subplots(figsize=(5, 2.5), facecolor=self.bg_color)
        self.test_ax.set_xlabel('Sample')
        self.test_ax.set_ylabel('Amplitude')
        self.test_ax.grid(True, alpha=0.3)
        self.test_ax.set_title('')
        
        self.test_canvas = FigureCanvasTkAgg(self.test_fig, master=test_plot)
        self.test_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Matching signal plot
        match_plot = ttk.LabelFrame(plot_frame, text="Matching Signal", padding="10")
        match_plot.pack(fill=tk.BOTH, expand=True)
        
        self.match_fig, self.match_ax = plt.subplots(figsize=(5, 2.5), facecolor=self.bg_color)
        self.match_ax.set_xlabel('Sample')
        self.match_ax.set_ylabel('Amplitude')
        self.match_ax.grid(True, alpha=0.3)
        self.match_ax.set_title('')
        
        self.match_canvas = FigureCanvasTkAgg(self.match_fig, master=match_plot)
        self.match_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_horizontal(self):
        filename = filedialog.askopenfilename(
            title="Select Horizontal Signals Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.horizontal_file = filename
            self.horizontal_label.config(text=os.path.basename(filename), foreground='black')
            self.check_files()
    
    def load_test(self):
        filename = filedialog.askopenfilename(
            title="Select Test Signal Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.test_file = filename
            self.test_label.config(text=os.path.basename(filename), foreground='black')
            self.check_files()
    
    def check_files(self):
        if self.horizontal_file and self.test_file:
            self.process_btn.config(state='normal')
            self.status_label.config(text="Files loaded. Ready to process.")
    
    def process_data(self):
        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Clear plots
        self.test_ax.clear()
        self.match_ax.clear()
        
        # Show progress
        self.progress.pack(pady=10)
        self.progress.start()
        self.status_label.config(text="Processing...")
        self.process_btn.config(state='disabled')
        self.root.update()
        
        try:
            # Run analysis
            results = self.analyzer.run_complete_analysis(self.horizontal_file, self.test_file)
            self.results = results
            
            # Display results in treeview
            for i, (name, idx, dist) in enumerate(results['distances']):
                self.tree.insert('', tk.END, values=(i+1, name, f"{dist:.6f}"))
            
            # Highlight the closest match
            children = self.tree.get_children()
            if children:
                self.tree.selection_set(children[0])
                self.tree.focus(children[0])
            
            # Plot test signal
            self.test_ax.plot(self.analyzer.test_signal_raw, color='#4a7a9c', linewidth=1.5)
            self.test_ax.set_xlabel('Sample')
            self.test_ax.set_ylabel('Amplitude')
            self.test_ax.grid(True, alpha=0.3)
            self.test_ax.set_title('Test Signal')
            self.test_canvas.draw()
            
            # Plot matching signal (ONLY the closest match)
            self.match_ax.plot(self.analyzer.matching_signal_raw, color='#27ae60', linewidth=1.5)
            self.match_ax.set_xlabel('Sample')
            self.match_ax.set_ylabel('Amplitude')
            self.match_ax.grid(True, alpha=0.3)
            self.match_ax.set_title(f'Closest Match: {results["closest_signal"]}')
            self.match_canvas.draw()
            
            self.status_label.config(
                text=f"Complete! Closest match: {results['closest_signal']} (Distance: {results['closest_distance']:.6f})"
            )
            
            # Show success message
            messagebox.showinfo(
                "Analysis Complete", 
                f"Analysis finished successfully!\n\n"
                f"Features matrix saved to: features_matrix.xlsx\n"
                f"Closest match: {results['closest_signal']}\n"
                f"Euclidean distance: {results['closest_distance']:.6f}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
            self.status_label.config(text="Processing failed")
        
        finally:
            self.progress.stop()
            self.progress.pack_forget()
            self.process_btn.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg='#dcdad5')
    app = EOGTaskGUI(root)
    root.mainloop()