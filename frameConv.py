import cv2
import numpy as np
import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

class FrameConversionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Frame Conversion Operator")
        self.root.geometry("500x520")
        self.root.resizable(False, False)

        # Variables
        self.input_files = [] # Store list of selected files
        self.input_display_text = tk.StringVar() # Text to show in the entry box
        self.output_path = tk.StringVar()
        self.frame_skip = tk.IntVar(value=1)
        self.mode = tk.StringVar(value="average")
        self.status_text = tk.StringVar(value="Ready")
        self.progress_val = tk.DoubleVar(value=0)
        self.is_processing = False

        self.create_widgets()

    def create_widgets(self):
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Frame Conversion Operator", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 20))

        # Input File Selection
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Input (Video, GIF, or Multiple Images):").pack(anchor=tk.W)
        
        # Entry is readonly because we manage the text manually based on multiple file selection
        input_entry = ttk.Entry(input_frame, textvariable=self.input_display_text, width=40, state="readonly")
        input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(side=tk.RIGHT)

        # Output File Selection
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=5)
        ttk.Label(output_frame, text="Output Image:").pack(anchor=tk.W)
        
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=40)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(output_frame, text="Save As", command=self.browse_output).pack(side=tk.RIGHT)

        # Mode Selection
        mode_frame = ttk.LabelFrame(main_frame, text="Processing Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=10)
        
        ttk.Radiobutton(mode_frame, text="Average (Silky Water/Clouds)", variable=self.mode, value="average").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Brightest (Light Trails/Star Trails)", variable=self.mode, value="brightest").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Darkest (Remove Bright Objects)", variable=self.mode, value="darkest").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Median (Ghost/Tourist Removal)", variable=self.mode, value="median").pack(anchor=tk.W)

        # Settings
        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(fill=tk.X, pady=5)
        ttk.Label(settings_frame, text="Frame Skip:").pack(side=tk.LEFT)
        skip_spinbox = ttk.Spinbox(settings_frame, from_=1, to=100, textvariable=self.frame_skip, width=5)
        skip_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(settings_frame, text="(1 = Best Quality, Higher = Faster)").pack(side=tk.LEFT, padx=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_val, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(20, 5))

        # Status Label
        status_label = ttk.Label(main_frame, textvariable=self.status_text, font=("Helvetica", 9))
        status_label.pack(pady=(0, 10))

        # Action Buttons
        self.start_button = ttk.Button(main_frame, text="Start Processing", command=self.start_thread)
        self.start_button.pack(fill=tk.X, pady=5)

    def browse_input(self):
        filenames = filedialog.askopenfilenames(
            filetypes=[
                ("All Supported", "*.mp4 *.avi *.mov *.mkv *.gif *.png *.jpg *.jpeg *.bmp"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.gif"),
                ("Images", "*.png *.jpg *.jpeg *.bmp")
            ]
        )
        
        if filenames:
            self.input_files = filenames
            count = len(filenames)
            if count == 1:
                self.input_display_text.set(filenames[0])
            else:
                self.input_display_text.set(f"{count} images selected")

            folder = os.path.dirname(filenames[0])
            self.output_path.set(os.path.join(folder, f"processed_result.png"))

    def browse_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
        if filename:
            self.output_path.set(filename)

    def start_thread(self):
        if self.is_processing:
            return

        files = self.input_files
        output_file = self.output_path.get()
        step = self.frame_skip.get()
        mode = self.mode.get()

        if not files:
            messagebox.showerror("Error", "Please select input video, GIF, or images.")
            return
        if not output_file:
            messagebox.showerror("Error", "Please select an output path.")
            return

        self.start_button.config(state=tk.DISABLED)
        self.is_processing = True
        self.progress_val.set(0)

        thread = threading.Thread(target=self.process_logic, args=(files, output_file, step, mode))
        thread.daemon = True
        thread.start()

    def process_logic(self, source_files, output_path, step, mode):
        cap = None
        try:
            self.update_status(f"Mode: {mode.title()} - Initializing...")
            
            # --- Check Source Type ---
            is_video = False
            if len(source_files) == 1:
                ext = os.path.splitext(source_files[0])[1].lower()
                # Added .gif here to treat it as a video source
                if ext in ['.mp4', '.avi', '.mov', '.mkv', '.gif']:
                    is_video = True

            # --- Get Dimensions & Count ---
            width, height, total_frames = 0, 0, 0
            
            if is_video:
                cap = cv2.VideoCapture(source_files[0])
                if not cap.isOpened():
                    raise Exception("Could not open video/GIF.")
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Fix for GIFs or streams that don't report frame count
                if total_frames <= 0:
                    self.update_status("Calculating frame count (required for GIF)...")
                    # Manually count frames
                    count = 0
                    while True:
                        ret, _ = cap.read()
                        if not ret: break
                        count += 1
                    total_frames = count
                    
                    # Reset to start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                    # If reset fails (some GIF backends), reopen
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) != 0:
                        cap.release()
                        cap = cv2.VideoCapture(source_files[0])

                cap.release() # Release to reopen cleanly later
            else:
                sample = cv2.imread(source_files[0])
                if sample is None: raise Exception("Could not read first image.")
                height, width, _ = sample.shape
                total_frames = len(source_files)

            if total_frames == 0:
                raise Exception("No valid frames found in input.")

            # --- Processing Logic Split ---
            
            if mode == "median":
                self.process_median_memory_safe(source_files, is_video, output_path, step, width, height, total_frames)
            else:
                self.process_incremental(source_files, is_video, output_path, step, mode, total_frames)
                
        except Exception as e:
            print(f"Error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred:\n{str(e)}"))
            self.update_status("Error occurred.")
        finally:
            if cap and is_video and cap.isOpened():
                cap.release()
            self.is_processing = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def process_incremental(self, source_files, is_video, output_path, step, mode, total_frames):
        """ Handles Average, Brightest, Darkest (Low Memory) """
        
        # Generator for frames
        def frame_gen():
            if is_video:
                cap = cv2.VideoCapture(source_files[0])
                current = 0
                while True:
                    if step > 1:
                        for _ in range(step - 1):
                            cap.read()
                            current += 1
                    ret, frame = cap.read()
                    if not ret: break
                    yield frame, current
                    current += 1
                cap.release()
            else:
                for i in range(0, len(source_files), step):
                    frame = cv2.imread(source_files[i])
                    if frame is not None:
                        yield frame, i

        buffer = None
        count = 0
        
        for frame, idx in frame_gen():
            # Update UI
            if count % 10 == 0:
                prog = (idx / max(1, total_frames)) * 100
                self.update_progress(prog, f"Processing frame {idx}/{total_frames}")

            if buffer is None:
                if mode == "average":
                    buffer = frame.astype(np.float64)
                else:
                    buffer = frame.copy()
            else:
                # Ensure shapes match (important for image sequences of varying sizes)
                if frame.shape != buffer.shape:
                    frame = cv2.resize(frame, (buffer.shape[1], buffer.shape[0]))

                if mode == "average":
                    buffer += frame.astype(np.float64)
                elif mode == "brightest":
                    buffer = np.maximum(buffer, frame)
                elif mode == "darkest":
                    buffer = np.minimum(buffer, frame)
            count += 1

        # Save result
        self.update_status("Finalizing...")
        if count == 0:
            raise Exception("No frames processed.")

        if mode == "average":
            result = (buffer / count).astype(np.uint8)
        else:
            result = buffer
            
        cv2.imwrite(output_path, result)
        self.update_progress(100, "Done!")
        self.root.after(0, lambda: messagebox.showinfo("Success", f"Saved to:\n{output_path}"))

    def process_median_memory_safe(self, source_files, is_video, output_path, step, width, height, total_frames):
        """ 
        Calculates Median by splitting image into horizontal strips.
        This iterates over the video multiple times but uses fixed memory.
        """
        
        # 1. Determine Memory Limits
        # Target RAM usage: ~512 MB for the buffer
        TARGET_RAM_BYTES = 512 * 1024 * 1024 
        
        # How many frames are we actually processing?
        frames_to_process = max(1, total_frames // step)
        
        # Bytes needed for 1 single row across all processed frames
        # (Frames * Width * 3 channels * 1 byte)
        bytes_per_row_stack = frames_to_process * width * 3
        
        # Calculate safe batch height (number of rows to process at once)
        batch_height = int(TARGET_RAM_BYTES / max(1, bytes_per_row_stack))
        batch_height = max(1, min(batch_height, height)) # Clamp between 1 and Image Height
        
        print(f"Memory Safe Mode: Processing {batch_height} rows per batch.")
        
        # Prepare final image buffer (this is small, just one image)
        final_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        total_batches = (height + batch_height - 1) // batch_height
        
        # 2. Iterate Batches (Strips of the image)
        for i, y_start in enumerate(range(0, height, batch_height)):
            y_end = min(y_start + batch_height, height)
            current_strip_height = y_end - y_start
            
            self.update_status(f"Median Pass {i+1}/{total_batches}: Reading frames...")
            self.update_progress((i / total_batches) * 100, f"Batch {i+1}/{total_batches}: Collecting frames...")

            # Buffer for this specific strip: (N_Frames, Strip_Height, Width, 3)
            # We use a list first, then numpy, to avoid pre-allocation issues if frame count varies slightly
            strip_stack = [] 

            # --- Read Video/Images for this strip ---
            if is_video:
                cap = cv2.VideoCapture(source_files[0])
                frame_idx = 0
                while True:
                    if step > 1:
                        for _ in range(step - 1): cap.read()
                    
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Crop only the strip we need
                    try:
                        strip = frame[y_start:y_end, :, :]
                        strip_stack.append(strip)
                    except:
                        pass # Handle rare frame errors
                    frame_idx += 1
                cap.release()
            else:
                for f_idx in range(0, len(source_files), step):
                    frame = cv2.imread(source_files[f_idx])
                    if frame is not None:
                        # Resize if sequence has varying sizes
                        if frame.shape[:2] != (height, width):
                            frame = cv2.resize(frame, (width, height))
                        
                        strip = frame[y_start:y_end, :, :]
                        strip_stack.append(strip)

            # --- Calculate Median for this strip ---
            if not strip_stack:
                continue

            self.update_status(f"Batch {i+1}/{total_batches}: Calculating median...")
            
            # Convert to numpy array (N, H, W, 3)
            stack_arr = np.array(strip_stack)
            
            # Median along axis 0 (time)
            median_strip = np.median(stack_arr, axis=0).astype(np.uint8)
            
            # Store in final image
            final_image[y_start:y_end, :, :] = median_strip
            
            # Clear memory explicitly
            del stack_arr
            del strip_stack

        # 3. Save
        cv2.imwrite(output_path, final_image)
        self.update_progress(100, "Done!")
        self.root.after(0, lambda: messagebox.showinfo("Success", f"Saved to:\n{output_path}"))

    def update_progress(self, value, text):
        self.root.after(0, lambda: self.progress_val.set(value))
        self.update_status(text)

    def update_status(self, text):
        self.root.after(0, lambda: self.status_text.set(text))

if __name__ == "__main__":

    from PIL import ImageTk
    import base64
    root = tk.Tk()
    data = 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABuUlEQVR4AcxQu0tCURj/zlXXyguR0YsgoiGqoWyKyqBV0wbp1pq2GC1BQ0tZYi46ZCVlD0qF/oAe5KOxKdJIbYm2pjDafJ2+e9SraNDg0uX8zvkev9/vfudwUOf3Tw1UXd1hXtV+x6s6i2gP8q2dN1gLNrV0RFp6+/Olm9dcoXd0LP3x/jYp+F6n5n0vGuE8PiVcJMVzGmuaBX9iXNXWQWoMBoeH1T0jY/nxZa9MbIY3DNSpFIhSiRnNw+mcGoBSBID0e2yVJ2hofkhq1CQHlLlPrF+SoZU0pD5zELHOguXWAvdWPUooEMbAEFfZgFBMGcQNSQTEILxlBJ1LD6JG65xBMwMOInaQjksyoIwCwDEZQCN/RnUuHYq1SINiFUDnRDMiyZDP2rhRBorXJBiBdWiJ2xw0kRKsLDazvGwHZQMiqoCQI6GPW/TFsyaEORDPmPyJrCmQRCQyjIJbNHRVoGMszUJxyK/U90B+bTW7H/PC3vMJ3X3yEnfsGNyPh3Bg7FMgv2ZJBtHQNeE9nhhn25FzNoec27YrZDaHQiaedoe8RlksSAbFvPoovEx1tSL/y6CC+ntYt8EPAAAA///NbPDCAAAABklEQVQDACqqlSHtXZKDAAAAAElFTkSuQmCC'
    icon = ImageTk.PhotoImage(data=base64.b64decode(data))
    root.iconphoto(True, icon)
    app = FrameConversionApp(root)
    root.mainloop()