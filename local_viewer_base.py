import tyro
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
from pathlib import Path
import time
import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image
import threading
import queue
import traceback
import signal
import sys
import math


@dataclass
class CameraConfig:
    """Camera configuration for viewer"""
    fov: float = 45.0  # Field of view in degrees
    near: float = 0.1  # Near clipping plane
    far: float = 1000.0  # Far clipping plane
    position: Tuple[float, float, float] = (0.0, 0.0, 5.0)  # Initial camera position
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Initial camera target
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # Camera up vector


class Camera:
    """Simple camera class for managing view transformations"""
    def __init__(self, config: CameraConfig):
        self.config = config
        self.position = np.array(config.position, dtype=np.float32)
        self.target = np.array(config.target, dtype=np.float32)
        self.up = np.array(config.up, dtype=np.float32)
        self.fov = config.fov
        self.near = config.near
        self.far = config.far
        
        # Camera orientation controls
        self.radius = np.linalg.norm(self.position - self.target)
        self.theta = math.atan2(self.position[0], self.position[2])
        self.phi = math.asin(self.position[1] / self.radius)
        
        # Update matrices
        self.update()
    
    def update(self):
        """Update camera matrices based on current parameters"""
        # Update position from spherical coordinates
        self.position[0] = self.radius * math.cos(self.phi) * math.sin(self.theta)
        self.position[1] = self.radius * math.sin(self.phi)
        self.position[2] = self.radius * math.cos(self.phi) * math.cos(self.theta)
        
        # Calculate view matrix
        self.view_matrix = self._look_at(self.position, self.target, self.up)
        
        # Calculate projection matrix
        self.proj_matrix = self._perspective(self.fov, 1.0, self.near, self.far)
        
        # Combined matrix
        self.view_proj_matrix = np.matmul(self.proj_matrix, self.view_matrix)
    
    def _look_at(self, eye, target, up):
        """Create a view matrix from eye, target and up vectors"""
        # Calculate forward vector
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        # Calculate right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Calculate up vector
        up = np.cross(right, forward)
        
        # Create view matrix
        view_matrix = np.identity(4, dtype=np.float32)
        view_matrix[0, 0:3] = right
        view_matrix[1, 0:3] = up
        view_matrix[2, 0:3] = -forward
        
        # Translation
        view_matrix[0, 3] = -np.dot(right, eye)
        view_matrix[1, 3] = -np.dot(up, eye)
        view_matrix[2, 3] = np.dot(forward, eye)
        
        return view_matrix
    
    def _perspective(self, fov_deg, aspect, near, far):
        """Create a perspective projection matrix"""
        fov_rad = math.radians(fov_deg)
        f = 1.0 / math.tan(fov_rad / 2.0)
        
        proj_matrix = np.zeros((4, 4), dtype=np.float32)
        proj_matrix[0, 0] = f / aspect
        proj_matrix[1, 1] = f
        proj_matrix[2, 2] = (far + near) / (near - far)
        proj_matrix[2, 3] = (2.0 * far * near) / (near - far)
        proj_matrix[3, 2] = -1.0
        
        return proj_matrix
    
    def rotate(self, delta_x, delta_y):
        """Rotate camera around target point"""
        self.theta += delta_x * 0.01
        self.phi += delta_y * 0.01
        
        # Clamp phi to avoid gimbal lock
        self.phi = max(min(self.phi, math.pi/2 - 0.1), -math.pi/2 + 0.1)
        
        self.update()
    
    def zoom(self, delta):
        """Zoom camera by changing radius"""
        self.radius *= (1.0 - delta * 0.1)
        self.radius = max(0.1, min(self.radius, 100.0))
        self.update()
    
    def get_position(self):
        """Get camera position as list"""
        return self.position.tolist()
    
    def get_view_matrix(self):
        """Get view matrix as list of lists"""
        return self.view_matrix.tolist()
    
    def get_projection_matrix(self):
        """Get projection matrix as list of lists"""
        return self.proj_matrix.tolist()
    
    def reset(self):
        """Reset camera to initial configuration"""
        self.position = np.array(self.config.position, dtype=np.float32)
        self.target = np.array(self.config.target, dtype=np.float32)
        self.up = np.array(self.config.up, dtype=np.float32)
        self.fov = self.config.fov
        
        # Reset orientation
        self.radius = np.linalg.norm(self.position - self.target)
        self.theta = math.atan2(self.position[0], self.position[2])
        self.phi = math.asin(self.position[1] / self.radius)
        
        self.update()


@dataclass
class GenericViewerConfig:
    """Configuration for the generic image viewer"""
    width: int = 1280
    """Window width"""
    height: int = 720
    """Window height"""
    save_folder: Path = Path("./viewer_output")
    """Default save folder"""
    fps: int = 30
    """Default playback framerate"""
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Default background color"""
    window_title: str = "Gaussian Splatting Viewer"
    """Window title"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    """Camera configuration"""


class GenericImageViewer:
    def __init__(self, cfg: GenericViewerConfig):
        self.cfg = cfg
        
        # UI parameters
        self.W = self.cfg.width
        self.H = self.cfg.height
        self.title = self.cfg.window_title
        
        # Playback controls
        self.playing = False
        self.playback_speed = 1.0
        self.last_frame_time = time.time()
        self.timestep = 0
        self.num_frames = 0
        
        # Image queue and cache
        self.image_queue = queue.Queue(maxsize=10)  # External renderer puts images here
        self.image_frames = {}  # Cache of rendered frames {frame_idx: image_array}
        
        # Camera setup
        self.camera = Camera(self.cfg.camera)
        self.mouse_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Camera change callback
        self.camera_change_callback = None
        
        # State control
        self.thread_running = True
        self.last_time_fresh = None
        self.render_buffer = None
        
        # Set up signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Initialize UI
        self.setup_ui()

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C, safely close the program"""
        print("Exiting gracefully...")
        self.thread_running = False
        sys.exit(0)
        
    def setup_ui(self):
        """Initialize DearPyGui interface"""
        dpg.create_context()
        dpg.create_viewport(title=self.title, width=self.W, height=self.H)
        dpg.setup_dearpygui()
        
        # Create texture
        with dpg.texture_registry():
            dpg.add_raw_texture(self.W, self.H, np.ones((self.H, self.W, 3)) * 0.1, 
                               format=dpg.mvFormat_Float_rgb, tag="_texture")
        
        # Create main window
        with dpg.window(label="Main View", tag="_main_window"):
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_main_view")
            
        self.define_gui()
        
        # Setup mouse interaction handlers for camera control
        with dpg.item_handler_registry(tag="_camera_handler"):
            dpg.add_item_clicked_handler(callback=self.handle_mouse_click)
            dpg.add_item_double_clicked_handler(callback=self.handle_mouse_double_click)
        
        dpg.bind_item_handler_registry("_main_view", "_camera_handler")
        
        # Set window
        dpg.set_primary_window("_main_window", True)
        dpg.show_viewport()
        
    def define_gui(self):
        """Define UI elements"""
        # Control window
        with dpg.window(label="Controls", tag="_control_window", pos=[10, 10], autosize=True):
            with dpg.group(horizontal=True):
                dpg.add_text("FPS:")
                dpg.add_text("0   ", tag="_log_fps")
                dpg.add_spacer(width=10)
                dpg.add_text("Frame:")
                dpg.add_text("0", tag="_log_current_frame")
                
            # Frame controls
            with dpg.group(horizontal=True):
                def callback_set_current_frame(sender, app_data):
                    if sender == "_slider_timestep":
                        self.request_frame(app_data)
                    elif sender in ["_button_timestep_minus", "_mvKey_Left"]:
                        self.request_frame(max(self.timestep - 1, 0))
                    elif sender in ["_button_timestep_plus", "_mvKey_Right"]:
                        self.request_frame(min(self.timestep + 1, self.num_frames - 1))
                    elif sender == "_mvKey_Home":
                        self.request_frame(0)
                    elif sender == "_mvKey_End":
                        self.request_frame(self.num_frames - 1)
                    
                dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
                dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)
                dpg.add_slider_int(label="Frame", tag='_slider_timestep', width=200, min_value=0, 
                                  max_value=self.num_frames - 1 if self.num_frames > 0 else 0, 
                                  format="%d", default_value=0, callback=callback_set_current_frame)
            
            # Playback controls
            with dpg.group(horizontal=True):
                def callback_play_pause(sender, app_data):
                    self.playing = not self.playing
                    if self.playing:
                        dpg.set_item_label("_button_play_pause", "Pause")
                        self.last_frame_time = time.time()
                    else:
                        dpg.set_item_label("_button_play_pause", "Play")
                
                dpg.add_button(label="Play", tag="_button_play_pause", callback=callback_play_pause)
                
                # Loop option
                dpg.add_checkbox(label="Loop", default_value=True, tag="_checkbox_loop")
            
            # Playback speed
            with dpg.group(horizontal=True):
                def callback_set_speed(sender, app_data):
                    self.playback_speed = app_data
                
                dpg.add_slider_float(label="Speed", min_value=0.1, max_value=2.0, 
                                    format="%.1fx", default_value=1.0, 
                                    callback=callback_set_speed, tag="_slider_speed", width=200)
            
            # Camera controls section
            with dpg.collapsing_header(label="Camera Controls", default_open=True):
                # Camera position display
                dpg.add_text("Camera Position:")
                with dpg.group(horizontal=True):
                    dpg.add_text("X:", tag="_camera_pos_x_label")
                    dpg.add_text("0.00", tag="_camera_pos_x")
                    dpg.add_text("Y:", tag="_camera_pos_y_label")
                    dpg.add_text("0.00", tag="_camera_pos_y")
                    dpg.add_text("Z:", tag="_camera_pos_z_label")
                    dpg.add_text("0.00", tag="_camera_pos_z")
                
                # Camera controls
                def callback_set_fov(sender, app_data):
                    self.camera.fov = app_data
                    self.camera.update()
                    self.notify_camera_change()
                
                dpg.add_slider_float(label="FOV", min_value=10.0, max_value=120.0, 
                                   format="%.1f°", default_value=self.camera.fov, 
                                   callback=callback_set_fov, tag="_slider_fov", width=200)
                
                # Camera reset button
                def callback_reset_camera(sender, app_data):
                    self.camera.reset()
                    self.update_camera_display()
                    self.notify_camera_change()
                
                dpg.add_button(label="Reset Camera", tag="_button_reset_camera", 
                             callback=callback_reset_camera)
                
                dpg.add_text("Camera Control: Drag to rotate, scroll to zoom")
            
            # Save image button
            def callback_save_image(sender, app_data):
                if not self.cfg.save_folder.exists():
                    self.cfg.save_folder.mkdir(parents=True)
                path = self.cfg.save_folder / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
                print(f"Saving image to: {path}")
                if self.render_buffer is not None:
                    Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
            dpg.add_button(label="Save Image", tag="_button_save_image", callback=callback_save_image)
                
        # Add keyboard event handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')
            
            # Add mouse wheel handler for frame control and zoom
            def callback_mouse_wheel(sender, app_data):
                delta = app_data
                if dpg.is_item_hovered("_slider_timestep"):
                    new_frame = min(max(self.timestep - delta, 0), self.num_frames - 1)
                    self.request_frame(new_frame)
                elif dpg.is_item_hovered("_main_view"):
                    # Zoom camera
                    self.camera.zoom(delta)
                    self.update_camera_display()
                    self.notify_camera_change()
            dpg.add_mouse_wheel_handler(callback=callback_mouse_wheel)
            
            # Add mouse drag handler for camera rotation
            def callback_mouse_drag(sender, app_data):
                if self.mouse_dragging and dpg.is_item_hovered("_main_view"):
                    mouse_pos = dpg.get_mouse_pos()
                    delta_x = mouse_pos[0] - self.last_mouse_x
                    delta_y = mouse_pos[1] - self.last_mouse_y
                    
                    self.camera.rotate(delta_x, delta_y)
                    self.update_camera_display()
                    self.notify_camera_change()
                    
                    self.last_mouse_x = mouse_pos[0]
                    self.last_mouse_y = mouse_pos[1]
            dpg.add_mouse_drag_handler(callback=callback_mouse_drag)
            
        # Initialize camera display
        self.update_camera_display()

    def handle_mouse_click(self, sender, app_data):
        """Handle mouse click for camera control"""
        if app_data[1] and dpg.is_item_hovered("_main_view"):  # Left button
            self.mouse_dragging = True
            self.last_mouse_x, self.last_mouse_y = dpg.get_mouse_pos()
    
    def handle_mouse_double_click(self, sender, app_data):
        """Handle mouse double click"""
        if app_data[1] and dpg.is_item_hovered("_main_view"):  # Left button
            # Reset camera on double click
            self.camera.reset()
            self.update_camera_display()
            self.notify_camera_change()
    
    def update_camera_display(self):
        """Update camera information display in UI"""
        pos = self.camera.get_position()
        dpg.set_value("_camera_pos_x", f"{pos[0]:.2f}")
        dpg.set_value("_camera_pos_y", f"{pos[1]:.2f}")
        dpg.set_value("_camera_pos_z", f"{pos[2]:.2f}")
        dpg.set_value("_slider_fov", self.camera.fov)
    
    def set_camera_change_callback(self, callback: Callable):
        """Set callback function to be called when camera changes
        
        The callback function should accept these parameters:
        - position: list of 3 floats
        - view_matrix: 4x4 list of lists
        - proj_matrix: 4x4 list of lists
        """
        self.camera_change_callback = callback
    
    def notify_camera_change(self):
        """Notify external renderer of camera change"""
        if self.camera_change_callback:
            self.camera_change_callback(
                self.camera.get_position(),
                self.camera.get_view_matrix(),
                self.camera.get_projection_matrix()
            )

    def refresh_stat(self):
        """Refresh FPS statistics"""
        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed if elapsed > 0 else 0
            dpg.set_value("_log_fps", f'{int(fps):<4d}')
        self.last_time_fresh = time.time()
    
    def request_frame(self, frame_idx):
        """Request display of a specific frame"""
        if frame_idx < 0 or (self.num_frames > 0 and frame_idx >= self.num_frames):
            return
            
        self.timestep = frame_idx
        dpg.set_value("_slider_timestep", frame_idx)
        dpg.set_value("_log_current_frame", f"{frame_idx}")
        
        # Check if this frame is already cached
        if frame_idx in self.image_frames:
            self.update_display(self.image_frames[frame_idx])
        
    def update_display(self, image_data):
        """Update displayed image"""
        if image_data is None or not isinstance(image_data, np.ndarray):
            return
            
        if image_data.shape[0] == self.H and image_data.shape[1] == self.W:
            self.render_buffer = image_data
            dpg.set_value("_texture", image_data)
            self.refresh_stat()
    
    def set_num_frames(self, num_frames):
        """Set total number of frames"""
        self.num_frames = max(0, num_frames)
        dpg.configure_item("_slider_timestep", max_value=self.num_frames - 1 if self.num_frames > 0 else 0)
    
    def add_frame(self, frame_idx, image_data):
        """Add frame to cache"""
        if not isinstance(image_data, np.ndarray):
            print(f"Error: Invalid image data type: {type(image_data)}")
            return False
            
        # Ensure image dimensions are correct
        if image_data.shape[0] != self.H or image_data.shape[1] != self.W:
            print(f"Error: Image dimensions mismatch, expected {self.H}x{self.W}, got {image_data.shape[0]}x{image_data.shape[1]}")
            return False
            
        # Update cache
        self.image_frames[frame_idx] = image_data.copy()
        
        # If this is the current frame, update display
        if frame_idx == self.timestep:
            self.update_display(image_data)
        
        # Update total frame count
        if frame_idx >= self.num_frames:
            self.set_num_frames(frame_idx + 1)
            
        return True
    
    def clear_cache(self):
        """Clear image cache"""
        self.image_frames.clear()
    
    def put_frame_to_queue(self, frame_idx, image_data):
        """External interface: Put frame into queue"""
        try:
            self.image_queue.put((frame_idx, image_data), block=False)
            return True
        except queue.Full:
            print("Warning: Image queue full, dropping frame")
            return False
            
    def run(self):
        """Main loop"""
        print("Starting Generic Image Viewer...")
        
        while dpg.is_dearpygui_running() and self.thread_running:
            current_time = time.time()
            
            # Handle mouse release for camera control
            if self.mouse_dragging and not dpg.is_mouse_button_down(0):  # Left button released
                self.mouse_dragging = False
            
            # Process images in queue
            try:
                while not self.image_queue.empty():
                    frame_idx, image_data = self.image_queue.get(block=False)
                    self.add_frame(frame_idx, image_data)
            except Exception as e:
                print(f"Error processing image queue: {e}")
            
            # Handle playback logic
            if self.playing and self.num_frames > 1:
                fps_target = self.cfg.fps * self.playback_speed
                time_since_last_frame = current_time - self.last_frame_time
                
                if time_since_last_frame >= 1.0 / fps_target:
                    self.last_frame_time = current_time
                    
                    next_frame = self.timestep + 1
                    
                    # Handle loop logic
                    if next_frame >= self.num_frames:
                        if dpg.get_value("_checkbox_loop"):
                            next_frame = 0
                        else:
                            self.playing = False
                            dpg.set_item_label("_button_play_pause", "Play")
                            continue
                    
                    self.request_frame(next_frame)
            
            # Render UI
            try:
                dpg.render_dearpygui_frame()
            except Exception as e:
                print(f"Error rendering UI frame: {e}")
                time.sleep(0.1)
            
            # Sleep to reduce CPU load
            time.sleep(0.005)
        
        # Cleanup resources
        dpg.destroy_context()
        print("Image viewer closed")

# Example usage
if __name__ == "__main__":
    cfg = tyro.cli(GenericViewerConfig)
    viewer = GenericImageViewer(cfg)
    
    # Example: Handle camera changes
    def on_camera_change(position, view_matrix, projection_matrix):
        print(f"Camera changed: Position = {position}")
        # This would typically send the updated camera to your GS renderer
    
    viewer.set_camera_change_callback(on_camera_change)
    
    # Example: Generate test frames
    def generate_test_frames():
        print("Generating test frames...")
        import time
        import numpy as np
        
        # Create 100 test frames
        for i in range(100):
            # Generate a gradient image
            img = np.ones((cfg.height, cfg.width, 3))
            
            # Add moving colored square
            t = i / 10.0
            x = int((cfg.width - 100) * (0.5 + 0.5 * np.sin(t)))
            y = int((cfg.height - 100) * (0.5 + 0.5 * np.cos(t)))
            
            # Draw colored square
            img[y:y+100, x:x+100, 0] = np.sin(t) * 0.5 + 0.5
            img[y:y+100, x:x+100, 1] = np.cos(t) * 0.5 + 0.5
            img[y:y+100, x:x+100, 2] = np.sin(t + 1.5) * 0.5 + 0.5
            
            # Add frame number
            viewer.put_frame_to_queue(i, img)
            
            # Wait a bit to simulate rendering time
            time.sleep(0.01)
            
        print("Test frame generation complete")
    
    # Run test frame generation in a separate thread
    import threading
    test_thread = threading.Thread(target=generate_test_frames)
    test_thread.daemon = True
    test_thread.start()
    
    # Run main loop
    viewer.run() 