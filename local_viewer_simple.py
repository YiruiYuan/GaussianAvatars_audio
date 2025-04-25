# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import json
import math
import tyro
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path
import time
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
import threading
import queue
import signal
import sys

from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig
from gaussian_renderer import GaussianModel, FlameGaussianModel
from gaussian_renderer import render
from mesh_renderer import NVDiffRenderer


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config(Mini3DViewerConfig):
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    """Pipeline settings for gaussian splatting rendering"""
    cam_convention: Literal["opengl", "opencv"] = "opencv"
    """Camera convention"""
    point_path: Optional[Path] = None
    """Path to the gaussian splatting file"""
    motion_path: Optional[Path] = None
    """Path to the motion file (npz)"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    background_color: tuple[float, float, float] = (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""
    fps: int = 25
    """default fps for playback"""
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""

class LocalViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # playback settings
        self.playing = False
        self.need_frame_update = False
        self.next_frame = 0
        
        # Thread synchronization
        self.render_queue = queue.Queue(maxsize=1)
        self.is_rendering = False
        self.thread_running = True
        self.render_thread = None
        self.abort_render = False
        
        print("Initializing 3D Gaussians...")
        self.init_gaussians()

        if self.gaussians.binding is not None:
            # rendering settings
            self.mesh_color = torch.tensor([1, 1, 1, 0.5])
            self.face_colors = None
            print("Initializing mesh renderer...")
            self.mesh_renderer = NVDiffRenderer(use_opengl=False)
        
        super().__init__(cfg, 'GaussianAvatars - Local Viewer')

        if self.gaussians.binding is not None:
            self.num_timesteps = self.gaussians.num_timesteps
            dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)
            self.gaussians.select_mesh_by_timestep(self.timestep)
            
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C to safely exit the program"""
        self.thread_running = False
        sys.exit(0)

    def init_gaussians(self):
        # load gaussians
        if self.cfg.point_path is not None and (Path(self.cfg.point_path).parent / "flame_param.npz").exists():
            self.gaussians = FlameGaussianModel(self.cfg.sh_degree)
        else:
            self.gaussians = GaussianModel(self.cfg.sh_degree)

        unselected_fid = []
        
        if self.cfg.point_path is not None:
            if self.cfg.point_path.exists():
                self.gaussians.load_ply(self.cfg.point_path, has_target=False, motion_path=self.cfg.motion_path, disable_fid=unselected_fid)
            else:
                raise FileNotFoundError(f'{self.cfg.point_path} does not exist.')

    def define_gui(self):
        super().define_gui()

        # window: rendering options
        with dpg.window(label="Render", tag="_render_window", autosize=True):
            with dpg.group(horizontal=True):
                dpg.add_text("Frame:")
                dpg.add_text("0", tag="_log_current_frame")

            dpg.add_text(f"number of points: {self.gaussians._xyz.shape[0]}")
            
            with dpg.group(horizontal=True):
                # show splatting
                def callback_show_splatting(sender, app_data):
                    self.need_update = True
                dpg.add_checkbox(label="show splatting", default_value=True, callback=callback_show_splatting, tag="_checkbox_show_splatting")

                dpg.add_spacer(width=10)

                if self.gaussians.binding is not None:
                    # show mesh
                    def callback_show_mesh(sender, app_data):
                        self.need_update = True
                    dpg.add_checkbox(label="show mesh", default_value=False, callback=callback_show_mesh, tag="_checkbox_show_mesh")
            
            # timestep slider and play controls
            if hasattr(self, 'num_timesteps') and self.num_timesteps is not None:
                def callback_set_current_frame(sender, app_data):
                    if sender == "_slider_timestep":
                        self.next_frame = app_data
                    elif sender in ["_button_timestep_plus", "_mvKey_Right"]:
                        self.next_frame = min(self.timestep + 1, self.num_timesteps - 1)
                    elif sender in ["_button_timestep_minus", "_mvKey_Left"]:
                        self.next_frame = max(self.timestep - 1, 0)
                    elif sender == "_mvKey_Home":
                        self.next_frame = 0
                    elif sender == "_mvKey_End":
                        self.next_frame = self.num_timesteps - 1

                    dpg.set_value("_slider_timestep", self.next_frame)
                    dpg.set_value("_log_current_frame", f"{self.next_frame}")
                    
                    self.need_frame_update = True
                    self.need_update = True

                with dpg.group(horizontal=True):
                    dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
                    dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)
                    dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=153, min_value=0, 
                                      max_value=self.num_timesteps - 1 if hasattr(self, 'num_timesteps') else 0, 
                                      format="%d", default_value=0, callback=callback_set_current_frame)
                
                # Play controls
                with dpg.group(horizontal=True):
                    def callback_play_pause(sender, app_data):
                        self.playing = not self.playing
                        if self.playing:
                            dpg.set_item_label("_button_play_pause", "Pause")
                            self.last_frame_time = time.time()
                        else:
                            dpg.set_item_label("_button_play_pause", "Play")
                    dpg.add_button(label="Play", tag="_button_play_pause", callback=callback_play_pause)
                    
                    # Loop checkbox
                    dpg.add_checkbox(label="Loop", default_value=True, tag="_checkbox_loop")

            # scaling_modifier slider
            def callback_set_scaling_modifier(sender, app_data):
                self.need_update = True
            dpg.add_slider_float(label="Scale modifier", min_value=0, max_value=1, format="%.2f", 
                                width=200, default_value=1, callback=callback_set_scaling_modifier, 
                                tag="_slider_scaling_modifier")

            # Camera controls
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    self.need_update = True
                dpg.add_button(label="Reset Camera", tag="_button_reset_pose", 
                              callback=callback_reset_camera)
                
                # Save image button
                def callback_save_image(sender, app_data):
                    if not self.cfg.save_folder.exists():
                        self.cfg.save_folder.mkdir(parents=True)
                    path = self.cfg.save_folder / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
                    if hasattr(self, 'render_buffer') and self.render_buffer is not None:
                        Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
                dpg.add_button(label="Save Image", tag="_button_save_image", callback=callback_save_image)

        # widget-dependent handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')
            
            # Add Escape key handler - abort rendering
            def callback_abort_key(sender, app_data):
                self.abort_render = True
            dpg.add_key_press_handler(dpg.mvKey_Escape, callback=callback_abort_key)

            def callbackmouse_wheel_slider(sender, app_data):
                delta = app_data
                if dpg.is_item_hovered("_slider_timestep"):
                    self.next_frame = min(max(self.timestep - delta, 0), self.num_timesteps - 1)
                    dpg.set_value("_slider_timestep", self.next_frame)
                    dpg.set_value("_log_current_frame", f"{self.next_frame}")
                    self.need_frame_update = True
                    self.need_update = True
            dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel_slider)

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    @torch.no_grad()
    def render_frame(self):
        """Render thread"""
        while self.thread_running:
            try:
                # Handle frame update requests
                if self.need_frame_update and not self.is_rendering:
                    try:
                        self.timestep = self.next_frame
                        self.gaussians.select_mesh_by_timestep(self.timestep)
                        self.need_frame_update = False
                        dpg.set_value("_log_current_frame", f"{self.timestep}")
                    except Exception as e:
                        self.need_frame_update = False
                
                # Execute rendering
                if self.need_update and not self.is_rendering:
                    # Start rendering
                    self.is_rendering = True
                    self.abort_render = False
                    
                    try:
                        # Get rendering parameters
                        show_splatting = dpg.get_value("_checkbox_show_splatting")
                        show_mesh = dpg.get_value("_checkbox_show_mesh") if self.gaussians.binding is not None else False
                        scaling_modifier = dpg.get_value("_slider_scaling_modifier")
                        
                        # Prepare camera
                        cam = self.prepare_camera()
                        
                        # Initialize with default background
                        rgb = torch.ones([self.H, self.W, 3])
                        
                        # Render gaussian point cloud
                        if show_splatting and not self.abort_render:
                            rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, 
                                                torch.tensor(self.cfg.background_color).cuda(), 
                                                scaling_modifier=scaling_modifier)["render"].permute(1, 2, 0).contiguous()
                        
                        # Render mesh
                        if show_mesh and not self.abort_render and self.gaussians.binding is not None:
                            out_dict = self.mesh_renderer.render_from_camera(
                                self.gaussians.verts, self.gaussians.faces, cam, face_colors=self.face_colors)
                            
                            rgba_mesh = out_dict['rgba'].squeeze(0)
                            rgb_mesh = rgba_mesh[:, :, :3]
                            alpha_mesh = rgba_mesh[:, :, 3:]
                            mesh_opacity = self.mesh_color[3:].cuda()
                        
                        # Composite final image
                        if not self.abort_render:
                            if show_splatting and show_mesh:
                                rgb = rgb_mesh * alpha_mesh * mesh_opacity + rgb_splatting * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                            elif show_splatting:
                                rgb = rgb_splatting
                            elif show_mesh:
                                rgb = rgb_mesh
                        
                        if not self.abort_render:
                            render_buffer = rgb.cpu().numpy()
                            if render_buffer.shape[0] == self.H and render_buffer.shape[1] == self.W:
                                try:
                                    self.render_queue.put(render_buffer, block=False)
                                except queue.Full:
                                    pass
                        
                    except Exception as e:
                        pass
                    
                    # Reset state regardless of outcome
                    self.need_update = False
                    self.is_rendering = False
                    self.abort_render = False
                else:
                    # No render task, brief sleep
                    time.sleep(0.01)
            
            except Exception as e:
                self.is_rendering = False
                self.need_update = False
                time.sleep(0.1)

    @torch.no_grad()
    def run(self):
        # Initialize playback variables
        self.last_frame_time = time.time()

        try:
            dpg.set_value("_log_current_frame", f"{self.timestep}")
        except:
            pass

        # Start render thread
        self.render_thread = threading.Thread(target=self.render_frame)
        self.render_thread.daemon = True
        self.render_thread.start()
        
        # Main loop
        while dpg.is_dearpygui_running():
            # Process render queue (get latest result only)
            latest_render = None
            try:
                while True:
                    latest_render = self.render_queue.get(block=False)
            except queue.Empty:
                pass
            
            # Process latest render result
            if latest_render is not None:
                try:
                    self.render_buffer = latest_render
                    dpg.set_value("_texture", self.render_buffer)
                except Exception as e:
                    pass
            
            # Handle playback logic - non-blocking
            if self.playing and not self.is_rendering and not self.need_frame_update and not self.need_update:
                time_since_last_frame = time.time() - self.last_frame_time
                
                # Check if should play next frame
                if time_since_last_frame >= 1.0 / self.cfg.fps:
                    self.last_frame_time = time.time()
                    
                    # Calculate next frame
                    next_frame = self.timestep + 1
                    
                    # Handle loop logic
                    if next_frame >= self.num_timesteps:
                        if dpg.get_value("_checkbox_loop"):
                            next_frame = 0
                        else:
                            self.playing = False
                            dpg.set_item_label("_button_play_pause", "Play")
                            continue
                    
                    # Set next frame
                    self.next_frame = next_frame
                    dpg.set_value("_slider_timestep", self.next_frame)
                    dpg.set_value("_log_current_frame", f"{self.next_frame}")
                    
                    self.need_frame_update = True
                    self.need_update = True
            
            # Render UI
            try:
                dpg.render_dearpygui_frame()
            except Exception as e:
                self.is_rendering = False
                self.need_update = False
                self.need_frame_update = False
                time.sleep(0.1)

        # Cleanup
        self.thread_running = False
        if self.render_thread:
            self.render_thread.join(timeout=2.0)

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = LocalViewer(cfg)
    gui.run()
