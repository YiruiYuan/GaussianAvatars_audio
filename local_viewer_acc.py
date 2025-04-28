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
import matplotlib
import threading
import queue
import traceback
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
    fps: int = 30
    """default fps for playback"""
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""
    render_timeout: float = 5.0
    """Maximum time allowed for a single frame render (seconds)"""

class LocalViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # playback settings
        self.playing = False
        self.need_frame_update = False
        self.next_frame = 0
        self.should_skip_complex_frames = True  # 自动跳过复杂帧
        self.max_render_time = 200.0  # 毫秒，大于这个值的帧会被标记为复杂帧
        self.complex_frames = set()   # 记录复杂帧
        
        # 线程同步相关
        self.render_queue = queue.Queue(maxsize=1)
        self.is_rendering = False
        self.render_start_time = 0     # 记录渲染开始时间
        self.thread_running = True
        self.render_thread = None
        self.render_time = 0.0
        self.abort_render = False      # 用户中断标志
        
        # 帧性能统计
        self.frame_times = {}          # 记录每一帧的渲染时间
        
        print("Initializing 3D Gaussians...")
        self.init_gaussians()

        if self.gaussians.binding is not None:
            # rendering settings
            self.mesh_color = torch.tensor([1, 1, 1, 0.5])
            self.face_colors = None
            print("Initializing mesh renderer...")
            self.mesh_renderer = NVDiffRenderer(use_opengl=False)
        
        # FLAME parameters
        if self.gaussians.binding is not None:
            print("Initializing FLAME parameters...")
            self.reset_flame_param()
        
        super().__init__(cfg, 'GaussianAvatars - Local Viewer')

        if self.gaussians.binding is not None:
            self.num_timesteps = self.gaussians.num_timesteps
            dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)
            self.gaussians.select_mesh_by_timestep(self.timestep)
            
        # 设置Ctrl+C处理
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """处理Ctrl+C，安全关闭程序"""
        print("Exiting gracefully...")
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

    def refresh_stat(self):
        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed if elapsed > 0 else 0
            dpg.set_value("_log_fps", f'{int(fps):<4d}')
        self.last_time_fresh = time.time()
    
    def reset_flame_param(self):
        if not hasattr(self.gaussians, 'n_expr'):
            return
        self.flame_param = {
            'expr': torch.zeros(1, self.gaussians.n_expr),
            'rotation': torch.zeros(1, 3),
            'neck': torch.zeros(1, 3),
            'jaw': torch.zeros(1, 3),
            'eyes': torch.zeros(1, 6),
            'translation': torch.zeros(1, 3),
        }

    def define_gui(self):
        super().define_gui()

        # window: rendering options
        with dpg.window(label="Render", tag="_render_window", autosize=True):
            with dpg.group(horizontal=True):
                dpg.add_text("FPS:", show=not self.cfg.demo_mode)
                dpg.add_text("0   ", tag="_log_fps", show=not self.cfg.demo_mode)
                dpg.add_spacer(width=10)
                dpg.add_text("Frame:", show=not self.cfg.demo_mode)
                dpg.add_text("0", tag="_log_current_frame", show=not self.cfg.demo_mode)
                dpg.add_spacer(width=10)
                dpg.add_text("Render:", show=not self.cfg.demo_mode)
                dpg.add_text("0.0ms", tag="_log_render_time", show=not self.cfg.demo_mode)
                dpg.add_spacer(width=10)
                dpg.add_text("Status:", show=not self.cfg.demo_mode)
                dpg.add_text("Ready", tag="_log_status", show=not self.cfg.demo_mode)

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
                    # 非阻塞方式
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
                    
                    # 标记需要更新帧
                    self.need_frame_update = True
                    self.need_update = True
                    dpg.set_value("_log_status", f"Requesting frame {self.next_frame}")

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
                            dpg.set_value("_log_status", "Playing")
                            self.last_frame_time = time.time()
                        else:
                            dpg.set_item_label("_button_play_pause", "Play")
                            dpg.set_value("_log_status", "Paused")
                    dpg.add_button(label="Play", tag="_button_play_pause", callback=callback_play_pause)
                    
                    # 中止渲染按钮
                    def callback_abort_render(sender, app_data):
                        self.abort_render = True
                        dpg.set_value("_log_status", "Aborting render...")
                    dpg.add_button(label="Abort Render", tag="_button_abort_render", callback=callback_abort_render)
                    
                    # Loop checkbox
                    dpg.add_checkbox(label="Loop", default_value=True, tag="_checkbox_loop")
                    
                    # 跳过复杂帧选项
                    def callback_skip_complex(sender, app_data):
                        self.should_skip_complex_frames = app_data
                    dpg.add_checkbox(label="Skip slow frames", default_value=self.should_skip_complex_frames, 
                                    callback=callback_skip_complex, tag="_checkbox_skip_complex")
                
                # Speed control
                with dpg.group(horizontal=True):
                    def callback_set_speed(sender, app_data):
                        self.playback_speed = app_data
                    dpg.add_slider_float(label="Speed", min_value=0.1, max_value=2.0, 
                                        format="%.1fx", default_value=1.0, 
                                        callback=callback_set_speed, tag="_slider_speed", width=200)

            # scaling_modifier slider
            def callback_set_scaling_modifier(sender, app_data):
                self.need_update = True
            dpg.add_slider_float(label="Scale modifier", min_value=0, max_value=1, format="%.2f", 
                                width=200, default_value=1, callback=callback_set_scaling_modifier, 
                                tag="_slider_scaling_modifier")

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True
            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, width=200, 
                              format="%d deg", default_value=self.cam.fovy, 
                              callback=callback_set_fovy, tag="_slider_fovy", 
                              show=not self.cfg.demo_mode)

            # Camera controls
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    dpg.set_value("_slider_fovy", self.cam.fovy)
                    self.need_update = True
                    dpg.set_value("_log_status", "Camera reset")
                dpg.add_button(label="Reset Camera", tag="_button_reset_pose", 
                              callback=callback_reset_camera, show=not self.cfg.demo_mode)
                
                # 保存图像按钮
                def callback_save_image(sender, app_data):
                    if not self.cfg.save_folder.exists():
                        self.cfg.save_folder.mkdir(parents=True)
                    path = self.cfg.save_folder / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
                    print(f"Saving image to {path}")
                    if hasattr(self, 'render_buffer') and self.render_buffer is not None:
                        Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
                        dpg.set_value("_log_status", "Image saved")
                dpg.add_button(label="Save Image", tag="_button_save_image", callback=callback_save_image)

        # widget-dependent handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')
            
            # 添加Escape键处理 - 中止渲染
            def callback_abort_key(sender, app_data):
                self.abort_render = True
                dpg.set_value("_log_status", "Aborting render (ESC)...")
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
        """渲染线程 - 带超时和中断机制"""
        while self.thread_running:
            try:
                # 处理帧更新请求
                if self.need_frame_update and not self.is_rendering:
                    dpg.set_value("_log_status", f"Updating frame to {self.next_frame}")
                    try:
                        self.timestep = self.next_frame
                        self.gaussians.select_mesh_by_timestep(self.timestep)
                        self.need_frame_update = False
                        dpg.set_value("_log_status", f"Updated to frame {self.timestep}")
                        dpg.set_value("_log_current_frame", f"{self.timestep}")
                    except Exception as e:
                        print(f"Error updating frame: {e}")
                        dpg.set_value("_log_status", f"Error: {str(e)[:20]}...")
                        self.need_frame_update = False  # 确保错误状态下也清除标志
                
                # 执行渲染
                if self.need_update and not self.is_rendering:
                    # 检查是否是已知的复杂帧
                    if self.should_skip_complex_frames and self.playing and self.timestep in self.complex_frames:
                        dpg.set_value("_log_status", f"Skipping slow frame {self.timestep}")
                        self.need_update = False
                        continue
                    
                    # 开始渲染
                    self.is_rendering = True
                    self.abort_render = False
                    
                    # 这里可能会阻塞UI线程
                    try:
                        dpg.set_value("_log_status", f"Rendering frame {self.timestep}...")
                    except:
                        print("Warning: Failed to update UI status")
                    
                    # 记录渲染开始时间
                    self.render_start_time = time.time()
                    render_start_time = self.render_start_time
                    
                    try:
                        # 获取渲染参数
                        show_splatting = dpg.get_value("_checkbox_show_splatting")
                        show_mesh = dpg.get_value("_checkbox_show_mesh") if self.gaussians.binding is not None else False
                        scaling_modifier = dpg.get_value("_slider_scaling_modifier")
                        
                        # 准备摄像机
                        cam = self.prepare_camera()
                        
                        # 初始化为默认背景
                        rgb = torch.ones([self.H, self.W, 3])
                        
                        # 执行渲染，并检查超时和中断
                        timeout_occurred = False
                        
                        # 渲染高斯点云
                        if show_splatting and not self.abort_render:
                            dpg.set_value("_log_status", "Rendering gaussians...")
                            
                            # 检查渲染是否超时
                            if time.time() - render_start_time > self.cfg.render_timeout:
                                dpg.set_value("_log_status", f"Timeout rendering gaussians on frame {self.timestep}")
                                timeout_occurred = True
                            else:
                                rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, 
                                                    torch.tensor(self.cfg.background_color).cuda(), 
                                                    scaling_modifier=scaling_modifier)["render"].permute(1, 2, 0).contiguous()
                        
                        # 渲染网格
                        if show_mesh and not self.abort_render and not timeout_occurred and self.gaussians.binding is not None:
                            dpg.set_value("_log_status", "Rendering mesh...")
                            
                            # 检查渲染是否超时
                            if time.time() - render_start_time > self.cfg.render_timeout:
                                dpg.set_value("_log_status", f"Timeout rendering mesh on frame {self.timestep}")
                                timeout_occurred = True
                            else:
                                out_dict = self.mesh_renderer.render_from_camera(
                                    self.gaussians.verts, self.gaussians.faces, cam, face_colors=self.face_colors)
                                
                                rgba_mesh = out_dict['rgba'].squeeze(0)
                                rgb_mesh = rgba_mesh[:, :, :3]
                                alpha_mesh = rgba_mesh[:, :, 3:]
                                mesh_opacity = self.mesh_color[3:].cuda()
                        
                        # 合成最终图像
                        if not timeout_occurred and not self.abort_render:
                            if show_splatting and show_mesh:
                                rgb = rgb_mesh * alpha_mesh * mesh_opacity + rgb_splatting * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                            elif show_splatting:
                                rgb = rgb_splatting
                            elif show_mesh:
                                rgb = rgb_mesh
                        
                        # 计算渲染时间
                        render_time = time.time() - render_start_time
                        
                        # 记录帧渲染性能
                        self.frame_times[self.timestep] = render_time * 1000  # 毫秒
                        
                        # 标记复杂帧
                        if render_time * 1000 > self.max_render_time:
                            self.complex_frames.add(self.timestep)
                            dpg.set_value("_log_status", f"Frame {self.timestep} is slow: {render_time*1000:.1f}ms")
                        
                        # 避免在渲染线程中过多地更新UI状态
                        status = ""
                        if not timeout_occurred and not self.abort_render:
                            render_buffer = rgb.cpu().numpy()
                            if render_buffer.shape[0] == self.H and render_buffer.shape[1] == self.W:
                                # 不要在这里清空队列，移到主线程中处理
                                try:
                                    # 使用非阻塞方式放入队列
                                    self.render_queue.put((render_buffer, render_time, self.timestep), 
                                                        block=False)
                                    status = "Render complete"
                                except queue.Full:
                                    status = "Render queue full, skipping update"
                            else:
                                status = f"Invalid render size: {render_buffer.shape}"
                        elif timeout_occurred:
                            status = f"Render timeout on frame {self.timestep}"
                        else:
                            status = f"Render aborted on frame {self.timestep}"
                        
                        try:
                            dpg.set_value("_log_status", status)
                        except:
                            print(f"Warning: Failed to update UI status: {status}")
                        
                    except Exception as e:
                        print(f"Render task error: {traceback.format_exc()}")
                        try:
                            dpg.set_value("_log_status", f"Render error: {str(e)[:20]}...")
                        except:
                            print(f"Warning: Failed to update UI with error: {str(e)[:20]}")
                    
                    # 确保无论如何都重置状态
                    self.need_update = False
                    self.is_rendering = False
                    self.abort_render = False
                else:
                    # 没有渲染任务，短暂休眠
                    time.sleep(0.01)
            
            except Exception as e:
                print(f"Render thread error: {traceback.format_exc()}")
                try:
                    dpg.set_value("_log_status", f"Thread error: {str(e)[:20]}...")
                except:
                    print(f"Warning: Failed to update UI with thread error")
                self.is_rendering = False
                self.need_update = False
                time.sleep(0.1)
                
            # 检查渲染是否运行超时
            if self.is_rendering and (time.time() - self.render_start_time > self.cfg.render_timeout + 1.0):
                print(f"Render timeout detected for frame {self.timestep}")
                try:
                    dpg.set_value("_log_status", f"Timeout forced abort for frame {self.timestep}")
                except:
                    print("Warning: Failed to update UI status for timeout")
                self.is_rendering = False
                self.need_update = False
                time.sleep(0.1)

    @torch.no_grad()
    def run(self):
        print("Running LocalViewer...")
        
        # 初始化播放相关变量
        self.playback_speed = 1.0
        self.last_frame_time = time.time()
        self.ui_update_time = time.time()  # 追踪UI更新时间
        
        # 性能分析变量
        timing_stats = {
            "ui_render": 0.0,
            "queue_process": 0.0,
            "playback_logic": 0.0,
            "total_frame": 0.0,
            "idle": 0.0
        }
        timing_counts = {k: 0 for k in timing_stats}
        last_stats_time = time.time()
        
        try:
            dpg.set_value("_log_current_frame", f"{self.timestep}")
        except:
            print("Warning: Failed to initialize UI frame number")

        # 启动渲染线程
        self.render_thread = threading.Thread(target=self.render_frame)
        self.render_thread.daemon = True
        self.render_thread.start()
        
        # 主循环
        ui_freeze_count = 0  # 用于监测UI冻结
        last_ui_time = time.time()
        
        while dpg.is_dearpygui_running():
            frame_start = time.time()
            current_time = frame_start
            
            # 监测UI响应
            if current_time - last_ui_time > 1.0:
                ui_freeze_count += 1
                print(f"Possible UI freeze detected: {ui_freeze_count}")
                if ui_freeze_count > 5:
                    print("UI appears frozen, forcing state reset")
                    self.is_rendering = False
                    self.need_update = False
                    self.need_frame_update = False
                    self.abort_render = True
                    ui_freeze_count = 0
            else:
                ui_freeze_count = 0
            last_ui_time = current_time
            
            # 清空队列前的元素，只保留最新的一个
            queue_start = time.time()
            latest_render = None
            try:
                while True:
                    render_data = self.render_queue.get(block=False)
                    latest_render = render_data  # 保存最新的渲染结果
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error clearing queue: {e}")
            
            # 处理最新渲染结果
            if latest_render is not None:
                try:
                    if isinstance(latest_render, tuple) and len(latest_render) >= 2:
                        render_buffer, render_time = latest_render[0], latest_render[1]
                        self.render_buffer = render_buffer
                        self.render_time = render_time
                        
                        # 更新UI
                        dpg.set_value("_texture", self.render_buffer)
                        dpg.set_value("_log_render_time", f'{render_time*1000:.1f}ms')
                        self.refresh_stat()
                        self.ui_update_time = current_time
                except Exception as e:
                    print(f"UI update error: {e}")
            
            queue_end = time.time()
            timing_stats["queue_process"] += (queue_end - queue_start)
            timing_counts["queue_process"] += 1
            
            # 处理播放逻辑 - 非阻塞方式
            playback_start = time.time()
            if self.playing and not self.is_rendering and not self.need_frame_update and not self.need_update:
                speed = getattr(self, 'playback_speed', 1.0)
                fps_target = self.cfg.fps * speed
                time_since_last_frame = current_time - self.last_frame_time
                
                # 检查是否应该播放下一帧
                if time_since_last_frame >= 1.0 / fps_target:
                    self.last_frame_time = current_time
                    
                    # 计算下一帧
                    next_frame = self.timestep + 1
                    
                    # 如果启用跳帧，尝试找到下一个非复杂帧
                    if self.should_skip_complex_frames:
                        skip_count = 0
                        while next_frame in self.complex_frames and skip_count < 10 and next_frame < self.num_timesteps - 1:
                            next_frame += 1
                            skip_count += 1
                        if skip_count > 0:
                            try:
                                dpg.set_value("_log_status", f"Skipped {skip_count} slow frames")
                            except:
                                pass
                    
                    # 处理循环逻辑
                    if next_frame >= self.num_timesteps:
                        if dpg.get_value("_checkbox_loop"):
                            next_frame = 0
                        else:
                            self.playing = False
                            try:
                                dpg.set_item_label("_button_play_pause", "Play")
                                dpg.set_value("_log_status", "Playback complete")
                            except:
                                print("Warning: Failed to update playback controls")
                            continue
                    
                    # 设置下一帧
                    self.next_frame = next_frame
                    try:
                        dpg.set_value("_slider_timestep", self.next_frame)
                        dpg.set_value("_log_current_frame", f"{self.next_frame}")
                    except:
                        print(f"Warning: Failed to update UI with frame {self.next_frame}")
                    
                    self.need_frame_update = True
                    self.need_update = True
                    
                    try:
                        if next_frame in self.complex_frames:
                            dpg.set_value("_log_status", f"Playing slow frame {self.next_frame}")
                        else:
                            dpg.set_value("_log_status", f"Playing frame {self.next_frame}")
                    except:
                        print("Warning: Failed to update status message")
            
            playback_end = time.time()
            timing_stats["playback_logic"] += (playback_end - playback_start)
            timing_counts["playback_logic"] += 1
            
            # 检测UI线程和渲染线程之间可能的死锁
            if self.is_rendering and (current_time - self.render_start_time > 10.0):
                print(f"Potential deadlock detected, forcing state reset")
                self.is_rendering = False
                self.need_update = False
                self.need_frame_update = False
                self.abort_render = True
            
            # 渲染UI
            ui_start = time.time()
            try:
                dpg.render_dearpygui_frame()
            except Exception as e:
                print(f"Error rendering UI frame: {e}")
                self.is_rendering = False
                self.need_update = False
                self.need_frame_update = False
                time.sleep(0.1)
            
            ui_end = time.time()
            timing_stats["ui_render"] += (ui_end - ui_start)
            timing_counts["ui_render"] += 1
            
            # 计算帧总时间和空闲时间
            idle_start = time.time()
            time.sleep(0.005)  # 略微增加延迟，减轻CPU负担
            idle_end = time.time()
            
            timing_stats["idle"] += (idle_end - idle_start)
            timing_counts["idle"] += 1
            
            frame_end = time.time()
            timing_stats["total_frame"] += (frame_end - frame_start)
            timing_counts["total_frame"] += 1
            
            # 每5秒打印一次性能统计
            if time.time() - last_stats_time > 5.0 and all(v > 0 for v in timing_counts.values()):
                avg_stats = {k: (timing_stats[k] * 1000 / timing_counts[k]) for k in timing_stats}
                total_time_per_frame = avg_stats["total_frame"]
                effective_fps = 1000 / total_time_per_frame if total_time_per_frame > 0 else 0
                
                print("\n--- Performance Statistics (ms) ---")
                print(f"Queue processing:  {avg_stats['queue_process']:.2f} ms")
                print(f"Playback logic:    {avg_stats['playback_logic']:.2f} ms")
                print(f"UI rendering:      {avg_stats['ui_render']:.2f} ms")
                print(f"Idle time:         {avg_stats['idle']:.2f} ms")
                print(f"Total frame time:  {avg_stats['total_frame']:.2f} ms")
                print(f"Effective FPS:     {effective_fps:.2f}")
                print(f"Render thread FPS: {1000 / (self.render_time*1000) if self.render_time else 0:.2f}")
                
                # 如果需要，打印当前渲染线程状态
                if self.is_rendering:
                    render_duration = time.time() - self.render_start_time
                    print(f"Currently rendering frame {self.timestep} for {render_duration:.2f}s")
                
                # 重置计数器
                timing_stats = {k: 0.0 for k in timing_stats}
                timing_counts = {k: 0 for k in timing_counts}
                last_stats_time = time.time()

        # 清理线程
        print("Shutting down render thread...")
        self.thread_running = False
        if self.render_thread:
            timeout = 2.0
            self.render_thread.join(timeout=timeout)
            if self.render_thread.is_alive():
                print("Warning: Render thread did not terminate cleanly")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = LocalViewer(cfg)
    gui.run()
