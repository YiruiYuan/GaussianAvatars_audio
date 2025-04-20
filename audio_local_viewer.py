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
from scipy.interpolate import interp1d
import matplotlib
import soundfile as sf
import sounddevice as sd
import threading
import subprocess
import shutil
import os
import requests
import sys

# 添加中文字体支持
is_windows = sys.platform.startswith('win')

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
    audio_path: Optional[Path] = None
    """Path to the audio file"""
    server_audio_path: Optional[str] = None
    """服务器上音频文件的绝对路径，用于API请求"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    background_color: tuple[float, float, float] = (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""
    fps: int = 25
    """default fps for recording"""
    keyframe_interval: int = 1
    """default keyframe interval"""
    ref_json: Optional[Path] = None
    """ Path to a reference json file. We copy file paths from a reference json into 
    the exported trajectory json file as placeholders so that `render.py` can directly
    load it like a normal sequence. """
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""
    lock_frame_rate: bool = True
    """Lock frame rate to match audio"""
    audio_offset: float = 0.0
    """Audio offset in seconds"""
    api_url: str = "http://localhost:5001"
    """API服务器URL"""
    test_api: bool = False
    """启动时测试API连接"""

class LocalViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # 设置中文字体支持
        dpg.create_context()
        if is_windows:
            with dpg.font_registry():
                with dpg.font("c:/windows/fonts/simhei.ttf", 18) as default_font:
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                    dpg.bind_font(default_font)
        
        # recording settings
        self.keyframes = []  # list of state dicts of keyframes
        self.all_frames = {}  # state dicts of all frames {key: [num_frames, ...]}
        self.num_record_timeline = 0
        self.playing = False
        
        # 自动模式 - 简化用户界面
        self.auto_mode = True
        
        # audio settings
        self.audio_data = None
        self.sample_rate = None
        self.audio_playing = False
        self.audio_stream = None
        self.audio_thread = None
        self.audio_time = 0
        self.audio_frame_index = 0
        self.audio_frame_duration = 1.0 / self.cfg.fps if self.cfg.fps > 0 else 0.04  # default 25fps
        
        # 加载音频文件
        self.load_audio()
        
        # 测试API连接
        if self.cfg.test_api:
            self.test_api_connection()

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
        
        super().__init__(cfg, 'GaussianAvatars - Audio Local Viewer')

        if self.gaussians.binding is not None:
            self.num_timesteps = self.gaussians.num_timesteps
            dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)

            self.gaussians.select_mesh_by_timestep(self.timestep)
            
        # 如果处于自动模式且加载了音频，自动设置关键帧
        if self.auto_mode and self.audio_data is not None and self.gaussians.binding is not None:
            # 延迟执行，确保GUI已经完全初始化
            def auto_setup():
                try:
                    time.sleep(0.5)  # 等待GUI完全初始化
                    
                    # 检查GUI是否已经初始化
                    if not dpg.does_item_exist("_slider_timestep") or not dpg.does_item_exist("_slider_record_timestep"):
                        print("等待GUI完全初始化...")
                        time.sleep(1)  # 再等待一秒
                    
                    # 再次检查GUI元素
                    if not dpg.does_item_exist("_slider_timestep"):
                        print("错误: GUI元素'_slider_timestep'未找到，无法进行自动设置")
                        return

                    # 自动调用API获取FLAME参数（如果有音频文件）
                    # 优先使用server_audio_path参数
                    audio_path = None
                    if self.cfg.server_audio_path:
                        audio_path = self.cfg.server_audio_path
                        print(f"自动模式：使用服务器上的音频路径: {audio_path}")
                    elif self.cfg.audio_path is not None:
                        audio_path = str(self.cfg.audio_path)
                        print(f"自动模式：使用本地音频路径: {audio_path}")
                    
                    if audio_path:
                        print("自动模式：尝试从API获取FLAME参数...")
                        # 使用默认参数调用API
                        success = self.get_flame_params_from_api(
                            audio_path,
                            subject_style="M003",
                            emotion="neutral", 
                            intensity=2
                        )
                        if success:
                            print("自动模式：成功从API获取FLAME参数")
                            return
                        else:
                            print("自动模式：无法从API获取FLAME参数，将使用默认关键帧设置")
                    
                    # 设置首帧关键帧
                    self.timestep = 0
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)
                    first_state = self.get_state_dict()
                    self.keyframes.append(first_state)
                    
                    # 设置末帧关键帧
                    if self.num_timesteps > 1:
                        # 保存当前帧
                        old_timestep = self.timestep
                        
                        # 切换到最后一帧
                        self.timestep = self.num_timesteps - 1
                        dpg.set_value("_slider_timestep", self.timestep)
                        self.gaussians.select_mesh_by_timestep(self.timestep)
                        
                        # 添加末帧关键帧
                        last_state = self.get_state_dict()
                        self.keyframes.append(last_state)
                        
                        # 恢复原始帧
                        self.timestep = old_timestep
                        dpg.set_value("_slider_timestep", self.timestep)
                        self.gaussians.select_mesh_by_timestep(self.timestep)
                    
                    # 设置循环模式
                    dpg.set_value("_input_cycles", 1)
                    
                    # 更新时间线
                    self.update_record_timeline()
                    
                    # 检查是否成功创建时间线
                    if self.num_record_timeline <= 0:
                        print("警告: 无法创建有效的播放时间线")
                        return
                    
                    # 重置到起点
                    dpg.set_value("_slider_record_timestep", 0)
                    
                    # 开始播放
                    self.playing = True
                    self.need_update = True
                    
                    # 启动音频
                    self.start_audio()
                    
                    print("自动设置完成: 已创建关键帧并开始播放")
                except Exception as e:
                    print(f"自动设置过程中出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 创建线程执行自动设置
            auto_thread = threading.Thread(target=auto_setup)
            auto_thread.daemon = True
            auto_thread.start()
        
    def load_audio(self):
        """加载音频文件"""
        if self.cfg.audio_path is not None and self.cfg.audio_path.exists():
            try:
                print(f"Loading audio file: {self.cfg.audio_path}")
                self.audio_data, self.sample_rate = sf.read(self.cfg.audio_path)
                # 如果是单声道，转换为立体声
                if len(self.audio_data.shape) == 1:
                    self.audio_data = np.stack([self.audio_data, self.audio_data], axis=1)
                print(f"Audio loaded: {self.audio_data.shape}, sr={self.sample_rate}Hz")
            except Exception as e:
                print(f"Error loading audio file: {e}")
                self.audio_data = None
                self.sample_rate = None
        else:
            print("No audio file specified or file does not exist.")
            
    def audio_callback(self, outdata, frames, time, status):
        """音频播放回调函数"""
        if status:
            print(f"Audio status: {status}")
        
        if self.audio_data is None or not self.audio_playing:
            outdata.fill(0)
            return
            
        # 获取当前记录时间线位置
        try:
            current_record_timestep = dpg.get_value("_slider_record_timestep")
            if current_record_timestep != self.audio_frame_index:
                # 视觉帧和音频帧不同步，重新同步
                self.audio_frame_index = current_record_timestep
        except:
            pass
            
        # 计算当前帧位置
        current_sample = int(self.audio_frame_index * self.audio_frame_duration * self.sample_rate)
        
        # 应用音频偏移
        current_sample += int(self.cfg.audio_offset * self.sample_rate)
        
        # 如果偏移使得起始位置为负，填充0
        if current_sample < 0:
            zero_frames = min(frames, abs(current_sample))
            audio_frames = frames - zero_frames
            outdata[:zero_frames].fill(0)
            
            if audio_frames > 0 and abs(current_sample) < len(self.audio_data):
                outdata[zero_frames:] = self.audio_data[:audio_frames]
            else:
                outdata[zero_frames:].fill(0)
                
            current_sample = audio_frames
        else:
            end_sample = current_sample + frames
            
            # 检查是否播放到文件末尾
            if current_sample >= len(self.audio_data):
                outdata.fill(0)
                # 不要自动停止音频，让视觉控制循环
                return
                
            # 计算需要多少音频帧
            if end_sample > len(self.audio_data):
                # 如果请求的帧超出了音频长度，填充剩余部分为0
                audio_part = self.audio_data[current_sample:len(self.audio_data)]
                zeros_part = np.zeros((frames - len(audio_part), self.audio_data.shape[1]), dtype=self.audio_data.dtype)
                outdata[:len(audio_part)] = audio_part
                outdata[len(audio_part):] = zeros_part
            else:
                # 正常情况
                outdata[:] = self.audio_data[current_sample:end_sample]
                
        # 更新音频帧索引
        self.audio_frame_index += frames / (self.audio_frame_duration * self.sample_rate)
    
    def start_audio(self):
        """开始播放音频"""
        if self.audio_data is None or self.audio_playing:
            return
            
        def audio_player():
            try:
                # 从当前视觉帧的对应位置开始播放音频
                self.audio_frame_index = dpg.get_value("_slider_record_timestep")
                
                # 计算开始时间(秒)
                start_time = self.audio_frame_index * self.audio_frame_duration
                
                # 使用 blocksize 参数控制音频缓冲区大小，可以提高同步精度
                # 设置一个相对较小的缓冲区大小，以减少延迟
                blocksize = 1024  # 可以根据需要调整
                
                # 创建音频流
                with sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=self.audio_data.shape[1],
                    callback=self.audio_callback,
                    blocksize=blocksize
                ) as stream:
                    self.audio_stream = stream
                    self.audio_playing = True
                    
                    # 等待直到音频播放停止
                    while self.audio_playing and dpg.is_dearpygui_running():
                        sd.sleep(100)
                        
            except Exception as e:
                print(f"Audio playback error: {e}")
            finally:
                self.audio_playing = False
                self.audio_stream = None
        
        # 终止任何现有的音频线程
        if self.audio_thread is not None and self.audio_thread.is_alive():
            self.stop_audio()
            time.sleep(0.1)  # 等待线程完全终止
        
        self.audio_thread = threading.Thread(target=audio_player)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
    def stop_audio(self):
        """停止音频播放"""
        self.audio_playing = False
        if self.audio_stream:
            self.audio_stream.abort()
            self.audio_stream = None

    def init_gaussians(self):
        # load gaussians
        if (Path(self.cfg.point_path).parent / "flame_param.npz").exists():
            self.gaussians = FlameGaussianModel(self.cfg.sh_degree)
        else:
            self.gaussians = GaussianModel(self.cfg.sh_degree)

        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['left_half'])
        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['right_half'])
        # unselected_fid = self.gaussians.flame_model.mask.get_fid_except_fids(selected_fid)
        unselected_fid = []
        
        if self.cfg.point_path is not None:
            if self.cfg.point_path.exists():
                self.gaussians.load_ply(self.cfg.point_path, has_target=False, motion_path=self.cfg.motion_path, disable_fid=unselected_fid)
            else:
                raise FileNotFoundError(f'{self.cfg.point_path} does not exist.')

    def refresh_stat(self):
        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{int(fps):<4d}')
        self.last_time_fresh = time.time()
    
    def update_record_timeline(self):
        cycles = dpg.get_value("_input_cycles")
        
        # 检查关键帧数量，避免对空列表或单个关键帧的错误处理
        if len(self.keyframes) == 0:
            self.num_record_timeline = 0
            self.all_frames = {}
            # 确保时间线滑块被配置为有效范围（即使是0）
            dpg.configure_item("_slider_record_timestep", min_value=0, max_value=0)
            return
        elif len(self.keyframes) == 1:
            # 只有一个关键帧时，使用其间隔作为时间线长度
            self.num_record_timeline = self.keyframes[0]['interval']
        else:
            # 多个关键帧时的正常处理
            if cycles == 0:
                self.num_record_timeline = sum([keyframe['interval'] for keyframe in self.keyframes[:-1]])
            else:
                self.num_record_timeline = sum([keyframe['interval'] for keyframe in self.keyframes]) * cycles

        # 确保至少有1帧的时间线长度
        self.num_record_timeline = max(1, self.num_record_timeline)
        dpg.configure_item("_slider_record_timestep", min_value=0, max_value=self.num_record_timeline-1)

        if len(self.keyframes) <= 0:
            self.all_frames = {}
            return
        else:
            k_x = []

            keyframes = self.keyframes.copy()
            if cycles > 0:
                # pad a cycle at the beginning and the end to ensure smooth transition
                keyframes = self.keyframes * (cycles + 2)
                t_couter = -sum([keyframe['interval'] for keyframe in self.keyframes])
            else:
                t_couter = 0

            for keyframe in keyframes:
                k_x.append(t_couter)
                t_couter += keyframe['interval']
            
            x = np.arange(self.num_record_timeline)
            self.all_frames = {}

            if len(keyframes) <= 1:
                for k in keyframes[0]:
                    k_y = np.concatenate([np.array(keyframe[k])[None] for keyframe in keyframes], axis=0)
                    self.all_frames[k] = np.tile(k_y, (self.num_record_timeline, 1))
            else:
                kind = 'linear' if len(keyframes) <= 3 else 'cubic'
            
                for k in keyframes[0]:
                    if k == 'interval':
                        continue
                    k_y = np.concatenate([np.array(keyframe[k])[None] for keyframe in keyframes], axis=0)
                  
                    interp_funcs = [interp1d(k_x, k_y[:, i], kind=kind, fill_value='extrapolate') for i in range(k_y.shape[1])]

                    y = np.array([interp_func(x) for interp_func in interp_funcs]).transpose(1, 0)
                    self.all_frames[k] = y

    def get_state_dict(self):
        return {
            'rot': self.cam.rot.as_quat(),
            'look_at': np.array(self.cam.look_at),
            'radius': np.array([self.cam.radius]).astype(np.float32),
            'fovy': np.array([self.cam.fovy]).astype(np.float32),
            'interval': self.cfg.fps*self.cfg.keyframe_interval,
        }

    def get_state_dict_record(self):
        # 安全地处理self.all_frames为空的情况
        if not self.all_frames:
            # 如果没有帧数据，返回当前状态
            return self.get_state_dict()
            
        record_timestep = dpg.get_value("_slider_record_timestep")
        # 确保record_timestep在有效范围内
        if self.num_record_timeline <= 0:
            return self.get_state_dict()
            
        record_timestep = min(record_timestep, self.num_record_timeline-1)
        record_timestep = max(0, record_timestep)
        state_dict = {k: self.all_frames[k][record_timestep] for k in self.all_frames}
        return state_dict

    def apply_state_dict(self, state_dict):
        if 'rot' in state_dict:
            self.cam.rot = R.from_quat(state_dict['rot'])
        if 'look_at' in state_dict:
            self.cam.look_at = state_dict['look_at']
        if 'radius' in state_dict:
            self.cam.radius = state_dict['radius'].item()
        if 'fovy' in state_dict:
            self.cam.fovy = state_dict['fovy'].item()
    
    def parse_ref_json(self):
        if self.cfg.ref_json is None:
            return {}
        else:
            with open(self.cfg.ref_json, 'r') as f:
                ref_dict = json.load(f)

        tid2paths = {}
        for frame in ref_dict['frames']:
            tid = frame['timestep_index']
            if tid not in tid2paths:
                tid2paths[tid] = frame
        return tid2paths
    
    def export_trajectory(self):
        tid2paths = self.parse_ref_json()

        if self.num_record_timeline <= 0:
            return
        
        timestamp = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        traj_dict = {'frames': []}
        timestep_indices = []
        camera_indices = []
        
        # 创建保存目录
        save_folder = self.cfg.save_folder / timestamp
        if not save_folder.exists():
            save_folder.mkdir(parents=True)
            
        # 如果有音频，保存音频参考信息（但不导出音频文件，因为这是实时播放器）
        if self.audio_data is not None and self.cfg.audio_path is not None:
            audio_info = {
                'path': str(self.cfg.audio_path),
                'fps': self.cfg.fps,
                'offset': self.cfg.audio_offset,
                'sample_rate': self.sample_rate
            }
            traj_dict['audio'] = audio_info
        
        for i in range(self.num_record_timeline):
            # update
            dpg.set_value("_slider_record_timestep", i)
            state_dict = self.get_state_dict_record()
            self.apply_state_dict(state_dict)

            self.need_update = True
            while self.need_update:
                time.sleep(0.001)

            # save image
            path = save_folder / f"{i:05d}.png"
            print(f"Saving image to {path}")
            Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)

            # cache camera parameters
            cx = self.cam.intrinsics[2]
            cy = self.cam.intrinsics[3]
            fl_x = self.cam.intrinsics[0].item() if isinstance(self.cam.intrinsics[0], np.ndarray) else self.cam.intrinsics[0]
            fl_y = self.cam.intrinsics[1].item() if isinstance(self.cam.intrinsics[1], np.ndarray) else self.cam.intrinsics[1]
            h = self.cam.image_height
            w = self.cam.image_width
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2

            c2w = self.cam.pose.copy()  # opencv convention
            c2w[:, [1, 2]] *= -1  # opencv to opengl
            
            timestep_index = self.timestep
            camera_indx = i
            timestep_indices.append(timestep_index)
            camera_indices.append(camera_indx)

            frame = {
                "cx": cx,
                "cy": cy,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "h": h,
                "w": w,
                "camera_angle_x": angle_x,
                "camera_angle_y": angle_y,
                "transform_matrix": c2w.tolist(),
                'timestep_index': timestep_index,
                'camera_indx': camera_indx,
            }
            if timestep_index in tid2paths:
                frame['file_path'] = tid2paths[timestep_index]['file_path']
                frame['fg_mask_path'] = tid2paths[timestep_index]['fg_mask_path']
                frame['flame_param_path'] = tid2paths[timestep_index]['flame_param_path']
            traj_dict['frames'].append(frame)

            # update timestep
            if dpg.get_value("_checkbox_dynamic_record"):
                self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                dpg.set_value("_slider_timestep", self.timestep)
                self.gaussians.select_mesh_by_timestep(self.timestep)
        
        traj_dict['timestep_indices'] = sorted(list(set(timestep_indices)))
        traj_dict['camera_indices'] = sorted(list(set(camera_indices)))
        
        # save trajectory parameters
        path = save_folder / f"trajectory.json"
        print(f"Saving trajectory to {path}")
        with open(path, 'w') as f:
            json.dump(traj_dict, f, indent=4)
            
        print(f"Successfully exported trajectory to {save_folder}")
        return save_folder

    def reset_flame_param(self):
        self.flame_param = {
            'expr': torch.zeros(1, self.gaussians.n_expr),
            'rotation': torch.zeros(1, 3),
            'neck': torch.zeros(1, 3),
            'jaw': torch.zeros(1, 3),
            'eyes': torch.zeros(1, 6),
            'translation': torch.zeros(1, 3),
        }
        
    def get_flame_params_from_api(self, audio_path, subject_style="M003", emotion="neutral", intensity=2):
        """从API获取FLAME参数并嵌入到现有的flame_param.npz中
        
        Args:
            audio_path (str): 音频文件路径
            subject_style (str, optional): 角色风格ID. 默认为 "M003".
            emotion (str, optional): 情绪类型. 默认为 "neutral".
            intensity (int, optional): 情绪强度. 默认为 2.
            
        Returns:
            bool: 是否成功获取并应用参数
        """
        if not os.path.exists(audio_path):
            print(f"警告: 本地音频文件不存在: {audio_path}")
            print("注意: API需要访问服务器上的音频文件路径，而不是本地路径")
            print("请确保提供的路径是服务器上的绝对路径，例如: /home/plm/inferno/assets/data/EMOTE_test_example_data/02_that.wav")
            # 不立即返回，因为API服务器可能有权限访问其他位置的文件
            
        try:
            # 准备请求数据
            api_url = f"{self.cfg.api_url}/api/flame_from_path"
            
            # 确保音频路径是绝对路径
            if not os.path.isabs(audio_path):
                print(f"警告: 提供的音频路径不是绝对路径: {audio_path}")
                print("正在尝试转换为绝对路径...")
                # 尝试转换为绝对路径
                abs_audio_path = os.path.abspath(audio_path)
                print(f"转换后的绝对路径: {abs_audio_path}")
                audio_path = abs_audio_path
            
            # 打印示例音频路径
            print("提示: 示例服务器音频路径格式: /home/plm/inferno/assets/data/EMOTE_test_example_data/02_that.wav")
            
            payload = {
                "audio_path": audio_path,
                "subject_style": subject_style,
                "emotion": emotion,
                "intensity": intensity
            }
            
            # 发送请求
            print(f"正在请求API: {api_url}")
            print(f"请求参数: {payload}")
            
            # 检查API服务是否正在运行
            try:
                health_check = requests.get(f"{self.cfg.api_url}/health", timeout=2)
                if health_check.status_code == 200:
                    print(f"API服务状态检查成功: {health_check.text}")
                else:
                    print(f"API服务状态异常: {health_check.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"无法连接到API服务，请确保服务正在运行: {e}")
                print(f"建议通过终端检查API服务状态或重新启动API服务")
                return False
            
            # 执行正式请求
            try:
                response = requests.post(api_url, json=payload, timeout=30)
            except requests.exceptions.Timeout:
                print(f"API请求超时 (30秒)，服务可能处理较慢")
                return False
            except requests.exceptions.RequestException as e:
                print(f"API请求异常: {e}")
                return False
            
            if response.status_code != 200:
                print(f"API请求失败: 状态码 {response.status_code}")
                print(f"错误信息: {response.text}")
                return False
                
            # 解析响应
            try:
                flame_data = response.json()
                print(f"成功获取FLAME参数，帧数: {flame_data['metadata']['num_frames']}")
            except ValueError as e:
                print(f"解析API响应失败，可能不是有效的JSON格式: {e}")
                print(f"响应内容: {response.text[:500]}...")  # 打印部分响应内容
                return False
            except KeyError as e:
                print(f"API响应格式不符合预期，缺少必要字段: {e}")
                print(f"响应内容中的键: {flame_data.keys()}")
                return False
            
            # 检查必要参数是否存在
            if 'expression' not in flame_data or 'jaw_pose' not in flame_data:
                print("API返回数据缺少必要的'expression'或'jaw_pose'参数")
                print(f"可用参数: {list(flame_data.keys())}")
                return False
                
            # 转换参数为适当的张量
            expression = np.array(flame_data['expression'])  # 预期形状：[num_frames, n_expr]
            jaw_pose = np.array(flame_data['jaw_pose'])      # 预期形状：[num_frames, 3]
            
            # 检查并打印参数形状
            print(f"API返回的表情参数形状: {expression.shape}")
            print(f"API返回的下颚姿态参数形状: {jaw_pose.shape}")
            
            # 获取现有FLAME参数的副本
            flame_param_copy = {}
            print("获取当前FLAME参数副本...")
            for k, v in self.gaussians.flame_param.items():
                flame_param_copy[k] = v.clone().cpu().numpy()
                print(f"  - 参数 {k}: 形状 {flame_param_copy[k].shape}")
            
            # 记录原始帧数
            original_frames = flame_param_copy['expr'].shape[0]
            api_frames = flame_data['metadata']['num_frames']
            
            print(f"原始NPZ帧数: {original_frames}, API返回帧数: {api_frames}")
            
            # 使用API返回的帧数，不再填充API数据到原始帧数
            target_frames = api_frames
            
            # 调整API数据帧数 - 不再需要
            # if api_frames < target_frames:
            #     # 填充API数据
            #     pad_frames = target_frames - api_frames
            #     expression = np.pad(expression, ((0, pad_frames), (0, 0)), mode='edge')
            #     jaw_pose = np.pad(jaw_pose, ((0, pad_frames), (0, 0)), mode='edge')
            #     print(f"已填充API数据到 {target_frames} 帧")
            # elif api_frames > target_frames:
            #     # 截断API数据
            #     expression = expression[:target_frames]
            #     jaw_pose = jaw_pose[:target_frames]
            #     print(f"已截断API数据到 {target_frames} 帧")
            
            # 不再填充原始NPZ参数，直接使用API返回的帧数
            print(f"使用API返回的帧数: {target_frames} 帧")
            
            # 调整表情参数维度
            target_expr_dim = flame_param_copy['expr'].shape[1]
            current_expr_dim = expression.shape[1]
            
            if current_expr_dim != target_expr_dim:
                print(f"表情参数维度不匹配: 原始={target_expr_dim}, API={current_expr_dim}")
                
                if current_expr_dim < target_expr_dim:
                    # 填充API表情数据
                    expression = np.pad(expression, ((0, 0), (0, target_expr_dim - current_expr_dim)), mode='constant')
                    print(f"已填充API表情参数维度到 {target_expr_dim}")
                else:
                    # 截断API表情数据
                    expression = expression[:, :target_expr_dim]
                    print(f"已截断API表情参数维度到 {target_expr_dim}")
            
            # 将API数据替换到flame_param_copy中
            print("替换FLAME参数...")
            
            # 创建最终参数字典
            final_flame_param = {}
            
            # 先添加API返回的表情和下颚参数
            final_flame_param['expr'] = expression
            final_flame_param['jaw_pose'] = jaw_pose
            print(f"已应用API返回的表情参数，形状: {expression.shape}")
            print(f"已应用API返回的下颚参数，形状: {jaw_pose.shape}")
            
            # 处理原始NPZ中的所有其他参数
            for key, value in flame_param_copy.items():
                if key in final_flame_param:  # 已处理的参数(expr, jaw_pose)
                    continue
                
                original_param = value  # 原始NumPy数据
                param_original_frames = original_param.shape[0] if original_param.ndim > 0 else 0
                
                # 检查是否是需要按时间调整的参数
                # 静态参数或维度不匹配原始帧数的，直接保留
                is_dynamic_param = original_param.ndim > 0 and param_original_frames == original_frames
                
                if not is_dynamic_param:
                    final_flame_param[key] = original_param
                    print(f"  保持静态或非时序参数 '{key}' 形状: {original_param.shape}")
                    continue
                
                # 处理动态参数的帧数调整
                print(f"  调整动态参数 '{key}' 帧数从 {original_frames} 到 {target_frames}")
                
                if original_frames == target_frames:
                    # 帧数相同，无需调整
                    final_flame_param[key] = original_param
                    print(f"    帧数相同，无需调整。")
                
                elif original_frames > target_frames:
                    # 原始帧数更多，截断
                    final_flame_param[key] = original_param[:target_frames]
                    print(f"    原始帧数 > 目标帧数，已截断。")
                
                elif original_frames < target_frames:
                    # 原始帧数更少，需要用重复序列的方式填充
                    if original_frames == 0:
                        # 特殊情况：原始数据没有帧，用0填充
                        print(f"    警告：原始参数 '{key}' 帧数为 0，将用 0 填充至 {target_frames} 帧。")
                        target_shape = (target_frames,) + original_param.shape[1:]
                        final_flame_param[key] = np.zeros(target_shape, dtype=original_param.dtype)
                    else:
                        # 计算需要重复多少次原始序列（向上取整）
                        num_repeats = math.ceil(target_frames / original_frames)
                        # 构建重复次数的元组，只在时间轴（第0轴）重复
                        tile_repeats = (num_repeats,) + (1,) * (original_param.ndim - 1)
                        # 使用 np.tile 进行重复
                        tiled_data = np.tile(original_param, tile_repeats)
                        # 截取到所需的精确目标帧数
                        final_flame_param[key] = tiled_data[:target_frames]
                        print(f"    原始帧数 < 目标帧数，已通过重复原始序列 {num_repeats} 次并截断来填充。")
                else:  # original_frames == 0 and target_frames == 0，理论上不应发生但做个保护
                    final_flame_param[key] = original_param
                
                print(f"  最终参数 '{key}' 形状: {final_flame_param[key].shape}")
            
            print("FLAME参数替换完成!")
            
            # 将修改后的参数转回CUDA张量并更新到模型
            updated_flame_param = {}
            for k, v in final_flame_param.items():
                updated_flame_param[k] = torch.from_numpy(v).float().cuda()
            
            # 保存临时NPZ文件以备用
            if self.cfg.motion_path:
                temp_npz_path = str(self.cfg.motion_path) + ".temp.npz"
                print(f"保存临时NPZ文件: {temp_npz_path}")
                np.savez(temp_npz_path, **final_flame_param)
                print(f"临时NPZ文件保存成功: {temp_npz_path}")
            
            # 更新FLAME参数
            print("更新模型FLAME参数...")
            self.gaussians.flame_param = updated_flame_param
            self.gaussians.num_timesteps = target_frames
            print(f"模型参数更新成功! 新的时间步数: {target_frames}")
            
            # 更新第一帧
            self.timestep = 0
            dpg.set_value("_slider_timestep", self.timestep)
            dpg.configure_item("_slider_timestep", max_value=target_frames - 1)
            self.num_timesteps = target_frames
            self.gaussians.select_mesh_by_timestep(self.timestep)
            self.need_update = True
            
            print("FLAME参数更新成功")
            # 自动开始播放
            self.playing = True
            
            # 重置到起点
            dpg.set_value("_slider_record_timestep", 0)
            if hasattr(self, 'audio_frame_index'):
                self.audio_frame_index = 0
                
            # 停止当前播放（如果有）
            if self.audio_playing:
                self.stop_audio()
                
            # 启动音频播放
            self.start_audio()
            
            # 自动开始渲染动画
            self.playing = True
            return True
            
        except Exception as e:
            print(f"获取FLAME参数时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_api_connection(self):
        """测试API连接"""
        try:
            print(f"测试API连接: {self.cfg.api_url}")
            health_check = requests.get(f"{self.cfg.api_url}/health", timeout=2)
            if health_check.status_code == 200:
                print(f"API服务正常: {health_check.text}")
                return True
            else:
                print(f"API服务状态异常: 状态码 {health_check.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"无法连接到API服务 {self.cfg.api_url}")
            print("请确保API服务已启动并可访问")
            return False
        except requests.exceptions.Timeout:
            print(f"API服务连接超时")
            return False
        except Exception as e:
            print(f"API连接测试出错: {e}")
            return False

    def define_gui(self):
        super().define_gui()

        # window: rendering options ==================================================================================================
        with dpg.window(label="Render", tag="_render_window", autosize=True):

            with dpg.group(horizontal=True):
                dpg.add_text("FPS:", show=not self.cfg.demo_mode)
                dpg.add_text("0   ", tag="_log_fps", show=not self.cfg.demo_mode)

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

                    # # show original mesh
                    # def callback_original_mesh(sender, app_data):
                    #     self.original_mesh = app_data
                    #     self.need_update = True
                    # dpg.add_checkbox(label="original mesh", default_value=self.original_mesh, callback=callback_original_mesh)
            
            # timestep slider and buttons
            if self.num_timesteps != None:
                def callback_set_current_frame(sender, app_data):
                    if sender == "_slider_timestep":
                        self.timestep = app_data
                    elif sender in ["_button_timestep_plus", "_mvKey_Right"]:
                        self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                    elif sender in ["_button_timestep_minus", "_mvKey_Left"]:
                        self.timestep = max(self.timestep - 1, 0)
                    elif sender == "_mvKey_Home":
                        self.timestep = 0
                    elif sender == "_mvKey_End":
                        self.timestep = self.num_timesteps - 1

                    dpg.set_value("_slider_timestep", self.timestep)
                    print(f"选择时间步: {self.timestep}/{self.num_timesteps - 1}")
                    self.gaussians.select_mesh_by_timestep(self.timestep)
                    print(f"时间步 {self.timestep} 的网格已加载")

                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
                    dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)
                    dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=153, min_value=0, max_value=self.num_timesteps - 1, format="%d", default_value=0, callback=callback_set_current_frame)

            # # render_mode combo
            # def callback_change_mode(sender, app_data):
            #     self.render_mode = app_data
            #     self.need_update = True
            # dpg.add_combo(('rgb', 'depth', 'opacity'), label='render mode', default_value=self.render_mode, callback=callback_change_mode)

            # scaling_modifier slider
            def callback_set_scaling_modifier(sender, app_data):
                self.need_update = True
            dpg.add_slider_float(label="Scale modifier", min_value=0, max_value=1, format="%.2f", width=200, default_value=1, callback=callback_set_scaling_modifier, tag="_slider_scaling_modifier")

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True
            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, width=200, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy, tag="_slider_fovy", show=not self.cfg.demo_mode)

            if self.gaussians.binding is not None:
                # visualization options
                def callback_visual_options(sender, app_data):
                    if app_data == 'number of points per face':
                        value, ct = self.gaussians.binding.unique(return_counts=True)
                        ct = torch.log10(ct + 1)
                        ct = ct.float() / ct.max()
                        cmap = matplotlib.colormaps["plasma"]
                        self.face_colors = torch.from_numpy(cmap(ct.cpu())[None, :, :3]).to(self.gaussians.verts)
                    else:
                        self.face_colors = self.mesh_color[:3].to(self.gaussians.verts)[None, None, :].repeat(1, self.gaussians.face_center.shape[0], 1)  # (1, F, 3)
                    
                    dpg.set_value('_checkbox_show_mesh', True)
                    self.need_update = True
                dpg.add_combo(["none", "number of points per face"], default_value="none", label='visualization', width=200, callback=callback_visual_options, tag="_visual_options")

                # mesh_color picker
                def callback_change_mesh_color(sender, app_data):
                    self.mesh_color = torch.tensor(app_data, dtype=torch.float32)  # only need RGB in [0, 1]
                    if dpg.get_value("_visual_options") == 'none':
                        self.face_colors = self.mesh_color[:3].to(self.gaussians.verts)[None, None, :].repeat(1, self.gaussians.face_center.shape[0], 1)
                    self.need_update = True
                dpg.add_color_edit((self.mesh_color*255).tolist(), label="Mesh Color", width=200, callback=callback_change_mesh_color, show=not self.cfg.demo_mode)

            # camera
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    self.need_update = True
                    dpg.set_value("_slider_fovy", self.cam.fovy)
                dpg.add_button(label="reset camera", tag="_button_reset_pose", callback=callback_reset_camera, show=not self.cfg.demo_mode)
                
                def callback_cache_camera(sender, app_data):
                    self.cam.save()
                dpg.add_button(label="cache camera", tag="_button_cache_pose", callback=callback_cache_camera, show=not self.cfg.demo_mode)

                def callback_clear_cache(sender, app_data):
                    self.cam.clear()
                dpg.add_button(label="clear cache", tag="_button_clear_cache", callback=callback_clear_cache, show=not self.cfg.demo_mode)
                
        # window: audio controls ==================================================================================================
        if self.audio_data is not None:
            with dpg.window(label="音频控制", tag="_audio_window", autosize=True, pos=(self.W-300, self.H//2)):
                # 显示音频信息
                audio_duration = len(self.audio_data) / self.sample_rate
                dpg.add_text(f"音频文件: {self.cfg.audio_path.name}")
                dpg.add_text(f"时长: {audio_duration:.2f}秒")
                dpg.add_text(f"采样率: {self.sample_rate}Hz")
                
                # 自动模式
                def callback_auto_mode(sender, app_data):
                    self.auto_mode = app_data
                    # 更新界面显示
                    if self.auto_mode:
                        # 隐藏复杂的Record窗口控件
                        dpg.configure_item("_keyframes_group", show=False)
                        dpg.configure_item("_keyframe_edit_group", show=False)
                    else:
                        # 显示所有Record窗口控件
                        dpg.configure_item("_keyframes_group", show=True)
                        dpg.configure_item("_keyframe_edit_group", show=True)
                dpg.add_checkbox(label="自动模式 (简化界面)", default_value=self.auto_mode, 
                                callback=callback_auto_mode, tag="_checkbox_auto_mode")
                
                # 重置关键帧按钮
                def callback_reset_keyframes(sender, app_data):
                    # 停止任何正在进行的播放
                    self.playing = False
                    if self.audio_playing:
                        self.stop_audio()
                    
                    # 清除现有关键帧
                    self.keyframes = []
                    dpg.configure_item("_listbox_keyframes", items=[])
                    
                    # 设置首帧关键帧
                    self.timestep = 0
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)
                    first_state = self.get_state_dict()
                    self.keyframes.append(first_state)
                    
                    # 设置末帧关键帧
                    if self.num_timesteps > 1:
                        self.timestep = self.num_timesteps - 1
                        dpg.set_value("_slider_timestep", self.timestep)
                        self.gaussians.select_mesh_by_timestep(self.timestep)
                        last_state = self.get_state_dict()
                        self.keyframes.append(last_state)
                        
                        # 返回第一帧
                        self.timestep = 0
                        dpg.set_value("_slider_timestep", self.timestep)
                        self.gaussians.select_mesh_by_timestep(self.timestep)
                    
                    # 更新关键帧列表
                    dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
                    
                    # 设置循环
                    dpg.set_value("_input_cycles", 1)
                    
                    # 更新时间线
                    self.update_record_timeline()
                    
                    # 重置到起点
                    dpg.set_value("_slider_record_timestep", 0)
                    
                    # 提示用户
                    print("关键帧已重置，已自动设置第一帧和最后一帧作为关键帧")
                
                dpg.add_button(label="重置关键帧", tag="_button_reset_keyframes", callback=callback_reset_keyframes)
                
                # 音频偏移设置
                def callback_set_audio_offset(sender, app_data):
                    self.cfg.audio_offset = app_data
                    # 如果正在播放，立即应用偏移
                    if self.audio_playing:
                        self.stop_audio()
                        self.start_audio()
                dpg.add_slider_float(label="音频偏移 (秒)", min_value=-5.0, max_value=5.0, format="%.2f", 
                                    default_value=self.cfg.audio_offset, callback=callback_set_audio_offset, width=200)
                
                # 锁定帧率
                def callback_lock_frame_rate(sender, app_data):
                    self.cfg.lock_frame_rate = app_data
                dpg.add_checkbox(label="锁定帧率", default_value=self.cfg.lock_frame_rate, 
                                callback=callback_lock_frame_rate, tag="_checkbox_lock_frame_rate")
                
                # 音频位置
                dpg.add_text("音频位置: 0.00秒", tag="_text_audio_position")
                
                # 音频播放进度条
                def callback_set_audio_position(sender, app_data):
                    # 设置新的时间线位置
                    frame_position = int(app_data * self.num_record_timeline / audio_duration)
                    frame_position = max(0, min(frame_position, self.num_record_timeline-1))
                    dpg.set_value("_slider_record_timestep", frame_position)
                    
                    # 更新状态
                    state_dict = self.get_state_dict_record()
                    self.apply_state_dict(state_dict)
                    self.need_update = True
                    
                    # 如果正在播放，更新音频位置
                    if self.audio_playing:
                        self.audio_frame_index = frame_position
                
                dpg.add_slider_float(label="播放进度", width=200, tag="_slider_audio_position",
                                     min_value=0.0, max_value=audio_duration, 
                                     format="%.2f秒", default_value=0.0, 
                                     callback=callback_set_audio_position)
                
                # 控制按钮
                with dpg.group(horizontal=True):
                    def callback_audio_play(sender, app_data):
                        if not self.audio_playing:
                            self.start_audio()
                            # 同时开始渲染动画
                            self.playing = True
                            self.need_update = True
                    dpg.add_button(label="播放", tag="_button_audio_play", callback=callback_audio_play)
                    
                    def callback_audio_stop(sender, app_data):
                        self.stop_audio()
                        self.playing = False
                    dpg.add_button(label="停止", tag="_button_audio_stop", callback=callback_audio_stop)
                    
                    def callback_audio_restart(sender, app_data):
                        self.stop_audio()
                        # 重置到起点
                        dpg.set_value("_slider_record_timestep", 0)
                        # 更新状态
                        state_dict = self.get_state_dict_record()
                        self.apply_state_dict(state_dict)
                        self.need_update = True
                    dpg.add_button(label="重新开始", tag="_button_audio_restart", callback=callback_audio_restart)
                
                # 添加自动播放按钮
                def callback_auto_play(sender, app_data):
                    # 自动创建关键帧
                    try:
                        if len(self.keyframes) == 0:
                            # 检查时间步数
                            if self.num_timesteps is None or self.num_timesteps < 1:
                                print("警告: 没有足够的时间步数来设置关键帧")
                                return
                                
                            # 添加第一帧
                            self.timestep = 0
                            dpg.set_value("_slider_timestep", self.timestep)
                            self.gaussians.select_mesh_by_timestep(self.timestep)
                            first_state = self.get_state_dict()
                            self.keyframes.append(first_state)
                            
                            # 如果有多个时间步，添加末帧
                            if self.num_timesteps > 1:
                                old_timestep = self.timestep
                                self.timestep = self.num_timesteps - 1
                                dpg.set_value("_slider_timestep", self.timestep)
                                self.gaussians.select_mesh_by_timestep(self.timestep)
                                
                                # 添加末帧
                                last_state = self.get_state_dict()
                                self.keyframes.append(last_state)
                                
                                # 更新关键帧列表
                                dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
                                
                                # 恢复原位置
                                self.timestep = old_timestep
                                dpg.set_value("_slider_timestep", self.timestep)
                                self.gaussians.select_mesh_by_timestep(self.timestep)
                            
                            # 设置循环
                            dpg.set_value("_input_cycles", 1)
                        
                        # 更新时间线
                        self.update_record_timeline()
                        
                        # 如果没有有效的时间线长度，无法播放
                        if self.num_record_timeline <= 0:
                            print("警告: 无法创建有效的播放时间线。请检查关键帧设置。")
                            return
                            
                        # 重置到起点
                        dpg.set_value("_slider_record_timestep", 0)
                        
                        # 开始播放
                        self.playing = True
                        self.need_update = True
                        
                        # 开始音频
                        if self.audio_data is not None and not self.audio_playing:
                            self.start_audio()
                    except Exception as e:
                        print(f"自动播放时出错: {e}")
                        import traceback
                        traceback.print_exc()
                
                dpg.add_button(label="自动播放", tag="_button_auto_play", callback=callback_auto_play)
        
        # window: recording ==================================================================================================
        with dpg.window(label="Record", tag="_record_window", autosize=True, pos=(0, self.H//2)):
            dpg.add_text("Keyframes")
            with dpg.group(horizontal=True, tag="_keyframes_group", show=not self.auto_mode):
                # list keyframes
                def callback_set_current_keyframe(sender, app_data):
                    idx = int(dpg.get_value("_listbox_keyframes"))
                    self.apply_state_dict(self.keyframes[idx])

                    record_timestep = sum([keyframe['interval'] for keyframe in self.keyframes[:idx]])
                    dpg.set_value("_slider_record_timestep", record_timestep)

                    self.need_update = True
                dpg.add_listbox(self.keyframes, width=200, tag="_listbox_keyframes", callback=callback_set_current_keyframe)

                # edit keyframes
                with dpg.group(tag="_keyframe_edit_group", show=not self.auto_mode):
                    # add
                    def callback_add_keyframe(sender, app_data):
                        if len(self.keyframes) == 0:
                            new_idx = 0
                        else:
                            new_idx = int(dpg.get_value("_listbox_keyframes")) + 1

                        states = self.get_state_dict()
                        
                        self.keyframes.insert(new_idx, states)
                        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
                        dpg.set_value("_listbox_keyframes", new_idx)

                        self.update_record_timeline()
                    dpg.add_button(label="add", tag="_button_add_keyframe", callback=callback_add_keyframe)

                    # delete
                    def callback_delete_keyframe(sender, app_data):
                        idx = int(dpg.get_value("_listbox_keyframes"))
                        self.keyframes.pop(idx)
                        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
                        dpg.set_value("_listbox_keyframes", idx-1)

                        self.update_record_timeline()
                    dpg.add_button(label="delete", tag="_button_delete_keyframe", callback=callback_delete_keyframe)

                    # update
                    def callback_update_keyframe(sender, app_data):
                        if len(self.keyframes) == 0:
                            return
                        else:
                            idx = int(dpg.get_value("_listbox_keyframes"))

                        states = self.get_state_dict()
                        states['interval'] = self.cfg.fps*self.cfg.keyframe_interval

                        self.keyframes[idx] = states
                    dpg.add_button(label="update", tag="_button_update_keyframe", callback=callback_update_keyframe)

            with dpg.group(horizontal=True):
                def callback_set_record_cycles(sender, app_data):
                    self.update_record_timeline()
                dpg.add_input_int(label="cycles", tag="_input_cycles", default_value=0, width=70, callback=callback_set_record_cycles)

                def callback_set_keyframe_interval(sender, app_data):
                    self.cfg.keyframe_interval = app_data
                    for keyframe in self.keyframes:
                        keyframe['interval'] = self.cfg.fps*self.cfg.keyframe_interval
                    self.update_record_timeline()
                dpg.add_input_int(label="interval", tag="_input_interval", default_value=self.cfg.keyframe_interval, width=70, callback=callback_set_keyframe_interval)
            
            def callback_set_record_timestep(sender, app_data):
                state_dict = self.get_state_dict_record()
                
                self.apply_state_dict(state_dict)
                self.need_update = True
                
                # 更新音频帧索引，使音频与视觉同步
                if self.audio_playing:
                    self.audio_frame_index = app_data
            dpg.add_slider_int(label="timeline", tag='_slider_record_timestep', width=200, min_value=0, max_value=0, format="%d", default_value=0, callback=callback_set_record_timestep)
            
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="dynamic", default_value=False, tag="_checkbox_dynamic_record")
                dpg.add_checkbox(label="loop", default_value=True, tag="_checkbox_loop_record")
            
            with dpg.group(horizontal=True):
                def callback_play(sender, app_data):
                    # 如果没有关键帧，自动创建关键帧
                    if len(self.keyframes) == 0:
                        try:
                            # 直接调用auto_play功能
                            callback_auto_play(None, None)
                            return
                        except Exception as e:
                            print(f"自动创建关键帧失败: {e}")
                            return
                            
                    # 如果时间线为空，无法播放
                    if self.num_record_timeline <= 0:
                        print("警告: 无法播放，请先添加有效的关键帧")
                        return
                        
                    self.playing = not self.playing
                    self.need_update = True
                    
                    # 同步音频播放状态
                    if self.playing and self.audio_data is not None and not self.audio_playing:
                        self.start_audio()
                    elif not self.playing and self.audio_playing:
                        self.stop_audio()
                dpg.add_button(label="play", tag="_button_play", callback=callback_play)

                def callback_export_trajectory(sender, app_data):
                    self.export_trajectory()
                dpg.add_button(label="export traj", tag="_button_export_traj", callback=callback_export_trajectory)
            
            def callback_save_image(sender, app_data):
                if not self.cfg.save_folder.exists():
                    self.cfg.save_folder.mkdir(parents=True)
                path = self.cfg.save_folder / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
                print(f"Saving image to {path}")
                Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
            with dpg.group(horizontal=True):
                dpg.add_button(label="save image", tag="_button_save_image", callback=callback_save_image)
        
        # window: FLAME ==================================================================================================
        if self.gaussians.binding is not None:
            with dpg.window(label="FLAME parameters", tag="_flame_window", autosize=True, pos=(self.W-300, 0)):
                def callback_enable_control(sender, app_data):
                    if app_data:
                        self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    else:
                        self.gaussians.select_mesh_by_timestep(self.timestep)
                    self.need_update = True
                dpg.add_checkbox(label="enable control", default_value=False, tag="_checkbox_enable_control", callback=callback_enable_control)

                dpg.add_separator()

                def callback_set_pose(sender, app_data):
                    joint, axis = sender.split('-')[1:3]
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    self.flame_param[joint][0, axis_idx] = app_data
                    if joint == 'eyes':
                        self.flame_param[joint][0, 3+axis_idx] = app_data
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                dpg.add_text(f'Joints')
                self.pose_sliders = []
                max_rot = 0.5
                for joint in ['neck', 'jaw', 'eyes']:
                    if joint in self.flame_param:
                        with dpg.group(horizontal=True):
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 0], callback=callback_set_pose, tag=f"_slider-{joint}-x", width=70)
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 1], callback=callback_set_pose, tag=f"_slider-{joint}-y", width=70)
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 2], callback=callback_set_pose, tag=f"_slider-{joint}-z", width=70)
                            self.pose_sliders.append(f"_slider-{joint}-x")
                            self.pose_sliders.append(f"_slider-{joint}-y")
                            self.pose_sliders.append(f"_slider-{joint}-z")
                            dpg.add_text(f'{joint:4s}')
                dpg.add_text('   roll       pitch      yaw')
                
                dpg.add_separator()
                
                def callback_set_expr(sender, app_data):
                    expr_i = int(sender.split('-')[2])
                    self.flame_param['expr'][0, expr_i] = app_data
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                self.expr_sliders = []
                dpg.add_text(f'Expressions')
                for i in range(5):
                    dpg.add_slider_float(label=f"{i}", min_value=-3, max_value=3, format="%.2f", default_value=0, callback=callback_set_expr, tag=f"_slider-expr-{i}", width=250)
                    self.expr_sliders.append(f"_slider-expr-{i}")

                def callback_reset_flame(sender, app_data):
                    self.reset_flame_param()
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                    for slider in self.pose_sliders + self.expr_sliders:
                        dpg.set_value(slider, 0)
                dpg.add_button(label="reset FLAME", tag="_button_reset_flame", callback=callback_reset_flame)
                
                dpg.add_separator()
                
                # 添加API获取FLAME参数的功能
                dpg.add_text("从API获取FLAME参数")
                
                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="角色ID", default_value="M003", tag="_input_subject_style", width=100)
                    dpg.add_combo(
                        ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"], 
                        label="情绪类型", 
                        default_value="neutral", 
                        tag="_combo_emotion", 
                        width=100
                    )
                
                with dpg.group(horizontal=True):
                    dpg.add_slider_int(label="情绪强度", min_value=0, max_value=2, default_value=2, tag="_slider_intensity", width=100)
                    
                    def callback_get_flame_from_api(sender, app_data):
                        subject_style = dpg.get_value("_input_subject_style")
                        emotion = dpg.get_value("_combo_emotion")
                        intensity = dpg.get_value("_slider_intensity")
                        
                        if self.cfg.audio_path is None:
                            print("请先设置音频文件路径")
                            return
                        
                        success = self.get_flame_params_from_api(
                            str(self.cfg.audio_path), 
                            subject_style=subject_style,
                            emotion=emotion,
                            intensity=intensity
                        )
                        
                        if success:
                            # 使控制生效
                            dpg.set_value("_checkbox_enable_control", True)
                            # 更新UI
                            self.need_update = True
                            
                            # 重置到起点
                            dpg.set_value("_slider_record_timestep", 0)
                            if hasattr(self, 'audio_frame_index'):
                                self.audio_frame_index = 0
                                
                            # 停止当前播放（如果有）
                            if self.audio_playing:
                                self.stop_audio()
                                
                            # 启动音频播放
                            self.start_audio()
                            
                            # 自动开始渲染动画
                            self.playing = True

        # widget-dependent handlers ========================================================================================
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')

            def callbackmouse_wheel_slider(sender, app_data):
                delta = app_data
                if dpg.is_item_hovered("_slider_timestep"):
                    self.timestep = min(max(self.timestep - delta, 0), self.num_timesteps - 1)
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)
                    self.need_update = True
            dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel_slider)

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    @torch.no_grad()
    def run(self):
        print("Running LocalViewer...")
        print(f"配置信息:")
        print(f"- 点云路径: {self.cfg.point_path}")
        print(f"- 动作路径: {self.cfg.motion_path}")
        print(f"- 音频路径: {self.cfg.audio_path}")
        print(f"- API URL: {self.cfg.api_url}")
        print(f"- 帧率: {self.cfg.fps}")
        print(f"- 音频偏移: {self.cfg.audio_offset}")
        print(f"- 自动模式: {self.auto_mode}")
        
        if self.gaussians.binding is not None:
            print(f"FLAME模型已加载:")
            print(f"- 总时间步数: {self.num_timesteps}")
            print(f"- FLAME参数:")
            for k, v in self.gaussians.flame_param.items():
                print(f"  - {k}: {v.shape}")
        else:
            print("未加载FLAME模型")
        
        # 帧率控制变量
        last_frame_time = time.time()
        target_frame_time = 1.0 / self.cfg.fps if self.cfg.fps > 0 else 0.04  # 默认25fps

        while dpg.is_dearpygui_running():
            current_time = time.time()
            
            # 更新音频位置显示
            if self.audio_playing and hasattr(self, 'audio_frame_index') and self.audio_data is not None:
                audio_position = self.audio_frame_index * self.audio_frame_duration
                # 更新文本显示
                if dpg.does_item_exist("_text_audio_position"):
                    dpg.set_value("_text_audio_position", f"音频位置: {audio_position:.2f}秒")
                # 更新进度条，但不触发回调以避免循环
                if dpg.does_item_exist("_slider_audio_position"):
                    # 临时移除回调
                    callback = dpg.get_item_callback("_slider_audio_position")
                    dpg.set_item_callback("_slider_audio_position", None)
                    # 更新值
                    dpg.set_value("_slider_audio_position", audio_position)
                    # 恢复回调
                    dpg.set_item_callback("_slider_audio_position", callback)

            if self.need_update or self.playing:
                # 帧率锁定 - 重要：这确保音频和视觉同步
                if self.playing and self.cfg.lock_frame_rate:
                    elapsed = current_time - last_frame_time
                    if elapsed < target_frame_time:
                        time.sleep(target_frame_time - elapsed)
                        current_time = time.time()
                
                # 渲染当前帧
                cam = self.prepare_camera()

                if dpg.get_value("_checkbox_show_splatting"):
                    # rgb
                    rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, torch.tensor(self.cfg.background_color).cuda(), scaling_modifier=dpg.get_value("_slider_scaling_modifier"))["render"].permute(1, 2, 0).contiguous()

                if self.gaussians.binding is not None and dpg.get_value("_checkbox_show_mesh"):
                    out_dict = self.mesh_renderer.render_from_camera(self.gaussians.verts, self.gaussians.faces, cam, face_colors=self.face_colors)

                    rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
                    rgb_mesh = rgba_mesh[:, :, :3]
                    alpha_mesh = rgba_mesh[:, :, 3:]
                    mesh_opacity = self.mesh_color[3:].cuda()

                if dpg.get_value("_checkbox_show_splatting") and dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_mesh * alpha_mesh * mesh_opacity  + rgb_splatting * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                elif dpg.get_value("_checkbox_show_splatting") and not dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_splatting
                elif not dpg.get_value("_checkbox_show_splatting") and dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_mesh
                else:
                    rgb = torch.ones([self.H, self.W, 3])

                self.render_buffer = rgb.cpu().numpy()
                if self.render_buffer.shape[0] != self.H or self.render_buffer.shape[1] != self.W:
                    continue
                dpg.set_value("_texture", self.render_buffer)

                self.refresh_stat()
                self.need_update = False
                last_frame_time = current_time

                if self.playing:
                    record_timestep = dpg.get_value("_slider_record_timestep")
                    next_timestep = record_timestep + 1
                    
                    # 检查是否到达末尾
                    if next_timestep >= self.num_record_timeline - 1:
                        if not dpg.get_value("_checkbox_loop_record"):
                            self.playing = False
                            if self.audio_playing:
                                self.stop_audio()
                        next_timestep = 0
                    
                    # 更新时间线
                    dpg.set_value("_slider_record_timestep", next_timestep)
                    
                    # 更新音频帧索引 - 重要：这让音频和视觉保持同步
                    if self.audio_playing:
                        self.audio_frame_index = next_timestep
                    
                    # 更新动态时间步
                    if dpg.get_value("_checkbox_dynamic_record"):
                        self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                        dpg.set_value("_slider_timestep", self.timestep)
                        self.gaussians.select_mesh_by_timestep(self.timestep)

                    # 应用状态
                    state_dict = self.get_state_dict_record()
                    self.apply_state_dict(state_dict)

            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    
    # 如果设置了test_api，先测试API是否可用
    if cfg.test_api:
        print("测试API连接...")
        try:
            api_url = cfg.api_url
            health_check = requests.get(f"{api_url}/health", timeout=2)
            if health_check.status_code == 200:
                print(f"API服务正常: {health_check.text}")
                
                # 尝试直接用API生成FLAME参数
                audio_path = None
                
                # 优先使用server_audio_path参数
                if cfg.server_audio_path:
                    audio_path = cfg.server_audio_path
                    print(f"使用服务器上的音频路径: {audio_path}")
                # 否则尝试使用audio_path参数
                elif cfg.audio_path:
                    audio_path = str(cfg.audio_path)
                    if not os.path.isabs(audio_path):
                        abs_audio_path = os.path.abspath(audio_path)
                        print(f"将本地路径转换为绝对路径: {abs_audio_path}")
                        audio_path = abs_audio_path
                    print(f"使用本地音频路径: {audio_path}")
                    print("警告: 服务器可能无法访问此本地路径，请考虑使用--server_audio_path参数")
                
                if audio_path:
                    # 构建请求
                    flame_api_url = f"{api_url}/api/flame_from_path"
                    payload = {
                        "audio_path": audio_path,
                        "subject_style": "M003",
                        "emotion": "neutral",
                        "intensity": 2
                    }
                    
                    print(f"发送请求到 {flame_api_url}")
                    print(f"请求数据: {payload}")
                    
                    try:
                        response = requests.post(flame_api_url, json=payload, timeout=30)
                        if response.status_code == 200:
                            print("API请求成功!")
                            try:
                                result = response.json()
                                print(f"返回数据包含以下键: {list(result.keys())}")
                                print(f"返回的元数据: {result.get('metadata', {})}")
                                if 'expression' in result:
                                    print(f"表情参数形状: {np.array(result['expression']).shape}")
                                if 'jaw_pose' in result:
                                    print(f"下颚参数形状: {np.array(result['jaw_pose']).shape}")
                            except:
                                print("无法解析返回的JSON数据")
                        else:
                            print(f"API请求失败: 状态码 {response.status_code}")
                            print(f"错误信息: {response.text}")
                    except Exception as e:
                        print(f"请求过程中发生错误: {e}")
                else:
                    print("未指定音频路径，跳过API请求测试")
                    print("请使用--server_audio_path参数指定服务器上的音频路径")
                    print("例如: --server_audio_path=/home/plm/inferno/assets/data/EMOTE_test_example_data/02_that.wav")
            else:
                print(f"API服务状态异常: 状态码 {health_check.status_code}")
                print("程序将继续但API功能可能无法使用")
        except Exception as e:
            print(f"API连接测试失败: {e}")
            print("程序将继续但API功能可能无法使用")
    
    gui = LocalViewer(cfg)
    gui.run()
