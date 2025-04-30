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
import signal
from scipy import signal as signal_scipy
import queue
import traceback

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
    fps: int = 30
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
    # audio_offset: float = 0.0
    # """Audio offset in seconds"""
    api_url: str = "http://10.112.241.111:5001"
    """API服务器URL"""
    tts_api_url: str = "http://10.112.208.173:5001"
    """TTS API服务器URL"""
    use_pulseaudio: bool = False
    """使用PulseAudio播放音频"""
    pulseaudio_server: str = "100.127.107.62:4713"
    """PulseAudio服务器地址"""
    test_api: bool = False
    """启动时测试API连接"""
    render_timeout: float = 5.0
    """Maximum time allowed for a single frame render (seconds)"""
    temp_audio_path: Path = Path("./temp_audio.wav")
    """临时音频文件保存路径"""

class LocalViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # 创建必要的目录
        # 确保临时音频路径存在有效的父目录
        if str(self.cfg.temp_audio_path) == "" or os.path.dirname(str(self.cfg.temp_audio_path)) == "":
            self.cfg.temp_audio_path = Path("./temp_audio.wav")
            print(f"临时音频路径无效，已设置为默认路径: {self.cfg.temp_audio_path}")
        
        os.makedirs(os.path.dirname(str(self.cfg.temp_audio_path)) if os.path.dirname(str(self.cfg.temp_audio_path)) else ".", exist_ok=True)
        os.makedirs(self.cfg.save_folder, exist_ok=True)
        
        # 设置中文字体支持
        dpg.create_context()
        if is_windows:
            # Windows特定字体设置
            # 检查华文楷体字体是否存在，如果不存在则使用系统默认字体
            font_path = "华文楷体.ttf"
            if not os.path.exists(font_path):
                # 尝试使用Windows常见中文字体
                windows_fonts = [
                    "C:/Windows/Fonts/simkai.ttf",  # 楷体
                    "C:/Windows/Fonts/simhei.ttf",  # 黑体
                    "C:/Windows/Fonts/simsun.ttc",  # 宋体
                    "C:/Windows/Fonts/msyh.ttc"     # 微软雅黑
                ]
                for wfont in windows_fonts:
                    if os.path.exists(wfont):
                        font_path = wfont
                        print(f"使用中文字体: {font_path}")
                        break
                else:
                    print("未找到合适的中文字体，将使用默认字体")
                    
            try:
                with dpg.font_registry():
                    with dpg.font(font_path, 18) as default_font:
                        dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                        dpg.bind_font(default_font)
            except Exception as e:
                print(f"加载中文字体失败: {e}")
        else:
            # Linux字体支持
            try:
                linux_fonts = [
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"  # Droid Sans
                ]
                for font_path in linux_fonts:
                    if os.path.exists(font_path):
                        print(f"使用Linux中文字体: {font_path}")
                        with dpg.font_registry():
                            with dpg.font(font_path, 18) as default_font:
                                dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                                dpg.bind_font(default_font)
                        break
            except Exception as e:
                print(f"Linux加载中文字体失败: {e}")
        
        # recording settings
        self.keyframes = []  # list of state dicts of keyframes
        self.all_frames = {}  # state dicts of all frames {key: [num_frames, ...]}
        self.num_record_timeline = 0
        self.playing = False
        
        # 自动模式 - 简化用户界面
        self.auto_mode = True
        
        # 渲染和性能优化相关变量
        self.need_frame_update = False
        self.next_frame = 0
        self.should_skip_complex_frames = True  # 自动跳过复杂帧
        self.max_render_time = 100.0  # 毫秒，大于这个值的帧会被标记为复杂帧
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
        
        # audio settings
        self.audio_data = None
        self.sample_rate = None
        self.current_sample = 0  # 当前音频样本索引，用于音频回调函数
        self.audio_frame_duration = 1.0 / self.cfg.fps if self.cfg.fps > 0 else 0.04  # default 25fps
        self.audio_playing = False
        self.audio_stream = None
        self.audio_thread = None
        self.audio_time = 0
        self.audio_frame_index = 0
        
        # 检测可用的音频设备
        self.check_audio_devices()
        
        # 设置PulseAudio环境
        self.pulse_sink_id = 0
        if self.cfg.use_pulseaudio:
            self.setup_pulseaudio_env()
        
        # 加载音频文件
        if self.cfg.audio_path is not None:
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
        else:
            print("Warning: No FLAME model binding available, some features may be limited")
        
        # 设置Ctrl+C处理
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # 打印同步模式信息
        sync_mode = "音频驱动模式" if self.cfg.lock_frame_rate else "时间驱动模式"
        print(f"当前同步模式: {sync_mode}")
        if self.cfg.lock_frame_rate:
            print("  - 视频帧将跟随音频位置，确保音视频同步")
        else:
            print("  - 音频将跟随视频帧位置，基于定时器控制")
        
        # 如果处于自动模式且加载了音频，自动设置关键帧
        if self.auto_mode and self.audio_data is not None and self.gaussians.binding is not None:
            # 延迟执行，确保GUI已经完全初始化
            def auto_setup():
                """自动设置模式"""
                print("自动模式：开始自动配置...")
                try:
                    # 检查是否有本地音频文件
                    if self.cfg.audio_path is not None and self.cfg.audio_path.exists():
                        print(f"自动模式：使用本地音频路径: {self.cfg.audio_path}")
                        
                        # 尝试从API获取FLAME参数
                        print("自动模式：尝试从API获取FLAME参数...")
                        try:
                            # 尝试使用本地路径或者服务器路径请求API
                            audio_path = None
                            if self.cfg.server_audio_path:
                                audio_path = self.cfg.server_audio_path
                                print(f"使用服务器上的音频路径: {audio_path}")
                            else:
                                audio_path = str(self.cfg.audio_path)
                                print(f"使用本地音频路径: {audio_path}")
                                print("提示: 示例服务器音频路径格式: /home/plm/inferno/assets/data/EMOTE_test_example_data/02_that.wav")
                            
                            self.get_flame_params_from_api(audio_path)
                            
                            # 如果成功，设置播放控制
                            if dpg.does_item_exist("_input_cycles"):
                                dpg.set_value("_input_cycles", 1)
                            if dpg.does_item_exist("_slider_timestep"):
                                dpg.set_value("_slider_timestep", 0)
                            
                        except Exception as e:
                            print(f"自动模式：无法从API获取FLAME参数，将使用默认关键帧设置")
                            print(f"错误: {e}")
                    else:
                        print("自动模式：未找到音频文件，将使用默认设置")
                except Exception as e:
                    print(f"自动设置过程中出错: {e}")
                
                # 确保UI是最新的
                try:
                    self.update_record_timeline()
                except Exception as e:
                    print(f"更新时间线出错: {e}")
                
                # 开始渲染第一帧
                try:
                    self.need_update = True
                except Exception as e:
                    print(f"触发首帧渲染出错: {e}")
            
            # 创建线程执行自动设置
            auto_thread = threading.Thread(target=auto_setup)
            auto_thread.daemon = True
            auto_thread.start()
        
    def load_audio(self):
        """加载音频文件"""
        if self.cfg.audio_path is not None and self.cfg.audio_path.exists():
            try:
                print(f"Loading audio file: {self.cfg.audio_path}")
                
                # 固定采样率为16000Hz
                self.sample_rate = 16000
                
                # 检查是否需要重采样
                audio_data, original_sr = sf.read(self.cfg.audio_path)
                
                if original_sr != self.sample_rate:
                    print(f"Resampling audio from {original_sr}Hz to {self.sample_rate}Hz")
                    # 使用FFmpeg进行重采样
                    try:
                        # 创建临时文件用于重采样输出
                        temp_audio_path = str(self.cfg.temp_audio_path) + ".temp.wav"
                        
                        # 调用FFmpeg进行重采样到16000Hz，保持音频时长不变
                        ffmpeg_cmd = [
                            'ffmpeg', '-y',
                            '-i', str(self.cfg.audio_path),
                            '-ar', str(self.sample_rate),
                            '-af', 'aresample=resampler=soxr',
                            '-ac', '2',
                            temp_audio_path
                        ]
                        
                        print(f"执行FFmpeg重采样: {' '.join(ffmpeg_cmd)}")
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                        
                        # 加载重采样后的音频
                        self.audio_data, actual_sr = sf.read(temp_audio_path)
                        print(f"Audio resampled with FFmpeg: {self.audio_data.shape}, sr={actual_sr}Hz")
                        
                        # 清理临时文件
                        try:
                            os.remove(temp_audio_path)
                        except:
                            pass
                    except Exception as e:
                        print(f"FFmpeg重采样失败: {e}")
                        print("回退到使用scipy进行重采样...")
                        # 使用scipy进行重采样（备选方案）
                        num_samples = int(len(audio_data) * self.sample_rate / original_sr)
                        self.audio_data = signal_scipy.resample(audio_data, num_samples)
                else:
                    # 不需要重采样
                    self.audio_data = audio_data
                
                # 如果是单声道，转换为立体声
                if len(self.audio_data.shape) == 1:
                    self.audio_data = np.stack([self.audio_data, self.audio_data], axis=1)
                
                print(f"Audio loaded: {self.audio_data.shape}, sr={self.sample_rate}Hz")
                self.current_sample = 0
            except Exception as e:
                print(f"Error loading audio file: {e}")
                import traceback
                traceback.print_exc()
                self.audio_data = None
                self.sample_rate = None
        else:
            print("No audio file specified or file does not exist.")
            
    def audio_callback(self, outdata, frames, time, status):
        """音频回调函数，用于提供音频数据流"""
        if status:
            print(f"Audio callback status: {status}")
            
        # 如果用户停止了播放或者应用关闭，返回静音数据
        if not self.audio_playing or not dpg.is_dearpygui_running():
            outdata.fill(0)
            return
        
        # 检查是否有有效的采样率和音频数据
        if self.sample_rate is None or self.audio_data is None:
            print("Warning: Audio callback found sample rate or audio data is None")
            outdata.fill(0)
            return
            
        try:
            # 计算当前音频帧对应的音频样本索引
            sample_index = int(self.timestep / self.cfg.fps * self.sample_rate) if self.cfg.fps > 0 else 0
            sample_index = max(0, sample_index) # 确保索引不为负
            
            # 当lock_frame_rate为False时，使用帧索引控制音频位置
            # 当lock_frame_rate为True时，使用流式播放，音频自然流动，视频跟随音频
            if not self.cfg.lock_frame_rate:
                # 调试信息
                # print(f"Audio callback: frame={self.timestep}, sample={sample_index}, total={len(self.audio_data)}")
                
                # 更新当前样本索引（如果需要同步）
                self.current_sample = sample_index
                
                # 计算结束索引
                end_idx = sample_index + frames
            else:
                # 在锁定帧率模式下，允许音频自然流动
                # 计算结束索引
                end_idx = self.current_sample + frames
                
                # 更新当前样本索引，用于渲染帧追踪
                sample_index = self.current_sample
            
            # 检查是否已经播放到文件末尾
            if sample_index >= len(self.audio_data) or sample_index < 0:
                # 如果索引已经超出范围，填充静音
                outdata.fill(0)
                
                # 判断是否循环播放 - 不在音频回调中直接修改UI，只设置标志
                try:
                    if dpg.get_value("_checkbox_loop"):
                        # 在主线程中处理循环逻辑
                        self.next_frame = 0
                        self.need_frame_update = True
                        self.need_update = True
                        # 在lock_frame_rate模式下重置音频位置
                        if self.cfg.lock_frame_rate:
                            self.current_sample = 0
                        # 不直接调用UI更新
                        print("Audio position out of range, resetting to start")
                    else:
                        # 停止播放
                        self.audio_playing = False
                        print("Audio position out of range, stopping playback")
                except Exception as e:
                    print(f"Error checking loop state: {e}")
                    self.audio_playing = False
                return
            elif end_idx >= len(self.audio_data):
                # 填充剩余的部分
                remaining = len(self.audio_data) - sample_index
                if remaining > 0:
                    outdata[:remaining] = self.audio_data[sample_index:len(self.audio_data)]
                    outdata[remaining:] = 0
                else:
                    outdata.fill(0)
                
                # 判断是否循环播放
                try:
                    if dpg.get_value("_checkbox_loop"):
                        # 在主线程中处理循环逻辑
                        self.next_frame = 0
                        self.need_frame_update = True
                        self.need_update = True
                        # 在lock_frame_rate模式下重置音频位置
                        if self.cfg.lock_frame_rate:
                            self.current_sample = 0
                        # 不直接调用UI更新
                        print("Audio playback ended, looping back to start")
                    else:
                        # 停止播放
                        self.audio_playing = False
                        print("Audio playback completed")
                except Exception as e:
                    print(f"Error checking loop state: {e}")
                    self.audio_playing = False
                return
            else:
                # 正常播放
                try:
                    outdata[:] = self.audio_data[sample_index:end_idx]
                    # 在lock_frame_rate模式下更新current_sample
                    if self.cfg.lock_frame_rate:
                        self.current_sample = end_idx  # 更新到下一个音频位置
                except ValueError as e:
                    print(f"Audio buffer error: {e}, filling with zeros")
                    outdata.fill(0)
                except Exception as e:
                    print(f"Unexpected error in audio buffer: {e}")
                    outdata.fill(0)
        except Exception as e:
            print(f"Audio callback exception: {e}")
            import traceback
            traceback.print_exc()
            outdata.fill(0)  # 发生错误时输出静音
    
    def start_audio(self):
        """开始播放音频"""
        if self.audio_data is None:
            print("Cannot play audio: No audio data loaded")
            return
            
        if self.audio_playing:
            print("Audio is already playing")
            return
            
        def audio_player():
            try:
                print("Starting audio playback thread")
                
                # 从当前时间步的对应位置开始播放音频
                self.audio_frame_index = self.timestep
                
                # 计算开始时间(秒)
                start_time = self.audio_frame_index / self.cfg.fps
                print(f"Audio start position: Frame {self.audio_frame_index}, Time {start_time:.2f}s")
                
                # 初始化当前音频样本索引
                self.current_sample = int(start_time * self.sample_rate)
                print(f"Initialize audio sample index: {self.current_sample}")
                
                # 使用 blocksize 参数控制音频缓冲区大小，可以提高同步精度
                # 设置一个相对较小的缓冲区大小，以减少延迟
                blocksize = 1024  # 可以根据需要调整
                
                # 获取系统默认设备信息
                try:
                    default_device = sd.default.device
                    print(f"System default audio devices: Input {default_device[0]}, Output {default_device[1]}")
                except Exception as e:
                    print(f"Failed to get default audio device: {e}")
                    default_device = (None, None)
                
                # 列出可用音频设备以供诊断
                available_outputs = []
                try:
                    devices = sd.query_devices()
                    print(f"System available audio devices:")
                    for i, dev in enumerate(devices):
                        is_output = dev['max_output_channels'] > 0
                        print(f"  Device {i}: {dev['name']} ({'Output' if is_output else 'Input'})")
                        if is_output:
                            available_outputs.append(i)
                    print(f"Found {len(available_outputs)} available output devices: {available_outputs}")
                except Exception as e:
                    print(f"Failed to get audio device list: {e}")
                    devices = []
                
                # 创建音频流，尝试使用找到的第一个输出设备
                device_to_use = None
                if available_outputs:
                    # 优先使用pulse设备（通常是设备0）
                    if 0 in available_outputs:
                        device_to_use = 0
                        print(f"Using PulseAudio device (device 0) as audio output device")
                    else:
                        device_to_use = available_outputs[0]
                        print(f"Using device {device_to_use} as audio output device")
                else:
                    print("No available audio devices found, will try using system default...")
                
                # 尝试创建音频流
                try:
                    print(f"Creating audio output stream, using device: {device_to_use}")
                    with sd.OutputStream(
                        samplerate=self.sample_rate,
                        channels=self.audio_data.shape[1],
                        callback=self.audio_callback,
                        blocksize=1024,
                        device=device_to_use
                    ) as stream:
                        print(f"Audio stream created successfully, starting playback, sample rate: {self.sample_rate}Hz, channels: {self.audio_data.shape[1]}")
                        self.audio_stream = stream
                        self.audio_playing = True
                        
                        # 等待直到音频播放停止
                        while self.audio_playing and dpg.is_dearpygui_running():
                            sd.sleep(100)
                except Exception as e:
                    print(f"Failed to create audio stream with specific device: {e}")
                    print("Trying to use 'default' device...")
                    try:
                        with sd.OutputStream(
                            samplerate=self.sample_rate,
                            channels=self.audio_data.shape[1],
                            callback=self.audio_callback,
                            blocksize=1024,
                            device="default"
                        ) as stream:
                            print(f"Audio stream created successfully using 'default' device")
                            self.audio_stream = stream
                            self.audio_playing = True
                            
                            # 等待直到音频播放停止
                            while self.audio_playing and dpg.is_dearpygui_running():
                                sd.sleep(100)
                    except Exception as e2:
                        print(f"Failed to create audio stream with 'default' device: {e2}")
                        
                        # 直接跳转到虚拟音频模式
                        try:
                            print("Trying to use virtual audio output...")
                            # 如果所有输出设备尝试失败，回落到虚拟模式
                            temp_buffer = np.zeros((1000, self.audio_data.shape[1]), dtype=self.audio_data.dtype)
                            self.audio_stream = None  # 不使用实际流
                            self.audio_playing = True
                            
                            # 模拟音频播放线程
                            while self.audio_playing and dpg.is_dearpygui_running():
                                # 更新时间步位置
                                if self.playing:
                                    next_frame = min(self.timestep + 1, self.num_timesteps - 1)
                                    
                                    # 检查是否需要循环
                                    if next_frame == self.num_timesteps - 1 and dpg.get_value("_checkbox_loop"):
                                        next_frame = 0
                                    
                                    # 设置下一帧
                                    if self.timestep != next_frame:
                                        self.next_frame = next_frame
                                        self.need_frame_update = True
                                        self.need_update = True
                                        
                                        # 更新音频位置
                                        self.current_sample = int(next_frame / self.cfg.fps * self.sample_rate)
                                
                                # 模拟适当的播放速度
                                sd.sleep(int(1000 / self.cfg.fps))
                        except Exception as e4:
                            print(f"All audio output methods failed: {e4}")
                            import traceback
                            traceback.print_exc()
                except Exception as e:
                    print(f"Failed to start audio playback: {e}")
                    import traceback
                    traceback.print_exc()
            finally:
                print("Audio playback thread ended")
                self.audio_playing = False
                self.audio_stream = None
        
        # 终止任何现有的音频线程
        if self.audio_thread is not None and self.audio_thread.is_alive():
            print("Stopping existing audio thread")
            self.stop_audio()
            time.sleep(0.1)  # 等待线程完全终止
        
        print("Starting new audio playback thread")
        self.audio_thread = threading.Thread(target=audio_player)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
    def stop_audio(self):
        """停止音频播放"""
        if not self.audio_playing:
            print("Audio is not playing")
            return
            
        # 标记停止音频播放
        print("Stopping audio playback")
        self.audio_playing = False
        
        # 关闭音频流
        if self.audio_stream is not None:
            try:
                print("Aborting audio stream")
                self.audio_stream.abort()
                self.audio_stream = None
            except Exception as e:
                print(f"Error aborting audio stream: {e}")
                
        # 等待音频线程结束
        if self.audio_thread is not None and self.audio_thread.is_alive():
            print("Waiting for audio thread to end")
            timeout = 1.0  # 最多等待1秒
            self.audio_thread.join(timeout)
            if self.audio_thread.is_alive():
                print("Warning: Audio thread did not terminate cleanly")
        
        print("Audio stopped")

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
            dpg.configure_item("_slider_timestep", min_value=0, max_value=0)
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
        dpg.configure_item("_slider_timestep", min_value=0, max_value=self.num_record_timeline-1)

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
            
        record_timestep = dpg.get_value("_slider_timestep")
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
            dpg.set_value("_slider_timestep", i)
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

        # window: rendering options
        with dpg.window(label="Render", tag="_render_window", width=430, height=0, no_resize=True, no_scrollbar=True):
            with dpg.group(horizontal=True):
                dpg.add_text("FPS:")
                dpg.add_text("0   ", tag="_log_fps")
                dpg.add_spacer(width=10)
                dpg.add_text("Frame:")
                dpg.add_text("0", tag="_log_current_frame")
                dpg.add_spacer(width=10)
                dpg.add_text("Render:")
                dpg.add_text("0.0ms", tag="_log_render_time")
                dpg.add_spacer(width=10)
                dpg.add_text("Status:")
                dpg.add_text("就绪", tag="_log_status")

            dpg.add_text(f"点数: {self.gaussians._xyz.shape[0]}")
            
            with dpg.group(horizontal=True):
                # 显示高斯点
                def callback_show_splatting(sender, app_data):
                    self.need_update = True
                dpg.add_checkbox(label="显示高斯点", default_value=True, callback=callback_show_splatting, tag="_checkbox_show_splatting")

                dpg.add_spacer(width=10)

                if self.gaussians.binding is not None:
                    # 显示网格
                    def callback_show_mesh(sender, app_data):
                        self.need_update = True
                    dpg.add_checkbox(label="显示网格", default_value=False, callback=callback_show_mesh, tag="_checkbox_show_mesh")
            
            # 时间步滑块和播放控件
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
                    dpg.set_value("_log_status", f"请求帧 {self.next_frame}")

                with dpg.group(horizontal=True):
                    dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
                    dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)
                    dpg.add_slider_int(label="时间步", tag='_slider_timestep', width=153, min_value=0, 
                                      max_value=self.num_timesteps - 1 if hasattr(self, 'num_timesteps') else 0, 
                                      format="%d", default_value=0, callback=callback_set_current_frame)
                
                # 播放控件
                with dpg.group(horizontal=True):
                    def callback_play_pause(sender, app_data):
                        self.playing = not self.playing
                        if self.playing:
                            dpg.set_item_label("_button_play_pause", "Pause")
                            dpg.set_value("_log_status", "Playing")
                            self.last_frame_time = time.time()
                            # 如果有音频，开始播放
                            if self.audio_data is not None and not self.audio_playing:
                                # 首先尝试使用系统命令播放
                                audio_path = str(self.cfg.audio_path.absolute())
                                system_play_success = self.play_audio_with_system(audio_path)
                                
                                # 如果系统命令播放失败且启用PulseAudio，尝试使用PulseAudio
                                if not system_play_success and self.cfg.use_pulseaudio:
                                    print("系统命令播放失败，尝试使用PulseAudio...")
                                    self.play_audio_with_pulseaudio(audio_path)
                                # 如果系统命令播放失败且未启用PulseAudio，使用内置方法
                                elif not system_play_success:
                                    print("系统命令播放失败，使用内置方法...")
                                    self.start_audio()
                        else:
                            dpg.set_item_label("_button_play_pause", "Play")
                            dpg.set_value("_log_status", "Paused")
                            # 如果音频正在播放，停止
                            if self.audio_playing:
                                self.stop_audio()
                    dpg.add_button(label="Play", tag="_button_play_pause", callback=callback_play_pause)
                    
                    # 中止渲染按钮
                    def callback_abort_render(sender, app_data):
                        self.abort_render = True
                        dpg.set_value("_log_status", "Aborting render...")
                    dpg.add_button(label="Abort Render", tag="_button_abort_render", callback=callback_abort_render)
                    
                    # 循环播放复选框
                    dpg.add_checkbox(label="Loop", default_value=True, tag="_checkbox_loop")
                    
                    # 跳过复杂帧选项
                    def callback_skip_complex(sender, app_data):
                        self.should_skip_complex_frames = app_data
                    dpg.add_checkbox(label="Skip Slow Frames", default_value=self.should_skip_complex_frames, 
                                    callback=callback_skip_complex, tag="_checkbox_skip_complex")
                
                # 音频控件 (如果有音频)
                if self.audio_data is not None:
                    with dpg.group(horizontal=True):
                        dpg.add_text("Audio Position:")
                        dpg.add_text("0.00s", tag="_text_audio_position")
                    
                    # 锁定帧率复选框
                    def callback_lock_frame_rate(sender, app_data):
                        self.cfg.lock_frame_rate = app_data
                        if app_data:
                            dpg.set_value("_log_status", "音频驱动模式: 视频帧将跟随音频位置")
                        else:
                            dpg.set_value("_log_status", "时间驱动模式: 音频将跟随视频帧位置")
                    dpg.add_checkbox(label="音频驱动视频 (Lock Frame Rate)", default_value=self.cfg.lock_frame_rate, 
                                    callback=callback_lock_frame_rate, tag="_checkbox_lock_frame_rate")
                
                # 速度控制
                with dpg.group(horizontal=True):
                    def callback_set_speed(sender, app_data):
                        self.playback_speed = app_data
                    dpg.add_slider_float(label="Speed", min_value=0.1, max_value=2.0, 
                                        format="%.1fx", default_value=1.0, 
                                        callback=callback_set_speed, tag="_slider_speed", width=200)

            # 缩放滑块
            def callback_set_scaling_modifier(sender, app_data):
                self.need_update = True
            dpg.add_slider_float(label="Scale Modifier", min_value=0, max_value=1, format="%.2f", 
                                width=200, default_value=1, callback=callback_set_scaling_modifier, 
                                tag="_slider_scaling_modifier")

            # 视角滑块
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True
            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, width=200, 
                              format="%d deg", default_value=self.cam.fovy, 
                              callback=callback_set_fovy, tag="_slider_fovy")

            # 相机控件
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    dpg.set_value("_slider_fovy", self.cam.fovy)
                    self.need_update = True
                    dpg.set_value("_log_status", "Camera reset")
                dpg.add_button(label="Reset Camera", tag="_button_reset_pose", callback=callback_reset_camera)
                
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

            # API功能 (如果需要)
            if self.gaussians.binding is not None:
                with dpg.group(horizontal=True):
                    def callback_get_flame_from_api(sender, app_data):
                        if self.cfg.audio_path is None:
                            dpg.set_value("_log_status", "Please set audio path first")
                            return
                        
                        dpg.set_value("_log_status", "Getting FLAME from API...")
                        success = self.get_flame_params_from_api(
                            str(self.cfg.audio_path), 
                            subject_style="M003",
                            emotion="neutral",
                            intensity=2
                        )
                        
                        if success:
                            # 重置到起点
                            dpg.set_value("_slider_timestep", 0)
                            # 启动音频播放
                            self.playing = True
                            dpg.set_item_label("_button_play_pause", "Pause")
                            
                            # 首先尝试使用系统命令播放
                            audio_path = str(self.cfg.audio_path.absolute())
                            system_play_success = self.play_audio_with_system(audio_path)
                            
                            # 如果系统命令播放失败且启用PulseAudio，尝试使用PulseAudio
                            if not system_play_success and self.cfg.use_pulseaudio:
                                print("系统命令播放失败，尝试使用PulseAudio...")
                                self.play_audio_with_pulseaudio(audio_path)
                            # 如果系统命令播放失败且未启用PulseAudio，使用内置方法
                            elif not system_play_success:
                                print("系统命令播放失败，使用内置方法...")
                                self.start_audio()
                    
                    dpg.add_button(label="Generate Animation from Audio", callback=callback_get_flame_from_api)

                # 添加文本输入和TTS功能
                with dpg.collapsing_header(label="TTS", default_open=True):
                    with dpg.group(width=400):
                        # 文本输入区域
                        dpg.add_text("Please input the text:")
                        dpg.add_input_text(
                            tag="_input_tts_text",
                            default_value="我们一起努力加油！",
                            multiline=True,
                            width=370,
                            height=80
                        )
                        
                        with dpg.group(horizontal=True):
                            # 文本转语音按钮
                            def callback_text_to_speech(sender, app_data):
                                # 获取输入文本
                                text = dpg.get_value("_input_tts_text")
                                if not text:
                                    dpg.set_value("_log_status", "Please input the text")
                                    return
                                    
                                # 处理文本到语音，然后生成动画
                                self.process_text_to_speech_and_animate(text)
                                
                            dpg.add_button(
                                label="Generate TTS", 
                                tag="_button_generate_tts",
                                callback=callback_text_to_speech
                            )
                            
                            # PulseAudio播放设置
                            def callback_toggle_pulseaudio(sender, app_data):
                                self.cfg.use_pulseaudio = app_data
                                
                            dpg.add_checkbox(
                                label="Use PulseAudio", 
                                default_value=self.cfg.use_pulseaudio,
                                callback=callback_toggle_pulseaudio
                            )
                            
                        # 添加服务器设置
                        with dpg.collapsing_header(label="Server Settings", default_open=False):
                            with dpg.group(width=400):
                                # TTS API URL
                                def callback_set_tts_api_url(sender, app_data):
                                    self.cfg.tts_api_url = app_data
                                
                                dpg.add_input_text(
                                    label="TTS API URL",
                                    default_value=self.cfg.tts_api_url,
                                    callback=callback_set_tts_api_url,
                                    width=370
                                )
                                
                                # PulseAudio服务器
                                def callback_set_pulse_server(sender, app_data):
                                    self.cfg.pulseaudio_server = app_data
                                    # 更新环境变量
                                    os.environ["PULSE_SERVER"] = app_data
                                
                                dpg.add_input_text(
                                    label="PulseAudio Server",
                                    default_value=self.cfg.pulseaudio_server,
                                    callback=callback_set_pulse_server,
                                    width=370
                                )

        # 键盘快捷键
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
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    def update_audio_display(self):
        """更新界面上的音频位置显示"""
        if not self.audio_playing or self.audio_data is None or self.sample_rate is None:
            return
            
        # 获取当前帧的音频位置
        audio_time = self.timestep / self.cfg.fps
        
        # 更新音频位置显示
        if dpg.does_item_exist("_text_audio_position"):
            dpg.set_value("_text_audio_position", f"{audio_time:.2f}s")
            
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
                        # 添加防御性检查
                        if hasattr(self.gaussians, 'binding') and self.gaussians.binding is not None:
                            self.gaussians.select_mesh_by_timestep(self.timestep)
                            dpg.set_value("_log_status", f"Updated to frame {self.timestep}")
                        else:
                            dpg.set_value("_log_status", f"Frame {self.timestep} (no FLAME model)")
                        
                        dpg.set_value("_log_current_frame", f"{self.timestep}")
                        # 更新时间线显示
                        self.update_record_timeline()
                        self.need_frame_update = False
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

    def check_audio_devices(self):
        """检测系统中可用的音频设备并输出诊断信息"""
        print("===== 音频设备诊断 =====")
        
        # 检查sounddevice库的版本
        print(f"sounddevice 版本: {sd.__version__}")
        
        # 检查系统信息
        import platform
        print(f"操作系统: {platform.system()} {platform.release()}")
        
        # 检查默认设备
        try:
            default_device = sd.default.device
            print(f"系统默认音频设备: 输入={default_device[0]}, 输出={default_device[1]}")
            if default_device[1] == -1:
                print("警告: 系统默认输出设备为-1，这可能会导致问题")
        except Exception as e:
            print(f"获取默认音频设备失败: {e}")
        
        # 检查音频输出设备
        available_outputs = []
        try:
            devices = sd.query_devices()
            print(f"系统中发现 {len(devices)} 个音频设备:")
            
            for i, dev in enumerate(devices):
                is_output = dev['max_output_channels'] > 0
                device_type = '输出' if is_output else '输入'
                if is_output:
                    available_outputs.append(i)
                
                # 打印详细设备信息
                print(f"  设备 {i}: {dev['name']} ({device_type})")
                print(f"    通道: 输入={dev['max_input_channels']}, 输出={dev['max_output_channels']}")
                print(f"    默认采样率: {dev['default_samplerate']}Hz")
                print(f"    格式: {dev.get('formats', '未知')}")
                
            if available_outputs:
                print(f"找到 {len(available_outputs)} 个输出设备: {available_outputs}")
            else:
                print("警告: 未找到任何音频输出设备！")
                
            # 如果默认设备无效，但找到了其他输出设备，建议使用第一个
            if default_device[1] == -1 and available_outputs:
                print(f"建议: 由于默认输出设备无效，可以使用设备ID {available_outputs[0]} 作为替代")
                
        except Exception as e:
            print(f"查询音频设备失败: {e}")
            import traceback
            traceback.print_exc()
            
        print("========================")

    def signal_handler(self, sig, frame):
        """处理Ctrl+C，安全关闭程序"""
        print("Exiting safely...")
        self.thread_running = False
        self.stop_audio()
        sys.exit(0)

    @torch.no_grad()
    def run(self):
        print("Running Audio Local Viewer...")
        print(f"Configuration:")
        print(f"- Point cloud path: {self.cfg.point_path}")
        print(f"- Motion path: {self.cfg.motion_path}")
        print(f"- Audio path: {self.cfg.audio_path}")
        print(f"- API URL: {self.cfg.api_url}")
        print(f"- FPS: {self.cfg.fps}")
        # print(f"- Audio offset: {self.cfg.audio_offset}")
        print(f"- Auto mode: {self.auto_mode}")
        
        if self.gaussians.binding is not None:
            print(f"FLAME model loaded:")
            print(f"- Total timesteps: {self.num_timesteps}")
            print(f"- FLAME parameters:")
            for k, v in self.gaussians.flame_param.items():
                print(f"  - {k}: {v.shape}")
        else:
            print("FLAME model not loaded - will use basic rendering mode")
            # Initialize a default number of timesteps if not set
            if not hasattr(self, 'num_timesteps') or self.num_timesteps is None:
                self.num_timesteps = 100  # Default value
                print(f"- Using default timesteps: {self.num_timesteps}")
                dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)
        
        # Initialize audio status
        if self.audio_data is not None and self.sample_rate is not None:
            print(f"Audio loaded: {self.audio_data.shape}, sample rate: {self.sample_rate}Hz")
            # 自动开始音频播放
            if not self.audio_playing:
                print("自动启动音频播放...")
                self.playing = True
                try:
                    dpg.set_item_label("_button_play_pause", "Pause")
                    dpg.set_value("_log_status", "Playing")
                except Exception as e:
                    print(f"更新UI状态失败: {e}")
                self.start_audio()
        else:
            print("No audio loaded or invalid audio data")
            
        # 初始化播放相关变量
        self.playback_speed = 1.0
        self.last_frame_time = time.time()
        self.ui_update_time = time.time()
        
        # 性能分析变量
        timing_stats = {
            "ui_render": 0.0,
            "queue_process": 0.0,
            "playback_logic": 0.0,
            "audio_update": 0.0,
            "total_frame": 0.0,
            "idle": 0.0
        }
        timing_counts = {k: 0 for k in timing_stats}
        last_stats_time = time.time()
        
        try:
            dpg.set_value("_log_current_frame", f"{self.timestep}")
        except:
            print("Warning: Failed to initialize UI frame number")
            
        # 添加状态显示
        if not dpg.does_item_exist("_log_status"):
            with dpg.group(parent="_render_window", horizontal=True):
                dpg.add_text("Status:")
                dpg.add_text("Ready", tag="_log_status")
        
        # 添加渲染时间显示
        if not dpg.does_item_exist("_log_render_time"):
            with dpg.group(parent="_render_window", horizontal=True):
                dpg.add_text("Render time:")
                dpg.add_text("0.0ms", tag="_log_render_time")

        # 启动渲染线程
        if self.render_thread is None or not self.render_thread.is_alive():
            self.render_thread = threading.Thread(target=self.render_frame)
            self.render_thread.daemon = True
            self.render_thread.start()
        
        # 主循环
        ui_freeze_count = 0
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
            
            # 处理渲染队列
            queue_start = time.time()
            latest_render = None
            try:
                while True:
                    render_data = self.render_queue.get(block=False)
                    latest_render = render_data
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error clearing queue: {e}")
            
            # 更新渲染结果
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
            
            # 更新音频相关UI
            audio_start = time.time()
            self.update_audio_display()
            audio_end = time.time()
            timing_stats["audio_update"] += (audio_end - audio_start)
            timing_counts["audio_update"] += 1
            
            # 处理播放逻辑
            playback_start = time.time()
            if self.playing and not self.is_rendering and not self.need_frame_update and not self.need_update:
                speed = getattr(self, 'playback_speed', 1.0)
                fps_target = self.cfg.fps * speed
                time_since_last_frame = current_time - self.last_frame_time
                
                # 决定是否应该播放下一帧
                should_advance_frame = False
                
                if self.cfg.lock_frame_rate and self.audio_playing and self.audio_data is not None:
                    # 当启用帧率锁定时，根据音频位置来决定下一帧
                    # 计算当前音频播放时间（秒）
                    audio_time_seconds = self.current_sample / self.sample_rate if self.sample_rate else 0
                    # 根据音频时间计算当前应该显示的帧
                    target_frame = int(audio_time_seconds * self.cfg.fps)
                    
                    # 如果计算出的目标帧超过当前帧，需要更新
                    if target_frame > self.timestep:
                        should_advance_frame = True
                        next_frame = target_frame  # 直接跳到音频对应的帧
                        # 输出调试信息，帮助理解同步过程
                        if target_frame - self.timestep > 1:
                            print(f"音频同步: 跳帧 {self.timestep} -> {target_frame}")
                    elif target_frame < self.timestep and target_frame > 0:
                        # 音频落后于视频，可以考虑暂时暂停渲染直到音频赶上
                        print(f"音频同步: 等待音频 (音频帧:{target_frame}, 视频帧:{self.timestep})")
                        should_advance_frame = False
                else:
                    # 未启用帧率锁定，使用原有的基于时间的逻辑
                    if time_since_last_frame >= 1.0 / fps_target:
                        self.last_frame_time = current_time
                        next_frame = self.timestep + 1
                        should_advance_frame = True
                
                # 如果决定要前进到下一帧
                if should_advance_frame:
                    if not self.cfg.lock_frame_rate:
                        self.last_frame_time = current_time  # 只在非锁定模式下更新帧时间
                    
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
                                # 如果音频正在播放，停止音频
                                if self.audio_playing:
                                    self.stop_audio()
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
            
            playback_end = time.time()
            timing_stats["playback_logic"] += (playback_end - playback_start)
            timing_counts["playback_logic"] += 1
            
            # 检测潜在的死锁
            if self.is_rendering and (current_time - self.render_start_time > 10.0):
                print(f"检测到潜在死锁，强制重置状态")
                self.is_rendering = False
                self.need_update = False
                self.need_frame_update = False
                self.abort_render = True
            
            # 渲染UI
            ui_start = time.time()
            try:
                dpg.render_dearpygui_frame()
            except Exception as e:
                print(f"渲染UI帧错误: {e}")
                self.is_rendering = False
                self.need_update = False
                self.need_frame_update = False
                time.sleep(0.1)
            
            ui_end = time.time()
            timing_stats["ui_render"] += (ui_end - ui_start)
            timing_counts["ui_render"] += 1
            
            # 空闲时间
            idle_start = time.time()
            #time.sleep(0.005)  # 减轻CPU负担
            idle_end = time.time()
            
            timing_stats["idle"] += (idle_end - idle_start)
            timing_counts["idle"] += 1
            
            frame_end = time.time()
            timing_stats["total_frame"] += (frame_end - frame_start)
            timing_counts["total_frame"] += 1
            
            # 性能统计
            if time.time() - last_stats_time > 5.0 and all(v > 0 for v in timing_counts.values()):
                avg_stats = {k: (timing_stats[k] * 1000 / timing_counts[k]) for k in timing_stats}
                total_time_per_frame = avg_stats["total_frame"]
                effective_fps = 1000 / total_time_per_frame if total_time_per_frame > 0 else 0
                
                print("\n--- 性能统计 (毫秒) ---")
                print(f"队列处理:     {avg_stats['queue_process']:.2f} ms")
                print(f"音频更新:     {avg_stats['audio_update']:.2f} ms")
                print(f"播放逻辑:     {avg_stats['playback_logic']:.2f} ms")
                print(f"UI渲染:       {avg_stats['ui_render']:.2f} ms")
                print(f"空闲时间:     {avg_stats['idle']:.2f} ms")
                print(f"总帧时间:     {avg_stats['total_frame']:.2f} ms")
                print(f"有效FPS:      {effective_fps:.2f}")
                print(f"渲染线程FPS:  {1000 / (self.render_time*1000) if self.render_time else 0:.2f}")
                
                # 重置计数器
                timing_stats = {k: 0.0 for k in timing_stats}
                timing_counts = {k: 0 for k in timing_counts}
                last_stats_time = time.time()

        # 清理
        print("关闭渲染线程...")
        self.thread_running = False
        self.stop_audio()
        if self.render_thread:
            timeout = 2.0
            self.render_thread.join(timeout=timeout)
            if self.render_thread.is_alive():
                print("警告: 渲染线程未正常终止")

    def setup_pulseaudio_env(self):
        """设置PulseAudio环境变量"""
        import os
        import subprocess
        
        print("Setting up PulseAudio environment...")
        
        # 始终设置PulseAudio服务器环境变量
        os.environ["PULSE_SERVER"] = self.cfg.pulseaudio_server
        print(f"PULSE_SERVER set to: {os.environ['PULSE_SERVER']}")
        
        # 尝试列出可用的PulseAudio设备
        try:
            print("Trying to list PulseAudio sinks...")
            result = subprocess.run(
                ["pactl", f"--server={os.environ['PULSE_SERVER']}", "list", "short", "sinks"],
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                print("Available PulseAudio sinks:")
                print(result.stdout)
                
                # 查找waveout设备
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line and 'waveout' in line:
                        parts = line.split('\t')[0]
                        try:
                            sink_id = int(parts)
                            print(f"Found waveout sink with ID: {sink_id}")
                            self.pulse_sink_id = sink_id
                            return sink_id
                        except:
                            print(f"Could not parse sink ID from: {parts}")
            else:
                print(f"pactl command failed with error: {result.stderr}")
        except Exception as e:
            print(f"Error running pactl: {e}")
        
        # 默认返回0
        self.pulse_sink_id = 0
        return 0

    def play_audio_with_pulseaudio(self, audio_path):
        """使用PulseAudio播放音频文件
        
        Args:
            audio_path (str): 音频文件路径
        """
        try:
            print(f"使用PulseAudio播放音频: {audio_path}")
            pulse_server = os.environ.get("PULSE_SERVER", self.cfg.pulseaudio_server)
            
            cmd = ["paplay", f"--server={pulse_server}", str(audio_path)]
            print(f"执行命令: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # 非阻塞方式获取输出
            def monitor_process():
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    print(f"PulseAudio播放错误: {stderr.decode('utf-8', errors='ignore')}")
                else:
                    print("PulseAudio播放完成")
                    
            monitor_thread = threading.Thread(target=monitor_process)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            return True
        except Exception as e:
            print(f"使用PulseAudio播放音频时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_tts_from_api(self, text):
        """从API获取TTS音频
        
        Args:
            text (str): 需要转换为语音的文本
            
        Returns:
            bool: 是否成功获取并保存音频
        """
        try:
            print(f"从API获取TTS音频，文本: {text}")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.cfg.temp_audio_path) if os.path.dirname(self.cfg.temp_audio_path) else ".", exist_ok=True)
            
            # 准备API请求
            api_url = f"{self.cfg.tts_api_url}/api/tts"
            print(f"请求URL: {api_url}")
            
            # 使用multipart/form-data格式发送请求
            files = {'target_text': (None, text)}
            
            # 发送请求
            print("发送TTS请求...")
            response = requests.post(api_url, files=files, stream=True, timeout=30)
            
            if response.status_code != 200:
                print(f"TTS API请求失败: 状态码 {response.status_code}")
                print(f"错误信息: {response.text}")
                return False
                
            # 保存音频文件
            audio_path = self.cfg.temp_audio_path
            print(f"保存TTS音频到: {audio_path}")
            
            with open(audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"TTS音频保存成功: {audio_path}")
            
            # 更新当前音频路径
            self.cfg.audio_path = audio_path
            
            # 加载音频
            self.load_audio()
            
            # 返回成功
            return True
            
        except Exception as e:
            print(f"获取TTS音频时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def process_text_to_speech_and_animate(self, text):
        """处理文本到语音，然后生成FLAME动画
        
        Args:
            text (str): 需要转换为语音的文本
            
        Returns:
            bool: 是否成功完成整个流程
        """
        try:
            # 显示处理状态
            dpg.set_value("_log_status", f"正在处理文本: {text}")
            
            # 第一步：从API获取TTS音频
            print("步骤1: 获取TTS音频...")
            tts_success = self.get_tts_from_api(text)
            
            if not tts_success:
                dpg.set_value("_log_status", "获取TTS音频失败")
                return False
            
            # 第二步：将音频文件路径保存到服务器
            print("步骤2: 准备音频文件路径...")
            audio_path = str(self.cfg.audio_path.absolute())
            print(f"音频文件路径: {audio_path}")
            
            # 第三步：从API获取FLAME参数
            print("步骤3: 获取FLAME参数...")
            dpg.set_value("_log_status", "正在获取FLAME参数...")
            
            # 如果设置了server_audio_path，优先使用它
            target_audio_path = self.cfg.server_audio_path if self.cfg.server_audio_path else audio_path
            
            flame_success = self.get_flame_params_from_api(
                target_audio_path,
                subject_style="M003", 
                emotion="neutral",
                intensity=2
            )
            
            if not flame_success:
                dpg.set_value("_log_status", "获取FLAME参数失败")
                return False
                
            # 第四步：播放音频
            print("步骤4: 播放音频...")
            audio_path = str(self.cfg.audio_path.absolute())
            
            # 首先尝试使用系统命令播放
            system_play_success = self.play_audio_with_system(audio_path)
            
            # 如果系统命令播放失败并且启用了PulseAudio，尝试使用PulseAudio
            if not system_play_success and self.cfg.use_pulseaudio:
                print("系统命令播放失败，尝试使用PulseAudio...")
                self.play_audio_with_pulseaudio(audio_path)
            # 如果系统命令播放失败且未启用PulseAudio，使用内置方法
            elif not system_play_success:
                print("系统命令播放失败，使用内置方法...")
                self.start_audio()
            
            # 重置到起点并开始播放动画
            dpg.set_value("_slider_timestep", 0)
            self.playing = True
            dpg.set_item_label("_button_play_pause", "Pause")
            
            dpg.set_value("_log_status", "处理完成，开始播放")
            return True
            
        except Exception as e:
            print(f"文本到语音动画处理出错: {e}")
            import traceback
            traceback.print_exc()
            dpg.set_value("_log_status", f"处理错误: {str(e)[:30]}...")
            return False

    def play_audio_with_system(self, audio_path):
        """使用系统命令播放音频
        
        Args:
            audio_path (str): 音频文件路径
            
        Returns:
            bool: 是否成功启动播放
        """
        try:
            import subprocess
            import os
            
            audio_path = str(audio_path)
            print(f"使用系统命令播放音频: {audio_path}")
            
            # 尝试不同的播放命令
            commands = []
            
            # 在Linux系统上
            if os.name == 'posix':
                # 首先尝试aplay
                commands.append(["aplay", audio_path])
                # 然后尝试ffplay
                commands.append(["ffplay", "-nodisp", "-autoexit", audio_path])
                # 尝试mpv
                commands.append(["mpv", "--no-video", audio_path])
            # 在Windows系统上
            elif os.name == 'nt':
                commands.append(["powershell", "-c", f"(New-Object Media.SoundPlayer '{audio_path}').PlaySync()"])
            
            # 尝试每一个命令
            for cmd in commands:
                try:
                    print(f"尝试使用命令: {' '.join(cmd)}")
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # 设置超时检查进程是否启动成功
                    time.sleep(0.5)
                    if process.poll() is None:  # 进程仍在运行
                        print(f"成功启动播放进程")
                        
                        # 监控进程的线程
                        def monitor_process():
                            stdout, stderr = process.communicate()
                            if process.returncode != 0:
                                print(f"播放命令错误: {stderr.decode('utf-8', errors='ignore')}")
                            else:
                                print("音频播放完成")
                                
                        monitor_thread = threading.Thread(target=monitor_process)
                        monitor_thread.daemon = True
                        monitor_thread.start()
                        
                        return True
                    else:
                        # 进程已退出
                        stderr = process.stderr.read().decode('utf-8', errors='ignore')
                        print(f"播放命令执行失败: {stderr}")
                except Exception as e:
                    print(f"执行命令时出错: {e}")
                    continue
                    
            print("所有播放命令都失败了")
            return False
            
        except Exception as e:
            print(f"使用系统命令播放音频时出错: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # 尝试初始化声卡系统
    print("正在初始化音频系统...")
    
    # 检查并尝试修复默认设备
    try:
        default_device = sd.default.device
        print(f"当前默认音频设备: 输入={default_device[0]}, 输出={default_device[1]}")
        
        # 如果默认输出设备无效，尝试设置为0
        if default_device[1] == -1:
            print("默认输出设备无效，尝试设置为设备0...")
            try:
                sd.default.device = (default_device[0], 0)
                print(f"已将默认输出设备设置为0")
            except Exception as e:
                print(f"设置默认输出设备失败: {e}")
    except Exception as e:
        print(f"检查默认音频设备失败: {e}")
    
    # 尝试列出可用设备
    try:
        devices = sd.query_devices()
        outputs = []
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                outputs.append(i)
        
        if outputs:
            print(f"找到 {len(outputs)} 个输出设备")
            try:
                default_device = sd.default.device
                if default_device[1] == -1 or default_device[1] not in outputs:
                    best_device = outputs[0]
                    print(f"自动选择输出设备 {best_device}")
                    try:
                        sd.default.device = (default_device[0], best_device)
                        print(f"已设置默认输出设备为 {best_device}")
                    except Exception as e:
                        print(f"设置默认设备失败: {e}")
            except Exception as e:
                print(f"检查默认设备状态失败: {e}")
        else:
            print("未找到有效的输出设备，将启用虚拟音频模式")
    except Exception as e:
        print(f"列出音频设备失败: {e}")
    
    # 最终检查配置
    try:
        print(f"音频系统初始化完成，当前默认设备: {sd.default.device}")
    except Exception as e:
        print(f"获取默认设备信息失败: {e}")
    
    try:
        # 添加额外的命令行参数说明
        parser_help = """
GaussianAvatars - Audio Local Viewer

主要功能:
- 从音频生成FLAME面部动画
- 支持文本到语音转换
- 使用PulseAudio实现远程音频播放

PulseAudio设置:
  --pulseaudio_server PULSEAUDIO_SERVER   PulseAudio服务器地址 (默认: 100.127.107.62:4713)
  --use_pulseaudio / --no-use_pulseaudio  是否使用PulseAudio (默认: 使用)

API设置:
  --api_url API_URL                       FLAME API服务器URL (默认: http://localhost:5001)
  --tts_api_url TTS_API_URL               TTS API服务器URL (默认: http://10.112.208.173:5001)
  --test_api                              启动时测试API连接
  
音频设置:
  --audio_path AUDIO_PATH                 本地音频文件路径
  --server_audio_path SERVER_AUDIO_PATH   服务器上的音频文件路径
  --audio_offset AUDIO_OFFSET             音频偏移量(秒)

示例:
  python audio_local_viewer.py --point_path=avatar.ply --motion_path=motion.npz
  python audio_local_viewer.py --tts_api_url=http://10.112.208.173:5001 --pulseaudio_server=100.127.107.62:4713
"""
        
        # 解析命令行参数
        cfg = tyro.cli(Config, description=parser_help)
        
        print("\n=== 配置信息 ===")
        print(f"- 点云路径: {cfg.point_path}")
        print(f"- 动作路径: {cfg.motion_path}")
        print(f"- 音频路径: {cfg.audio_path}")
        print(f"- FLAME API URL: {cfg.api_url}")
        print(f"- TTS API URL: {cfg.tts_api_url}")
        print(f"- PulseAudio服务器: {cfg.pulseaudio_server}")
        print(f"- 使用PulseAudio: {cfg.use_pulseaudio}")
        print(f"- 服务器音频路径: {cfg.server_audio_path}")
        print(f"- 临时音频路径: {cfg.temp_audio_path}")
        print("================\n")
        
        # 检查必要参数
        if cfg.point_path is None:
            print("警告: 未指定点云路径 (--point_path)，可能会影响渲染功能")
            
        # 设置PulseAudio环境变量
        if cfg.use_pulseaudio:
            os.environ["PULSE_SERVER"] = cfg.pulseaudio_server
            print(f"已设置PULSE_SERVER环境变量为: {cfg.pulseaudio_server}")
        else:
            # 如果不使用PulseAudio，确保环境变量被清除
            if "PULSE_SERVER" in os.environ:
                del os.environ["PULSE_SERVER"]
                print("已清除PULSE_SERVER环境变量")
            print("已禁用PulseAudio，将使用本地音频设备")
        
        # 测试API连接
        if cfg.test_api:
            print("正在测试API连接...")
            try:
                # 测试FLAME API
                flame_api_url = cfg.api_url
                print(f"测试FLAME API: {flame_api_url}")
                try:
                    health_check = requests.get(f"{flame_api_url}/health", timeout=2)
                    if health_check.status_code == 200:
                        print(f"FLAME API服务正常: {health_check.text}")
                    else:
                        print(f"FLAME API服务状态异常: 状态码 {health_check.status_code}")
                except Exception as e:
                    print(f"FLAME API连接失败: {e}")
                
                # 测试TTS API
                tts_api_url = cfg.tts_api_url
                print(f"测试TTS API: {tts_api_url}")
                try:
                    # 简单发送HEAD请求测试连接
                    head_check = requests.head(f"{tts_api_url}/api/tts", timeout=2)
                    if head_check.status_code in [200, 405]:  # 405是Method Not Allowed，但说明服务存在
                        print(f"TTS API服务正常: 状态码 {head_check.status_code}")
                    else:
                        print(f"TTS API服务状态异常: 状态码 {head_check.status_code}")
                except Exception as e:
                    print(f"TTS API连接失败: {e}")
            except Exception as e:
                print(f"API测试过程中出错: {e}")
        
        # 创建并运行查看器
        gui = LocalViewer(cfg)
        gui.run()
    except Exception as e:
        print(f"程序初始化时出错: {e}")
        import traceback
        traceback.print_exc()
