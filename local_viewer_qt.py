import sys
import time
import json
import math
import threading
import queue
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
import tyro

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from utils.viewer_utils import Mini3DViewerConfig, Mini3DViewer, OrbitCamera
from gaussian_renderer import GaussianModel, FlameGaussianModel, render
from mesh_renderer import NVDiffRenderer

# -------------------- Configuration Classes --------------------
@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False

@dataclass
class Config(Mini3DViewerConfig):
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    cam_convention: Literal["opengl", "opencv"] = "opencv"
    point_path: Optional[Path] = None
    motion_path: Optional[Path] = None
    sh_degree: int = 3
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    save_folder: Path = Path("./viewer_output")
    fps: int = 30
    demo_mode: bool = False

# -------------------- Rendering Thread --------------------
class RenderThread(QThread):
    image_ready = pyqtSignal(object)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.running = True

    def run(self):
        while self.running:
            if self.viewer.need_frame_update and not self.viewer.is_rendering:
                try:
                    self.viewer.timestep = self.viewer.next_frame
                    self.viewer.gaussians.select_mesh_by_timestep(self.viewer.timestep)
                    self.viewer.frame_label.setText(str(self.viewer.timestep))
                except Exception:
                    pass
                finally:
                    self.viewer.need_frame_update = False

            if self.viewer.need_update and not self.viewer.is_rendering:
                self.viewer.is_rendering = True
                self.viewer.abort_render = False
                try:
                    show_splat = self.viewer.cb_splat.isChecked()
                    show_mesh = self.viewer.cb_mesh.isChecked() if self.viewer.gaussians.binding is not None else False
                    scale_mod = self.viewer.slider_scale.value() / 100.0
                    cam = self.viewer.prepare_camera()
                    rgb = torch.ones([self.viewer.H, self.viewer.W, 3])

                    if show_splat and not self.viewer.abort_render:
                        out = render(
                            cam, self.viewer.gaussians, self.viewer.cfg.pipeline,
                            torch.tensor(self.viewer.cfg.background_color).cuda(),
                            scaling_modifier=scale_mod
                        )
                        rgb_splat = out["render"].permute(1, 2, 0).contiguous()
                    if show_mesh and not self.viewer.abort_render and self.viewer.gaussians.binding is not None:
                        mesh_out = self.viewer.mesh_renderer.render_from_camera(
                            self.viewer.gaussians.verts, self.viewer.gaussians.faces,
                            cam, face_colors=self.viewer.face_colors
                        )
                        rgba = mesh_out["rgba"].squeeze(0)
                        rgb_mesh = rgba[:, :, :3]
                        alpha = rgba[:, :, 3:4]
                        opacity = self.viewer.mesh_color[3:].cuda()

                    if not self.viewer.abort_render:
                        if show_splat and show_mesh:
                            rgb = rgb_mesh * alpha * opacity + rgb_splat * (alpha * (1 - opacity) + (1 - alpha))
                        elif show_splat:
                            rgb = rgb_splat
                        elif show_mesh:
                            rgb = rgb_mesh

                    if not self.viewer.abort_render:
                        arr = rgb.detach().cpu().numpy()
                        if arr.shape[0] == self.viewer.H and arr.shape[1] == self.viewer.W:
                            self.image_ready.emit(arr)
                            self.viewer.need_update = False
                except Exception:
                    pass
                finally:
                    self.viewer.is_rendering = False
                    self.viewer.abort_render = False
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False
        self.wait(2000)

# -------------------- Main Viewer --------------------
class LocalViewer(QMainWindow):
    def __init__(self, cfg: Config):
        # 只初始化 Qt 主窗口
        QMainWindow.__init__(self, parent=None)
        # 设置窗口标题
        self.setWindowTitle('GaussianAvatars - Local Viewer')
        self.cfg = cfg
        
        # 从 Mini3DViewer 借用相机设置
        self.W = cfg.W
        self.H = cfg.H
        self.cam = OrbitCamera(self.W, self.H, r=cfg.radius, fovy=cfg.fovy, convention=cfg.cam_convention)

        # State
        self.playing = False
        self.need_update = True
        self.need_frame_update = False
        self.next_frame = 0
        self.timestep = 0
        self.is_rendering = False
        self.abort_render = False
        self.face_colors = None

        print("Initializing 3D Gaussians...")
        self.init_gaussians()
        if self.gaussians.binding is not None:
            print("Initializing mesh renderer...")
            self.mesh_renderer = NVDiffRenderer(use_opengl=False)
            self.mesh_color = torch.tensor([1.0, 1.0, 1.0, 0.5])
            self.num_timesteps = self.gaussians.num_timesteps
        else:
            self.num_timesteps = 0

        # Prepare UI
        self.setup_ui()

        # Rendering thread
        self.render_thread = RenderThread(self)
        self.render_thread.image_ready.connect(self.on_image_ready)

        # Playback timer
        self.last_frame_time = time.time()
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.playback)
        self.play_timer.start(int(1000 / self.cfg.fps))

        # Handle Ctrl+C
        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigint(self, sig, frame):
        self.close()

    def init_gaussians(self):
        if self.cfg.point_path and (Path(self.cfg.point_path).parent / "flame_param.npz").exists():
            self.gaussians = FlameGaussianModel(self.cfg.sh_degree)
        else:
            self.gaussians = GaussianModel(self.cfg.sh_degree)
        if self.cfg.point_path:
            if Path(self.cfg.point_path).exists():
                self.gaussians.load_ply(
                    Path(self.cfg.point_path), has_target=False,
                    motion_path=self.cfg.motion_path, disable_fid=[]
                )
            else:
                raise FileNotFoundError(f"{self.cfg.point_path} does not exist.")

    def setup_ui(self):
        self.setWindowTitle('Gaussian Avatars Viewer')
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Frame label
        h0 = QHBoxLayout()
        h0.addWidget(QLabel("Frame:"))
        self.frame_label = QLabel("0")
        h0.addWidget(self.frame_label)
        layout.addLayout(h0)

        # Number of points
        layout.addWidget(QLabel(f"number of points: {self.gaussians._xyz.shape[0]}"))

        # Checkboxes for splatting/mesh
        h1 = QHBoxLayout()
        self.cb_splat = QCheckBox("show splatting")
        self.cb_splat.setChecked(True)
        self.cb_splat.stateChanged.connect(lambda _: self.trigger_update())
        h1.addWidget(self.cb_splat)
        if self.gaussians.binding is not None:
            self.cb_mesh = QCheckBox("show mesh")
            self.cb_mesh.setChecked(False)
            self.cb_mesh.stateChanged.connect(lambda _: self.trigger_update())
            h1.addWidget(self.cb_mesh)
        layout.addLayout(h1)

        # Slider and buttons
        h2 = QHBoxLayout()
        self.btn_prev = QPushButton("-")
        self.btn_prev.clicked.connect(self.prev_frame)
        h2.addWidget(self.btn_prev)
        self.btn_next = QPushButton("+")
        self.btn_next.clicked.connect(self.next_frame_slot)
        h2.addWidget(self.btn_next)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.num_timesteps - 1))
        self.slider.valueChanged.connect(self.slider_changed)
        h2.addWidget(self.slider)
        layout.addLayout(h2)

        # Play controls
        h3 = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.play_pause)
        h3.addWidget(self.btn_play)
        self.cb_loop = QCheckBox("Loop")
        self.cb_loop.setChecked(True)
        h3.addWidget(self.cb_loop)
        layout.addLayout(h3)

        # Scale slider
        layout.addWidget(QLabel("Scale modifier"))
        self.slider_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_scale.setMinimum(0)
        self.slider_scale.setMaximum(100)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(lambda _: self.trigger_update())
        layout.addWidget(self.slider_scale)

        # Camera & Save buttons
        h4 = QHBoxLayout()
        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(self.reset_camera)
        h4.addWidget(btn_reset)
        btn_save = QPushButton("Save Image")
        btn_save.clicked.connect(self.save_image)
        h4.addWidget(btn_save)
        layout.addLayout(h4)

        # Image display
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.cam.image_width, self.cam.image_height)
        layout.addWidget(self.image_label)

    def trigger_update(self):
        self.need_update = True

    def prev_frame(self):
        self.next_frame = max(self.timestep - 1, 0)
        self.slider.setValue(self.next_frame)
        self.frame_label.setText(str(self.next_frame))
        self.need_frame_update = True
        self.need_update = True

    def next_frame_slot(self):
        self.next_frame = min(self.timestep + 1, self.num_timesteps - 1)
        self.slider.setValue(self.next_frame)
        self.frame_label.setText(str(self.next_frame))
        self.need_frame_update = True
        self.need_update = True

    def slider_changed(self, val):
        self.next_frame = val
        self.frame_label.setText(str(val))
        self.need_frame_update = True
        self.need_update = True

    def play_pause(self):
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")

    def playback(self):
        if self.playing and not self.is_rendering:
            now = time.time()
            if now - self.last_frame_time >= 1.0 / self.cfg.fps:
                self.last_frame_time = now
                nf = self.timestep + 1
                if nf >= self.num_timesteps:
                    if self.cb_loop.isChecked():
                        nf = 0
                    else:
                        self.playing = False
                        self.btn_play.setText("Play")
                        return
                self.next_frame = nf
                self.slider.setValue(nf)
                self.frame_label.setText(str(nf))
                self.need_frame_update = True
                self.need_update = True

    def reset_camera(self):
        self.cam.reset()
        self.need_update = True

    def save_image(self):
        if not self.cfg.save_folder.exists():
            self.cfg.save_folder.mkdir(parents=True)
        fname = self.cfg.save_folder / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
        if hasattr(self, 'current_image'):
            Image.fromarray(self.current_image).save(fname)

    def on_image_ready(self, arr):
        print(f"▶︎ 主线程收到图像: shape={arr.shape}")
        self.current_image = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        h, w, _ = self.current_image.shape
        img = QImage(self.current_image.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

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

    def closeEvent(self, event):
        self.render_thread.stop()
        event.accept()

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    cfg = tyro.cli(Config)
    viewer = LocalViewer(cfg)
    viewer.show()
    QTimer.singleShot(0, lambda: (viewer.trigger_update(), viewer.render_thread.start()))
    sys.exit(app.exec())