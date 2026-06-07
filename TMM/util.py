import numpy as np
from scipy.ndimage import map_coordinates
import cv2


# Based on https://github.com/sunset1995/py360convert
class Equirec2Cube:
    def __init__(self, equ_h, equ_w, face_w):
        '''
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        face_w: int, the length of each face of the cubemap
        '''

        self.equ_h = equ_h
        self.equ_w = equ_w
        self.face_w = face_w

        self._xyzcube()
        self._xyz2coor()

        # For convert R-distance to Z-depth for CubeMaps
        cosmap = 1 / np.sqrt((2 * self.grid[..., 0]) ** 2 + (2 * self.grid[..., 1]) ** 2 + 1)
        self.cosmaps = np.concatenate(6 * [cosmap], axis=1)[..., np.newaxis]

    def _xyzcube(self):
        '''
        Compute the xyz cordinates of the unit cube in [F R B L U D] format.
        '''
        self.xyz = np.zeros((self.face_w, self.face_w * 6, 3), np.float32)
        rng = np.linspace(-0.5, 0.5, num=self.face_w, dtype=np.float32)
        self.grid = np.stack(np.meshgrid(rng, -rng), -1)

        # Front face (z = 0.5)
        self.xyz[:, 0 * self.face_w:1 * self.face_w, [0, 1]] = self.grid
        self.xyz[:, 0 * self.face_w:1 * self.face_w, 2] = 0.5

        # Right face (x = 0.5)
        self.xyz[:, 1 * self.face_w:2 * self.face_w, [2, 1]] = self.grid[:, ::-1]
        self.xyz[:, 1 * self.face_w:2 * self.face_w, 0] = 0.5

        # Back face (z = -0.5)
        self.xyz[:, 2 * self.face_w:3 * self.face_w, [0, 1]] = self.grid[:, ::-1]
        self.xyz[:, 2 * self.face_w:3 * self.face_w, 2] = -0.5

        # Left face (x = -0.5)
        self.xyz[:, 3 * self.face_w:4 * self.face_w, [2, 1]] = self.grid
        self.xyz[:, 3 * self.face_w:4 * self.face_w, 0] = -0.5

        # Up face (y = 0.5)
        self.xyz[:, 4 * self.face_w:5 * self.face_w, [0, 2]] = self.grid[::-1, :]
        self.xyz[:, 4 * self.face_w:5 * self.face_w, 1] = 0.5

        # Down face (y = -0.5)
        self.xyz[:, 5 * self.face_w:6 * self.face_w, [0, 2]] = self.grid
        self.xyz[:, 5 * self.face_w:6 * self.face_w, 1] = -0.5

    def _xyz2coor(self):

        # x, y, z to longitude and latitude
        x, y, z = np.split(self.xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        lat = np.arctan2(y, c)

        # longitude and latitude to equirectangular coordinate
        self.coor_x = (lon / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=0):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, 1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, 1)
        e_img = np.concatenate([e_img, pad_d, pad_u], 0)
        # pad_l = e_img[:, [0]]
        # pad_r = e_img[:, [-1]]
        # e_img = np.concatenate([e_img, pad_l, pad_r], 1)

        return map_coordinates(e_img, [self.coor_y, self.coor_x],
                               order=order, mode='wrap')[..., 0]

    def run(self, equ_img, equ_dep=None):
        h, w = equ_img.shape[:2]
        if h != self.equ_h or w != self.equ_w:
            equ_img_np = equ_img.detach().cpu().numpy()

            # 处理 CHW 转 HWC 格式（支持1/3/4通道）
            if equ_img_np.ndim == 3 and equ_img_np.shape[0] in (1, 3, 4):  # 修改条件包含通道数1
                equ_img_np = equ_img_np.transpose(1, 2, 0)
                
                # 仅当通道数为3时做RGB到BGR转换（单通道不需要）
                if equ_img_np.shape[2] == 3:
                    equ_img_np = cv2.cvtColor(equ_img_np, cv2.COLOR_RGB2BGR)
            
            # 执行resize
            equ_img_resized = cv2.resize(equ_img_np, (self.equ_w, self.equ_h))
            
            # 确保至少是三维数组（当输入为单通道时）
            if equ_img_resized.ndim == 2:
                equ_img_resized = equ_img_resized[..., np.newaxis]

            if equ_dep is not None:
                equ_dep = cv2.resize(equ_dep, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        # 处理立方体贴图采样
        cube_img = np.stack([
            self.sample_equirec(equ_img_resized[..., i], order=1) 
            for i in range(equ_img_resized.shape[2])
        ], axis=-1)
        
        # 转回CHW格式（兼容单通道）
        cube_img = cube_img.transpose(2, 0, 1)  # HWC -> CHW

        # 处理深度图
        if equ_dep is not None:
            cube_dep = np.stack([
                self.sample_equirec(equ_dep[..., i], order=0)
                for i in range(equ_dep.shape[2])
            ], axis=-1)
            cube_dep = cube_dep * self.cosmaps

        return (cube_img, cube_dep) if equ_dep is not None else cube_img