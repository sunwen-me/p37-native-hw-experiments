"""
PyTorch数据集类，用于加载地图和路径数据
"""
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset
from typing import Tuple, Optional
import cv2
import hashlib


def _map_hash(value: np.ndarray, include_shape: bool = False) -> str:
    """Return the canonical hash used by compact native-H/W archives.

    Legacy task archives hashed only the raw map bytes, so callers that are
    validating legacy ``maps`` metadata keep ``include_shape=False``.  New
    ``unique_maps`` archives include H/W to prevent a byte-identical raster
    with a different declared shape from sharing an identity.
    """
    array = np.ascontiguousarray(np.asarray(value), dtype=np.uint8)
    digest = hashlib.sha256()
    if include_shape:
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


class _IndexedMapView:
    """Lazy per-task view over a compact unique-map table."""

    def __init__(self, maps, map_ids, unique_map_hashes=None):
        self._maps = maps
        self._map_ids = np.asarray(map_ids, dtype=np.int64)
        self.map_hashes = (
            None if unique_map_hashes is None
            else np.asarray(unique_map_hashes)[self._map_ids]
        )

    def __len__(self):
        return len(self._map_ids)

    def __getitem__(self, index):
        return self._maps[int(self._map_ids[int(index)])]


class PathPlanningDataset(Dataset):
    """地图条件的路径规划数据集"""

    def __init__(self, data_path: str, horizon: int = 128,
                 normalize: bool = True, augment: bool = False,
                 distance_scale: Optional[float] = None,
                 validate_geometry: bool = True,
                 cache_distance_fields: bool = False):
        """
        Args:
            data_path: npz数据文件路径
            horizon: 固定路径长度（进行插值或截断）
            normalize: 是否归一化坐标到[-1, 1]
            augment: 是否进行数据增强（旋转、翻转）
        """
        self.horizon = horizon
        self.normalize = normalize
        self.augment = augment
        self.validate_geometry = bool(validate_geometry)
        self.cache_distance_fields = bool(cache_distance_fields and not augment)
        if distance_scale is not None and distance_scale <= 0:
            raise ValueError("distance_scale must be positive")
        self.distance_scale = distance_scale

        # 加载数据
        data = np.load(data_path, allow_pickle=True)
        self._validated_geometry_indices = set()
        self._distance_field_cache = {}
        self.geometry = None
        if 'geometry_json' in data:
            from utils.robot_geometry import RobotGeometry
            self.geometry=RobotGeometry.from_json(data['geometry_json'])
        self.map_ids = data['map_ids'] if 'map_ids' in data else None
        if 'unique_maps' in data:
            if self.map_ids is None:
                raise ValueError("unique_maps storage requires one map_id per task")
            self._unique_maps = data['unique_maps']
            if self.map_ids.ndim != 1 or self.map_ids.dtype.kind not in 'iu':
                raise ValueError("map_ids must be a one-dimensional integer array")
            if (len(self.map_ids) == 0 or self.map_ids.min() < 0
                    or self.map_ids.max() >= len(self._unique_maps)):
                raise ValueError("map_ids reference an out-of-range unique map")
            unique_map_hashes = data['unique_map_hashes'] if 'unique_map_hashes' in data else None
            if unique_map_hashes is not None:
                if len(unique_map_hashes) != len(self._unique_maps):
                    raise ValueError("unique_map_hashes must match unique_maps")
                verified_hashes = np.asarray([
                    _map_hash(value, include_shape=True)
                    for value in self._unique_maps
                ])
                if not np.array_equal(verified_hashes, unique_map_hashes):
                    raise ValueError("unique map hash metadata does not match map table")
            self.maps = _IndexedMapView(
                self._unique_maps, self.map_ids, unique_map_hashes,
            )
        elif 'maps' in data:
            self.maps = data['maps']  # [N, H, W] or a ragged per-task map array
        else:
            raise ValueError("dataset archive requires maps or unique_maps")
        self.starts = data['starts']  # [N, 2]
        self.goals = data['goals']  # [N, 2]
        self.paths = data['paths']  # object array, 每个元素是[T, 2]
        self.path_lengths = data['path_lengths']  # [N]
        if len(self.starts) != len(self.goals) or len(self.starts) != len(self.maps):
            raise ValueError("maps, starts, and goals must have the same task count")
        if len(self.paths) != len(self.starts) or len(self.path_lengths) != len(self.starts):
            raise ValueError("paths and path_lengths must have one row per task")
        if self.cache_distance_fields and self.map_ids is None:
            raise ValueError("distance-field caching requires map_ids")

        self.n_samples = len(self.maps)
        self.map_shapes = [tuple(np.asarray(value).shape) for value in self.maps]
        if any(len(shape) != 2 or min(shape) < 4 for shape in self.map_shapes):
            raise ValueError("each map must be a 2D grid with H,W >= 4")
        self.unique_map_shapes = tuple(sorted(set(self.map_shapes)))
        self.map_height, self.map_width = self.map_shapes[0]
        self.map_size = (
            self.map_height if self.map_height == self.map_width else None
        )

        print(f"Loaded dataset: {self.n_samples} samples")
        print(f"Map shapes: {self.unique_map_shapes}")
        print(f"Horizon: {self.horizon}")

    def __len__(self):
        return self.n_samples

    def shape_at(self, idx: int) -> tuple[int, int]:
        return self.map_shapes[int(idx)]

    def resample_path(self, path: np.ndarray, target_length: int) -> np.ndarray:
        """将路径重采样到固定长度"""
        if len(path) == target_length:
            return path

        # 使用线性插值
        original_indices = np.linspace(0, len(path) - 1, len(path))
        target_indices = np.linspace(0, len(path) - 1, target_length)

        resampled_x = np.interp(target_indices, original_indices, path[:, 0])
        resampled_y = np.interp(target_indices, original_indices, path[:, 1])

        return np.stack([resampled_x, resampled_y], axis=1).astype(np.float32)

    def _coordinate_scales(self, height=None, width=None, task_ids=None):
        if task_ids is not None:
            ids = np.asarray(task_ids, dtype=np.int64).reshape(-1)
            if not np.isin(ids, np.arange(self.n_samples)).all():
                raise ValueError("task_ids contain an out-of-range sample")
            shapes = np.asarray([self.map_shapes[int(i)] for i in ids], dtype=np.float32)
            return np.stack([(shapes[:, 1] - 1) / 2,
                             (shapes[:, 0] - 1) / 2], axis=-1)
        if height is None and width is None:
            if len(self.unique_map_shapes) != 1:
                raise ValueError("height/width or task_ids required for mixed-size maps")
            height, width = self.unique_map_shapes[0]
        elif height is None or width is None:
            raise ValueError("height and width must be supplied together")
        height, width = int(height), int(width)
        if min(height, width) < 2:
            raise ValueError("height and width must be at least 2")
        return np.asarray([(width - 1) / 2, (height - 1) / 2], dtype=np.float32)

    def normalize_coordinates(self, coords: np.ndarray, height=None, width=None,
                              task_ids=None) -> np.ndarray:
        """Normalize pixel (x,y) with independent width and height scales."""
        values = np.asarray(coords, dtype=np.float32)
        scale = self._coordinate_scales(height, width, task_ids)
        if task_ids is not None:
            if values.shape[0] != len(scale):
                raise ValueError("one task_id is required for each coordinate batch row")
            scale = scale.reshape((len(scale),) + (1,) * (values.ndim - 2) + (2,))
        return values / scale - 1.0

    def denormalize_coordinates(self, coords: np.ndarray, height=None, width=None,
                                task_ids=None) -> np.ndarray:
        """Convert normalized (x,y) back to per-axis pixel coordinates."""
        values = np.asarray(coords, dtype=np.float32)
        scale = self._coordinate_scales(height, width, task_ids)
        if task_ids is not None:
            if values.shape[0] != len(scale):
                raise ValueError("one task_id is required for each coordinate batch row")
            scale = scale.reshape((len(scale),) + (1,) * (values.ndim - 2) + (2,))
        return (values + 1.0) * scale

    def denormalize_paths(self, coords: np.ndarray, task_ids) -> np.ndarray:
        """Restore exact input endpoints only for exact hard constraints.

        A float32 normalized coordinate may round-trip with >1e-5 pixel
        error on large maps. This restores its known original coordinate,
        without snapping incorrect predicted endpoints or changing the
        offline geometry checker's tolerance.
        """
        values = np.asarray(coords, dtype=np.float32)
        ids = np.asarray(task_ids, dtype=np.int64)
        if values.ndim < 3 or values.shape[-1] != 2 or values.shape[-2] < 2:
            raise ValueError("paths must have shape [B,...,T,2] with T >= 2")
        world = self.denormalize_coordinates(values, task_ids=ids)
        endpoint_shape = (len(ids),) + (1,) * (values.ndim - 3) + (2,)
        for position, original in ((0, self.starts[ids]), (-1, self.goals[ids])):
            normalized = self.normalize_coordinates(original, task_ids=ids).reshape(endpoint_shape)
            fixed = (values[..., position, :] == normalized).all(axis=-1)
            world[..., position, :] = np.where(
                fixed[..., None], original.reshape(endpoint_shape), world[..., position, :],
            )
        return world

    def augment_data(self, map_data: np.ndarray, path: np.ndarray,
                     start: np.ndarray, goal: np.ndarray) -> Tuple:
        """数据增强：随机旋转和翻转"""
        # 随机选择变换
        k = np.random.randint(0, 4)  # 旋转0/90/180/270度
        flip = np.random.rand() > 0.5  # 是否水平翻转

        # Rotate coordinates with the current width; each quarter turn swaps
        # H/W. This matches numpy.rot90 and keeps x=column, y=row.
        height, width = map_data.shape
        def rotate_coords(coords):
            current_height, current_width = height, width
            for _ in range(k):
                x, y = coords[..., 0], coords[..., 1]
                coords = np.stack([y, current_width - 1 - x], axis=-1)
                current_height, current_width = current_width, current_height
            return coords

        path = rotate_coords(path)
        start = rotate_coords(start.reshape(1, 2)).squeeze()
        goal = rotate_coords(goal.reshape(1, 2)).squeeze()
        map_data = np.rot90(map_data, k)

        # 水平翻转
        if flip:
            map_data = np.fliplr(map_data).copy()  # 复制避免负步长
            rotated_width = map_data.shape[1]
            path[..., 0] = rotated_width - 1 - path[..., 0]
            start[0] = rotated_width - 1 - start[0]
            goal[0] = rotated_width - 1 - goal[0]

        return map_data, path, start, goal

    def compute_distance_field(self, map_data: np.ndarray) -> np.ndarray:
        """计算距离场：到最近障碍物的距离"""
        # 将障碍物设为0，自由空间设为1
        free_space = (1 - map_data).astype(np.uint8) * 255

        # The geometry checker treats the map exterior as occupied.  Pad the
        # free-space image with an occupied border so the learned distance
        # feature uses the same boundary convention.
        padded = np.pad(free_space, 1, mode="constant", constant_values=0)
        dist_transform = cv2.distanceTransform(
            padded, cv2.DIST_L2, 5
        )[1:-1, 1:-1]

        # The legacy feature normalized each map independently.  A fixed
        # scale keeps the same physical clearance comparable across maps.
        if self.distance_scale is None:
            if dist_transform.max() > 0:
                dist_transform = dist_transform / dist_transform.max()
        else:
            dist_transform = np.clip(dist_transform / self.distance_scale, 0.0, 1.0)

        return dist_transform.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        """返回一个训练样本"""
        # 加载数据
        map_data = self.maps[idx].copy()
        path = self.paths[idx].copy()
        start = self.starts[idx].copy().astype(np.float32)
        goal = self.goals[idx].copy().astype(np.float32)

        # 重采样路径到固定长度
        if self.geometry is None:
            path = self.resample_path(path, self.horizon)
        else:
            from utils.robot_geometry import resample_safe_polyline
            if len(path) != self.horizon:
                path = resample_safe_polyline(path, self.horizon)

        # 数据增强
        if self.augment:
            map_data, path, start, goal = self.augment_data(map_data, path, start, goal)
            # 确保map_data是连续数组（rot90可能产生负步长）
            map_data = np.ascontiguousarray(map_data)

        # 计算距离场（可选特征）
        if self.cache_distance_fields:
            cache_key = int(self.map_ids[idx])
            distance_field = self._distance_field_cache.get(cache_key)
            if distance_field is None:
                distance_field = self.compute_distance_field(map_data)
                self._distance_field_cache[cache_key] = distance_field
        else:
            distance_field = self.compute_distance_field(map_data)

        if (self.validate_geometry and self.geometry is not None
                and (self.augment or idx not in self._validated_geometry_indices)):
            from utils.robot_geometry import FootprintChecker
            if not np.allclose(path[0],start,rtol=0,atol=1e-5) or not np.allclose(path[-1],goal,rtol=0,atol=1e-5):
                raise ValueError(f"reference endpoint mismatch at sample {idx}")
            if not FootprintChecker(map_data,self.geometry).check_path(path)[0]:
                raise ValueError(f"unsafe reference path at sample {idx} under {self.geometry.version}")
            if not self.augment:self._validated_geometry_indices.add(idx)

        # 归一化坐标
        if self.normalize:
            height, width = map_data.shape
            path = self.normalize_coordinates(path, height, width)
            start = self.normalize_coordinates(start, height, width)
            goal = self.normalize_coordinates(goal, height, width)

        # 转换为torch tensor
        map_tensor = torch.from_numpy(map_data).unsqueeze(0).float()  # [1, H, W]
        distance_tensor = torch.from_numpy(distance_field).unsqueeze(0).float()  # [1, H, W]
        path_tensor = torch.from_numpy(path).float()  # [T, 2]
        start_tensor = torch.from_numpy(start).float()  # [2]
        goal_tensor = torch.from_numpy(goal).float()  # [2]

        return {
            'map': map_tensor,  # 占用栅格
            'distance_field': distance_tensor,  # 距离场
            'path': path_tensor,  # 目标路径
            'start': start_tensor,  # 起点
            'goal': goal_tensor,  # 终点
            'map_raw': torch.from_numpy(np.asarray(self.maps[idx]).copy()).float(),  # 原始地图（用于可视化）
            'spatial_size': torch.tensor(map_data.shape, dtype=torch.long),
        }


def collate_fn(batch):
    """Stack one H/W bucket; never silently mix map geometries."""
    sizes = {tuple(item['map'].shape[-2:]) for item in batch}
    if len(sizes) != 1:
        raise ValueError(
            "collate_fn requires a same-H/W batch; use ShapeBucketBatchSampler"
        )
    return {
        'map': torch.stack([item['map'] for item in batch]),
        'distance_field': torch.stack([item['distance_field'] for item in batch]),
        'path': torch.stack([item['path'] for item in batch]),
        'start': torch.stack([item['start'] for item in batch]),
        'goal': torch.stack([item['goal'] for item in batch]),
        'map_raw': torch.stack([item['map_raw'] for item in batch]),
        'spatial_size': torch.stack([item['spatial_size'] for item in batch]),
    }


class ShapeBucketBatchSampler(Sampler[list[int]]):
    """Yield batches containing maps with the same spatial dimensions."""

    def __init__(self, dataset, batch_size: int, indices=None, shuffle=True,
                 drop_last=False, generator=None, max_pixels_per_batch=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.indices = list(range(len(dataset))) if indices is None else list(map(int, indices))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator
        if max_pixels_per_batch is not None and int(max_pixels_per_batch) < 1:
            raise ValueError("max_pixels_per_batch must be positive when supplied")
        self.max_pixels_per_batch = (
            None if max_pixels_per_batch is None else int(max_pixels_per_batch)
        )

    def _shape_at(self, local_index):
        dataset, index = self.dataset, int(local_index)
        while isinstance(dataset, Subset):
            index = int(dataset.indices[index])
            dataset = dataset.dataset
        if hasattr(dataset, "shape_at"):
            return tuple(dataset.shape_at(index))
        if hasattr(dataset, "map_shapes"):
            return tuple(dataset.map_shapes[index])
        raise TypeError("dataset must expose map_shapes or shape_at(index)")

    def __iter__(self):
        buckets = {}
        for local_index in self.indices:
            buckets.setdefault(self._shape_at(local_index), []).append(local_index)
        batches = []
        for shape, values in buckets.items():
            if self.shuffle:
                order = torch.randperm(len(values), generator=self.generator).tolist()
                values = [values[index] for index in order]
            batch_size = self.batch_size
            if self.max_pixels_per_batch is not None:
                pixels = int(shape[0]) * int(shape[1])
                batch_size = min(batch_size, max(1, self.max_pixels_per_batch // pixels))
            for begin in range(0, len(values), batch_size):
                batch = values[begin:begin + batch_size]
                if len(batch) == batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle and batches:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self):
        buckets = {}
        for local_index in self.indices:
            shape = self._shape_at(local_index)
            buckets[shape] = buckets.get(shape, 0) + 1
        total = 0
        for shape, count in buckets.items():
            batch_size = self.batch_size
            if self.max_pixels_per_batch is not None:
                pixels = int(shape[0]) * int(shape[1])
                batch_size = min(batch_size, max(1, self.max_pixels_per_batch // pixels))
            total += count // batch_size if self.drop_last else (
                count + batch_size - 1
            ) // batch_size
        return total


if __name__ == "__main__":
    # 测试数据集
    dataset = PathPlanningDataset(
        data_path="../data/test_dataset.npz",
        horizon=128,
        normalize=True,
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")

    # 测试加载一个样本
    sample = dataset[0]
    print(f"Map shape: {sample['map'].shape}")
    print(f"Path shape: {sample['path'].shape}")
    print(f"Start: {sample['start']}")
    print(f"Goal: {sample['goal']}")
