import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import random
import glob
import torch.nn.functional as F

# 필요한 전처리 및 데이터 증강 모듈들을 임포트합니다.
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
from random_erasing import RandomErasing

class VideoClsDatasetFrame(Dataset):
    """프레임 기반의 비디오 분류 데이터셋을 로드하는 클래스"""
    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340,keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,
                 args=None, file_ext='jpg', task='classification'):
        
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.test_num_segment = test_num_segment
        self.test_num_crop = test_num_crop
        self.args = args
        self.file_ext = file_ext

        self.aug = mode == 'train'
        self.rand_erase = self.aug and self.args.reprob > 0

        # 주어진 주석 파일에서 샘플과 레이블을 읽어옵니다.
        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])
        if task != 'classification':  # regression일 경우
            self.label_array = np.array(cleaned.values[:, 1:], dtype=np.float32)
        else:  # classification일 경우
            self.label_array = list(cleaned.values[:, 1])

        # 데이터 전처리 및 변형을 정의합니다.
        if self.aug:  # 학습 모드
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size, short_side_size), interpolation='bilinear'),
            ])
        else:  # 검증/테스트 모드
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size, short_side_size), interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        if self.mode == 'train':
            sample = self.dataset_samples[index]
            buffer = self._load_sample_frames(sample)

            # 프레임을 16개로 interpolate
            buffer = self._interpolate_frames(buffer, self.clip_len)

            if self.args.num_sample > 1:
                frame_list, label_list, index_list = [], [], []
                for _ in range(self.args.num_sample):
                    new_frames = self._aug_frame(buffer)
                    frame_list.append(new_frames)
                    label_list.append(self.label_array[index])
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer)
                return buffer, self.label_array[index], index, {}

        elif self.mode in ['validation', 'test']:
            sample = self.dataset_samples[index]
            buffer = self._load_sample_frames(sample)

            # 프레임을 16개로 interpolate
            buffer = self._interpolate_frames(buffer, self.clip_len)

            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample

    def _load_sample_frames(self, sample):
        """샘플 프레임을 로드하는 함수"""
        buffer = self.load_video(sample)
        while len(buffer) == 0:
            warnings.warn(f"video {sample} not correctly loaded")
            sample = self.dataset_samples[random.randint(0, len(self.dataset_samples)-1)]
            buffer = self.load_video(sample)
        return self.data_resize(buffer) if self.aug else buffer

    def _aug_frame(self, buffer):
        """프레임에 대한 데이터 증강을 수행하는 함수"""
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=self.args.aa,
            interpolation=self.args.train_interpolation,
        )
        buffer = aug_transform(buffer)
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer).permute(0, 2, 3, 1)  # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        buffer = buffer.permute(3, 0, 1, 2)  # C T H W

        buffer = spatial_sampling(
            buffer, min_scale=256, max_scale=320, crop_size=self.crop_size,
            random_horizontal_flip=self.args.data_set != 'SSV2',
            aspect_ratio=[0.75, 1.3333], scale=[0.08, 1.0]
        )
        if self.rand_erase:
            erase_transform = RandomErasing(
                self.args.reprob, mode=self.args.remode,
                max_count=self.args.recount, num_splits=self.args.recount,
                device="cpu"
            )
            buffer = erase_transform(buffer.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        return buffer

    def load_video(self, sample):
        """Decord를 사용해 비디오를 로드하는 함수"""
        if not os.path.exists(sample):
            return []

        if os.path.isfile(sample):  # 이미지 파일인 경우
            with open(sample, "rb") as f:
                img = Image.open(f)
                return [img.convert("RGB")] * self.clip_len

        frames = sorted(glob.glob(os.path.join(sample, f'*.{self.file_ext}')))
        return [self.pil_loader(frame) for frame in frames]

    def pil_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __len__(self):
        return len(self.dataset_samples)

    def _interpolate_frames(self, buffer, target_frame_count):
        """프레임을 target_frame_count로 보간(interpolate)하는 함수"""
        current_frame_count = len(buffer)
        if current_frame_count == target_frame_count:
            return buffer
        elif current_frame_count > target_frame_count:
            # 균등하게 프레임을 선택하여 샘플링
            indices = np.linspace(0, current_frame_count - 1, target_frame_count).astype(int)
            return [buffer[i] for i in indices]
        else:
            # 부족한 프레임을 보간하여 생성
            buffer = torch.stack([transforms.ToTensor()(img) for img in buffer])  # T, C, H, W
            buffer = buffer.unsqueeze(0).permute(0, 2, 1, 3, 4)  # 1, C, T, H, W
            buffer = F.interpolate(buffer, size=(target_frame_count, buffer.shape[-2], buffer.shape[-1]), mode='linear', align_corners=False)
            buffer = buffer.squeeze(0).permute(1, 2, 3, 0)  # T, H, W, C
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            return buffer

def tensor_normalize(tensor, mean, std):
    """텐서를 정규화하는 함수"""
    tensor = tensor.float() / 255.0 if tensor.dtype == torch.uint8 else tensor
    mean, std = torch.tensor(mean), torch.tensor(std)
    return (tensor - mean) / std

def spatial_sampling(frames, min_scale, max_scale, crop_size, random_horizontal_flip, aspect_ratio, scale):
    """주어진 비디오 프레임에 대해 공간 샘플링을 수행"""
    frames, _ = video_transforms.random_resized_crop(
        images=frames, target_height=crop_size, target_width=crop_size,
        scale=scale, ratio=aspect_ratio
    )
    if random_horizontal_flip:
        frames, _ = video_transforms.horizontal_flip(0.5, frames)
    return frames
