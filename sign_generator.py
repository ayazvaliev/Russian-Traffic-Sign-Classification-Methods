from concurrent.futures import ProcessPoolExecutor
import skimage
import numpy as np
import typing
import os
import scipy
from PIL import Image
import json
import shutil
import tqdm
import typing as tp


class SignGenerator(object):

    def __init__(self, background_path: str) -> None:
        super().__init__()
        self.background_path = background_path
        self.bg_points = [(350, 300), (350, 700)]
        self.background_names = os.listdir(background_path)

    @staticmethod
    def resize_icon(icon : np.ndarray, ratio_bounds: tuple[float, float]) -> np.ndarray:
        resize_ratio = np.random.uniform(low=ratio_bounds[0], high=ratio_bounds[1])
        return skimage.transform.resize(icon, output_shape=(icon.shape[0] * resize_ratio, icon.shape[1] * resize_ratio))


    @staticmethod
    def pad_icon(icon : np.ndarray, ratio_bounds: tuple[float, float]) -> np.ndarray:
        pad_ratio = np.random.uniform(low=ratio_bounds[0], high=ratio_bounds[1])
        pad_size = int(icon.shape[0] * pad_ratio)
        return np.pad(icon, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)))


    @staticmethod
    def recolor_icon(icon: np.ndarray, 
                     saturation_bounds: tuple[float, float],
                     brightness_bounds: tuple[float, float]
                     ) -> np.ndarray:
        if icon.shape[-1] < 4:
            return icon
        hsv_icon = skimage.color.rgb2hsv(icon[:, :, :-1])
        saturation_modificator = np.random.uniform(*saturation_bounds)
        value_modificator = np.random.uniform(*brightness_bounds)
        hsv_icon[:, :, 1] = (hsv_icon[:, :, 1] * saturation_modificator).clip(max=1)
        hsv_icon[:, :, 2] = (hsv_icon[:, :, 2] * value_modificator).clip(max=1)
        
        result = np.concatenate((skimage.color.hsv2rgb(hsv_icon), icon[:, :, -1][..., None]),axis=-1)
        assert result.shape == icon.shape
        return result
    
    @staticmethod
    def rotate_icon(icon: np.ndarray, angle_bounds: tuple[int, int]) -> np.ndarray:
        rotation_angle = np.random.randint(*angle_bounds)
        return skimage.transform.rotate(icon, rotation_angle)

    @staticmethod
    def motion_blur_icon(icon: np.ndarray, kernel_size, rotation_bounds: tuple[int, int]) -> np.ndarray:
        kernel_size = 3
        hor_shift_kernel = np.zeros((kernel_size, kernel_size))
        hor_shift_kernel[kernel_size // 2] = np.ones(kernel_size) / kernel_size
        rotation_angle = np.random.randint(*rotation_bounds)
        motion_blur_kernel = skimage.transform.rotate(hor_shift_kernel, rotation_angle, resize=False)
        blurred_channels = []
        for i in range(3):
            blurred_component = scipy.signal.convolve2d(icon[:, :, i], motion_blur_kernel, mode = 'same')
            blurred_channels.append(blurred_component)
        return np.stack(blurred_channels, axis=-1)
    
    @staticmethod
    def gauss_blur_icon(icon: np.ndarray, sigma) -> np.ndarray:
        blurred_rgb = skimage.filters.gaussian(icon[:, :, :-1], sigma=sigma)
        return np.concatenate((blurred_rgb, icon[:, :, -1][..., None]), axis=-1)


    def get_sample(self, icon: np.ndarray, gen_kwargs: dict[str, tp.Any]) -> np.ndarray:

        #initial downsampling of imgage to avoid crop errors
        init_resize_ratio = max(50 / icon.shape[0], 50 / icon.shape[1])
        icon = skimage.transform.resize(icon, output_shape=(int(icon.shape[0] * init_resize_ratio), int(icon.shape[1] * init_resize_ratio)))


        icon = SignGenerator.resize_icon(icon, ratio_bounds=gen_kwargs['resize_ratio_bounds'])
        icon = SignGenerator.pad_icon(icon, ratio_bounds=gen_kwargs['pad_ratio_bounds'])
        icon = SignGenerator.recolor_icon(icon, 
                                          saturation_bounds=gen_kwargs['saturation_bounds'], 
                                          brightness_bounds=gen_kwargs['brightness_bounds'])
        icon = SignGenerator.rotate_icon(icon, angle_bounds=gen_kwargs['angle_bounds'])
        icon = SignGenerator.gauss_blur_icon(icon, sigma=gen_kwargs['gauss_blur_sigma'])

        index = np.random.randint(low=0, high=len(self.background_names))
        pos = np.random.randint(0, len(self.bg_points))
        bg_path = self.background_path + '/' + self.background_names[index]
        bg = np.array(Image.open(bg_path).convert('RGB'))        
        h_coord, w_coord = self.bg_points[pos]
        cropped_bg = skimage.util.img_as_float(bg[h_coord - icon.shape[0] // 2 : h_coord - icon.shape[0] // 2 + icon.shape[0], 
                        w_coord - icon.shape[1] // 2 : w_coord - icon.shape[1] // 2 + icon.shape[1], :])
        
        rgb_icon = icon[:, :, :-1]
        icon = np.where(icon[:, :, -1][..., None] == 0, cropped_bg, rgb_icon)
        
        
        return skimage.util.img_as_ubyte(icon.clip(min=-1,max=1))


def generate_one_icon(args: typing.Tuple[str, str, str, int], gen_kwargs: dict[str, tp.Any]) -> None:
    signs_generated = 0
    try:
        icon_path, out_path, bg_path, q_examples = args
        icon = np.array(Image.open(icon_path))
        sign_generator = SignGenerator(background_path=bg_path)
        new_path = out_path + '/' + icon_path[icon_path.rfind('/') + 1 : icon_path.rfind('.')]
        os.makedirs(new_path)
        for i in range(q_examples):
            new_icon = sign_generator.get_sample(icon, gen_kwargs)
            Image.fromarray(new_icon, mode='RGB').save(new_path + '/' + str(i) + '.jpg')
            signs_generated += 1
    except:
        print(f'unable to generate all synt data for sign={icon_path}, signs generated={signs_generated}')


def generate_rare_data(
    output_folder: str,
    icons_path: str,
    background_path: str,
    gen_kwargs: dict[str, tp.Any],
    samples_per_class: int = 800,
    num_workers: int = 10,
) -> None:
    shutil.rmtree(output_folder, ignore_errors=True)
    with open('classes.json', 'r') as f_json:
        type_data = json.load(f_json)
    with ProcessPoolExecutor(num_workers) as executor:
        params = [
            [
                os.path.join(icons_path, icon_file),
                output_folder,
                background_path,
                samples_per_class,
                gen_kwargs
            ]
            for icon_file in os.listdir(icons_path) if type_data[icon_file[:icon_file.rfind('.')]]['type'] == 'rare'
        ]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))