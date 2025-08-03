import yaml
import argparse
from sign_generator import generate_rare_data

parser = argparse.ArgumentParser(description='Generates synthetic data for rare signs (not present in training dataset)')
parser.add_argument('--output_path', type=str, help='Output directory for synt data', required=True)
parser.add_argument('--config_path', type=str, help='Path to sign generator config file', default='configs/sign_gen.yaml')
parser.add_argument('--bg_path', type=str, help='Path to background images', required=True)
parser.add_argument('--icons_path', type=str, help='Path to icon images', required=True)
parser.add_argument('--samples_per_class', type=int, help='Samples generated per rare class', default=800)
parser.add_argument('--num_workers', type=int, help='Max multiprocessing pool capacity', default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path, 'r') as f_yaml:
        config_data = yaml.safe_load(f_yaml)
    generate_rare_data(
        output_folder=args.output_path,
        icons_path=args.icons_path,
        background_path=args.bg_path,
        gen_kwargs=config_data,
        samples_per_class=args.samples_per_class,
        num_workers=args.num_workers
    )