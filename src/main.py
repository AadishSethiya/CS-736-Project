import os
import argparse
import time
import json

from utils import *
from deblur import get_method, get_available_methods

DEFAULT_KERNEL_SIZE = 20
DEFAULT_ANGLE = 0
DEFAULT_NOISE_SIGMA = 3
DEFAULT_PAD_SIZE = 30
DEFAULT_OUTPUT_DIR = '../results'
DEFAULT_DATA_DIR = '../data'
DEFAULT_METHOD = 'wiener'
DEFAULT_NSR = 0.01
DEFAULT_RL_ITERATIONS = 50
DEFAULT_TV_LAMBDA = 0.01
DEFAULT_TV_ITERATIONS = 500

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Deblurring Framework')
    parser.add_argument('--image', type=str, default=None, 
                        help='Path to input image. If not provided, all images in data directory are used.')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Directory containing input images (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--kernel_size', type=int, default=DEFAULT_KERNEL_SIZE)
    parser.add_argument('--angle', type=float, default=DEFAULT_ANGLE)
    parser.add_argument('--noise_sigma', type=float, default=DEFAULT_NOISE_SIGMA)
    parser.add_argument('--method', type=str, default=DEFAULT_METHOD, choices=get_available_methods())
    parser.add_argument('--nsr', type=float, default=DEFAULT_NSR)
    parser.add_argument('--pad_size', type=int, default=DEFAULT_PAD_SIZE)
    parser.add_argument('--rl_iterations', type=int, default=DEFAULT_RL_ITERATIONS)
    parser.add_argument('--tv_lambda', type=float, default=DEFAULT_TV_LAMBDA)
    parser.add_argument('--tv_iterations', type=int, default=DEFAULT_TV_ITERATIONS)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    return parser.parse_args()

def get_method_specific_params(args):
    if args.method == 'wiener':
        return {"nsr": args.nsr, "pad_size": args.pad_size}
    elif args.method == 'richardson_lucy':
        return {"iterations": args.rl_iterations}
    elif args.method == 'total_variation':
        return {"lambda_param": args.tv_lambda, "iterations": args.tv_iterations, "pad_size": args.pad_size}
    return {}

def process_image(image_path, args):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(args.output, args.method, image_name)
    os.makedirs(output_dir, exist_ok=True)

    original_image = load_image(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"\n===== Processing {image_name} with {args.method} =====")
    print(f"- Kernel Size: {args.kernel_size}")
    print(f"- Angle: {args.angle}")
    print(f"- Noise Sigma: {args.noise_sigma}")

    method_params = get_method_specific_params(args)
    for param, value in method_params.items():
        print(f"- {param}: {value}")
    print(f"Output Directory: {output_dir}")

    blurred_image, kernel = apply_motion_blur(original_image, args.kernel_size, args.angle)

    kernel_vis_path = os.path.join(output_dir, "psf_visualization.png")
    visualize_kernel(kernel, save_path=kernel_vis_path)

    noisy_blurred_image = add_gaussian_noise(blurred_image, mean=0, sigma=args.noise_sigma)

    deblur_method = get_method(args.method)
    method_name = deblur_method.get_name()

    print(f"Applying {method_name}...")
    start_time = time.time()
    deblurred_image = deblur_method.deblur(noisy_blurred_image, kernel, **method_params)
    execution_time = time.time() - start_time
    print(f"Completed in {execution_time:.2f} seconds")

    metrics = evaluate_deblurring(original_image, noisy_blurred_image, deblurred_image)

    metrics_with_params = {
        "parameters": {
            "method": args.method,
            "kernel_size": args.kernel_size,
            "angle": args.angle,
            "noise_sigma": args.noise_sigma,
            **method_params
        },
        "metrics": metrics,
        "execution_time": execution_time
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(str(metrics_with_params), f, indent=4)

    if not args.no_save:
        comparison_path = os.path.join(output_dir, "comparison.png")
        create_comparison_figure(original_image, noisy_blurred_image, deblurred_image,
                                 method_name, metrics, comparison_path)
        
    print(f"LPIPS: {metrics['blurred']['lpips']:.4f}, Deblurred: {metrics['deblurred']['lpips']:.4f}")
    print(f"Improvement: {metrics['blurred']['lpips'] - metrics['deblurred']['lpips']}")

    if args.display:
        display_results(
            [original_image, noisy_blurred_image, deblurred_image],
            ["Original", "Blurred & Noisy", f"Deblurred ({method_name})"],
            [None,
             {"LPIPS": metrics["blurred"]["lpips"]},
             {"LPIPS": metrics["deblurred"]["lpips"]}]
        )

def main():
    args = parse_arguments()

    if args.image:
        process_image(args.image, args)
    else:
        print(f"\nNo image specified. Processing all images in {args.data_dir}...")
        supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        all_images = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
                      if os.path.splitext(f.lower())[1] in supported_extensions]
        if not all_images:
            print("No images found in the specified data directory.")
            return
        for image_path in all_images:
            process_image(image_path, args)

    print("\n===== All Processing Complete =====")

if __name__ == "__main__":
    main()
