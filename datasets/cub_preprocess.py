"""
CUB数据集背景去除预处理工具
递归处理data/CUB_200_2011/images目录中的所有图像，去除背景
"""

import torch  # inuse: for CUDA
import sys
import re
from pathlib import Path
from typing import Any, Optional
import argparse
from tqdm import tqdm
import PIL.Image
import numpy as np
import rembg


def remove_background(image: PIL.Image.Image,
                     rembg_session: Any = None,
                     force: bool = False,
                     **rembg_kwargs,
) -> PIL.Image.Image:
    """
    去除图像背景并用灰色填充
    
    Args:
        image: PIL图像对象
        rembg_session: rembg会话对象
        force: 是否强制去除背景
        **rembg_kwargs: rembg其他参数
        
    Returns:
        处理后的PIL图像对象（RGB模式，背景为灰色）
    """
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    
    # 确保图像是RGBA模式
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # 创建灰色背景
    gray_bg = np.full((image.size[1], image.size[0], 3), 127, dtype=np.uint8)
    gray_bg_image = PIL.Image.fromarray(gray_bg, 'RGB')
    
    # 将前景粘贴到灰色背景上
    gray_bg_image.paste(image, (0, 0), image)
    
    return gray_bg_image


def get_image_files(directory: Path) -> list[Path]:
    """
    递归获取目录中的所有JPG图像文件
    
    Args:
        directory: 目录路径
        
    Returns:
        图像文件路径列表
    """
    image_files = []
    
    for file_path in directory.rglob('*.jpg'):
        if file_path.is_file():
            image_files.append(file_path)
    
    return sorted(image_files)


def process_cub_images(input_dir: str, 
                      output_dir: Optional[str] = None,
                      model_name: str = 'isnet-general-use',
                      overwrite: bool = False,
                      force_remove: bool = False) -> None:
    """
    处理CUB数据集图像，去除背景并用灰色填充
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则覆盖原文件
        model_name: rembg模型名称
        overwrite: 是否覆盖已存在的文件
        force_remove: 是否强制去除背景
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 默认就地处理
    in_place = output_dir is None
    output_path = input_path if in_place else Path(output_dir)
    
    if not in_place:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化rembg会话
    print(f"初始化rembg模型: {model_name}")
    rembg_session = rembg.new_session(model_name)
    
    # 获取所有图像文件
    print("扫描图像文件...")
    image_files = get_image_files(input_path)
    print(f"找到 {len(image_files)} 个图像文件")
    
    if not image_files:
        print("未找到任何图像文件")
        return
    
    # 处理图像
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for image_file in tqdm(image_files, desc="处理图像"):
        try:
            # 计算输出路径
            if in_place:
                output_file = image_file
            else:
                relative_path = image_file.relative_to(input_path)
                output_file = output_path / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查是否需要跳过
            if output_file.exists() and not overwrite and not in_place:
                skipped_count += 1
                continue
            
            # 读取并处理图像
            with PIL.Image.open(image_file) as image:
                # 去除背景并填充灰色
                processed_image = remove_background(
                    image, 
                    rembg_session=rembg_session, 
                    force=force_remove
                )
                
                # 保存为JPG格式，直接覆盖原文件
                processed_image.save(output_file, 'JPEG', quality=100)
                
                success_count += 1
                
        except Exception as e:
            print(f"处理文件 {image_file} 时出错: {str(e)}")
            error_count += 1
            continue
    
    # 打印统计信息
    print(f"\n处理完成:")
    print(f"  成功: {success_count}")
    print(f"  错误: {error_count}")
    print(f"  跳过: {skipped_count}")
    print(f"  总计: {len(image_files)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CUB数据集背景去除工具")
    parser.add_argument(
        "input_dir", 
        nargs='?',
        default="data/CUB_200_2011/images",
        help="输入目录路径 (默认: data/CUB_200_2011/images)"
    )
    parser.add_argument(
        "-o", "--output", 
        help="输出目录路径 (如果不指定则就地修改原文件)"
    )
    parser.add_argument(
        "-m", "--model", 
        default="isnet-general-use",
        choices=[
            'u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 
            'silueta', 'isnet-general-use', 'isnet-anime', 'sam',
            'birefnet-general', 'birefnet-general-lite', 'birefnet-portrait',
            'birefnet-dis', 'birefnet-hrsod', 'birefnet-cod', 'birefnet-massive'
        ],
        help="rembg模型名称 (默认: isnet-general-use)"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="覆盖已存在的输出文件"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="强制去除背景，即使图像已有透明通道"
    )

    args = parser.parse_args()
    
    try:
        process_cub_images(
            input_dir=args.input_dir,
            output_dir=args.output,
            model_name=args.model,
            overwrite=args.overwrite,
            force_remove=args.force
        )
    except KeyboardInterrupt:
        print("\n处理已被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
