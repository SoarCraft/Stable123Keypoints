"""
CUB数据集背景去除预处理工具
递归处理data/CUB_200_2011/images目录中的所有图像，去除背景
"""

import sys
import re
from pathlib import Path
from typing import Any, Optional
import argparse
from tqdm import tqdm
import PIL.Image
import rembg


def remove_background(image: PIL.Image.Image,
                     rembg_session: Any = None,
                     force: bool = False,
                     **rembg_kwargs,
) -> PIL.Image.Image:
    """
    去除图像背景
    
    Args:
        image: PIL图像对象
        rembg_session: rembg会话对象
        force: 是否强制去除背景
        **rembg_kwargs: rembg其他参数
        
    Returns:
        处理后的PIL图像对象
    """
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def update_images_txt(input_dir: str) -> None:
    """
    更新images.txt文件，将所有.jpg扩展名改为.png
    
    Args:
        input_dir: 输入目录路径
    """
    input_path = Path(input_dir)
    images_txt_path = input_path.parent / 'images.txt'
    
    if not images_txt_path.exists():
        print(f"警告: 找不到images.txt文件: {images_txt_path}")
        return
    
    print(f"更新images.txt文件: {images_txt_path}")
    
    # 读取原文件内容
    with open(images_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 替换.jpg为.png
    updated_lines = []
    changes_count = 0
    
    for line in lines:
        if '.jpg' in line.lower():
            # 替换所有的.jpg为.png（忽略大小写）
            updated_line = re.sub(r'\.jpe?g', '.png', line, flags=re.IGNORECASE)
            updated_lines.append(updated_line)
            if updated_line != line:
                changes_count += 1
        else:
            updated_lines.append(line)
    
    # 写回文件
    with open(images_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"已更新 {changes_count} 行记录，将.jpg扩展名改为.png")


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
                      model_name: str = 'birefnet-general',
                      overwrite: bool = False,
                      force_remove: bool = False,
                      delete_original: bool = True,
                      update_txt: bool = True) -> None:
    """
    处理CUB数据集图像，去除背景
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则覆盖原文件
        model_name: rembg模型名称
        overwrite: 是否覆盖已存在的文件
        force_remove: 是否强制去除背景
        delete_original: 是否删除原始文件（当原始文件不是PNG格式时）
        update_txt: 是否更新images.txt文件中的扩展名
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 设置输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        in_place = False
    else:
        output_path = input_path
        in_place = True
    
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
            
            # 读取图像
            with PIL.Image.open(image_file) as image:
                # 如果不是就地处理且需要保持原始文件，则复制一份
                if not in_place:
                    processed_image = remove_background(
                        image.copy(), 
                        rembg_session=rembg_session, 
                        force=force_remove
                    )
                else:
                    processed_image = remove_background(
                        image, 
                        rembg_session=rembg_session, 
                        force=force_remove
                    )
                
                # 确定是否需要转换格式
                original_is_png = image_file.suffix.lower() == '.png'
                needs_format_conversion = not original_is_png
                
                # 保存处理后的图像
                # 确保以PNG格式保存以支持透明度
                if output_file.suffix.lower() != '.png':
                    output_file = output_file.with_suffix('.png')
                
                processed_image.save(output_file, 'PNG')
                
                # 如果需要删除原始文件且原始文件不是PNG格式
                if delete_original and needs_format_conversion and in_place:
                    try:
                        # 删除原始文件（非PNG格式）
                        if image_file.exists() and image_file != output_file:
                            image_file.unlink()
                    except Exception as e:
                        print(f"删除原始文件 {image_file} 时出错: {str(e)}")

                success_count += 1
                
        except Exception as e:
            print(f"处理文件 {image_file} 时出错: {str(e)}")
            error_count += 1
            continue
    
    # 更新images.txt文件
    if update_txt and in_place and success_count > 0:
        try:
            update_images_txt(input_dir)
        except Exception as e:
            print(f"更新images.txt文件时出错: {str(e)}")
    
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
        default="birefnet-general",
        choices=[
            'u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 
            'silueta', 'isnet-general-use', 'isnet-anime', 'sam',
            'birefnet-general', 'birefnet-general-lite', 'birefnet-portrait',
            'birefnet-dis', 'birefnet-hrsod', 'birefnet-cod', 'birefnet-massive'
        ],
        help="rembg模型名称 (默认: birefnet-general)"
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
    parser.add_argument(
        "--no-delete-original", 
        action="store_true",
        help="不删除原始文件（默认会删除非PNG格式的原始文件）"
    )
    parser.add_argument(
        "--no-update-txt", 
        action="store_true",
        help="不更新images.txt文件中的扩展名（默认会将.jpg改为.png）"
    )
    
    args = parser.parse_args()
    
    try:
        process_cub_images(
            input_dir=args.input_dir,
            output_dir=args.output,
            model_name=args.model,
            overwrite=args.overwrite,
            force_remove=args.force,
            delete_original=not args.no_delete_original,
            update_txt=not args.no_update_txt
        )
    except KeyboardInterrupt:
        print("\n处理已被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
