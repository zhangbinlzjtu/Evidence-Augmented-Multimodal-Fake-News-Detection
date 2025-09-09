import os
import sys
from data_prepare import DataProcessor

def main():
    """
    数据预处理示例运行脚本
    使用方法：
        python run_data_prepare.py 输入文件路径 输出文件路径 [图片基础目录]
    
    例如：
        python run_data_prepare.py ../../data/dataset_items_test.json ../../data/processed_data.json ../../data
    """
    
    if len(sys.argv) < 3:
        print("使用方法: python run_data_prepare.py 输入文件路径 输出文件路径 [图片基础目录]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    img_base_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"开始处理数据...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"图片基础目录: {img_base_dir}")
    
    # 创建处理器并处理数据
    processor = DataProcessor(input_file, output_file, img_base_dir)
    processor.process_data()
    
    print("数据处理完成！")

if __name__ == "__main__":
    main() 