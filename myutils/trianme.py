import os

import sys
sys.path.insert(0, 'E:\\ultralytics')  # 确保这个路径在环境路径之前
from ultralytics import YOLO
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def main():

    model = YOLO('yolo11n.pt')  # 从官方预训练模型加载

    # --- 2. 定义训练参数 ---
    # data: 指向您的 data.yaml 文件的路径。因为文件在同一目录下，所以直接写文件名即可。
    # epochs: 训练的总轮数。
    # imgsz: 训练过程中输入图像的尺寸。
    # batch: 每批处理的图像数量。-1 表示自动根据您的GPU显存调整，非常方便。
    # project: 训练结果（日志、权重等）保存的根目录。我们将其设置在项目根目录下的 'runs' 文件夹。
    # name: 本次训练的实验名称，结果会保存在 'project/name' 目录下。
    # device: 指定训练设备，0代表第一块GPU，也可以用 'cpu'。
    
    print(f"当前工作目录: {os.getcwd()}")
    print("开始训练YOLOv8模型...")
    
    try:
        results = model.train(
            data='data.yaml',
            epochs=50,
            imgsz=320,
            batch=-1,  # 自动批处理大小
            project='../runs/detect',  # 将结果保存在项目根目录的 runs/detect 文件夹下
            name='face_mask_detection_v2',
            exist_ok=True, # 如果实验文件夹已存在，则覆盖
            device=0  # 使用第一个GPU
        )
        print("训练成功完成！")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")


if __name__ == '__main__':
    main()