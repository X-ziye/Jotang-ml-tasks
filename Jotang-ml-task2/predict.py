import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from task2 import SimpleCNN

def predict_image(image_path, model_path='cnn_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型结构，并加载训练好的权重
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换到评估模式，关闭 Dropout

    # 2. 预处理（必须与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. 读取并处理图片
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device) # (3, 32, 32) -> (1, 3, 32, 32)

    # 4. 推理
    with torch.no_grad():
        output = model(img_tensor).item()
        prob = output if output > 0.5 else 1 - output
        class_name = "狗" if output > 0.5 else "猫"

    print(f"图片路径: {image_path}")
    print(f"预测结果: {class_name}")
    print(f"预测概率 (置信度): {prob:.4f}")

    # 5. 可视化处理后的图片
    # 需要反归一化才能在 matplotlib 正常显示 (从 [-1, 1] 变回 [0, 1])
    img_show = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    plt.imshow(img_show)
    plt.title(f"Prediction: {class_name} ({prob:.2f})")
    plt.axis('off')
    plt.savefig('predict_result.png', dpi=150)
    print("预测结果图已保存为 predict_result.png")

# 测试一下（请确保 test_pet.jpg 在你的文件夹里）
predict_image('test_pet.jpg')