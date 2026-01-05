from transformers import ASTModel

# 1. 定义模型名称
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

print(f"正在下载 {model_name} ...")
# 2. 加载模型 (这会自动从 HuggingFace 下载)
model = ASTModel.from_pretrained(model_name)

# 3. 【关键一步】保存到本地文件夹
save_path = "./ast_model_local"
model.save_pretrained(save_path)

print(f"✅ 模型已成功保存在: {save_path}")
print("请将该文件夹上传到 HPC。")