import glob
import os
import zipfile

def zip_npy_json_files(folder_path, output_zip):
    """
    使用 glob 将 folder_path 下的所有 .npy 和 .json 文件打包成 zip，
    保持原有目录结构。
    """
    # 使用递归匹配
    patterns = ["**/*.npy", "**/*.json"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder_path, pattern), recursive=True))

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files:
            rel_path = os.path.relpath(file_path, folder_path)  # 保留相对路径
            zipf.write(file_path, rel_path)
            print(f"Added: {rel_path}")

    print(f"\n✅ 打包完成: {output_zip}")

if __name__ == "__main__":
    folder_path = r"../data"   # 替换为目标文件夹路径
    output_zip = r"output.zip"       # 输出 zip 文件路径
    zip_npy_json_files(folder_path, output_zip)
