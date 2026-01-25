import os

def split_file(file_path, save_dir="../knowledge_base/model2", save_name="vae_model.pt", chunk_size_mb=21):
    chunk_size = chunk_size_mb * 1024 * 1024  # 20MB in bytes
    os.makedirs(save_dir, exist_ok=True)
    with open(file_path, 'rb') as f:
        chunk_num = 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_name = f"{save_name}.part{chunk_num}"
            with open(os.path.join(save_dir, chunk_name), 'wb') as chunk_file:
                chunk_file.write(chunk)
            print(f"Created: {chunk_name}")
            chunk_num += 1

# Run it
split_file("/home/xtanghao/THPycharm/dendrites_pfm_vae/generative_ai/results/VAEv12_MDN_time=20260126_004954/ckpt/best.pt")