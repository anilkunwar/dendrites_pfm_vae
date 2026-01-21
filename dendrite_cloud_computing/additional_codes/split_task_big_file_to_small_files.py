import os

def split_file(file_path, chunk_size_mb=21):
    chunk_size = chunk_size_mb * 1024 * 1024  # 20MB in bytes
    file_name = os.path.basename(file_path)
    
    with open(file_path, 'rb') as f:
        chunk_num = 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_name = f"{file_name}.part{chunk_num}"
            with open(chunk_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            print(f"Created: {chunk_name}")
            chunk_num += 1

# Run it
split_file("vae_model.pt")
