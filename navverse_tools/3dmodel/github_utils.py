import os
import requests
import subprocess
import shutil
import tempfile
from urllib.parse import urlparse, unquote, quote
import multiprocessing
from functools import partial

def is_lfs_pointer(chunk):
    """Check if the beginning of the file looks like an LFS pointer."""
    if len(chunk) > 300:
        return False
    try:
        text = chunk.decode('utf-8')
        return text.startswith('version https://git-lfs.github.com/spec/v1')
    except UnicodeDecodeError:
        return False

def download_single_file(url, output_dir):
    """
    Downloads a single file from a GitHub blob URL.
    Handles raw downloads, 404s, and LFS pointers.
    Returns (url, success, message) tuple.
    """
    # Parse URL
    try:
        parsed = urlparse(url)
        # Unquote path to handle %20 or real spaces uniformly
        path = unquote(parsed.path)
        path_parts = path.strip('/').split('/')
        
        if len(path_parts) < 4: # User, Repo, blob/raw, ...
            return (url, False, f"Invalid URL format: {url}", None)
        
        # Determine if it's blob
        if parsed.netloc == "github.com" and "blob" in path_parts:
            blob_idx = path_parts.index("blob")
            user = path_parts[0]
            repo = path_parts[1]
            commit = path_parts[blob_idx + 1]
            # file_path might contain multiple segments
            file_path_segments = path_parts[blob_idx + 2:]
            file_path = '/'.join(file_path_segments)
            filename = path_parts[-1]
            
            # Construct Raw URL
            # Encode each segment of the file path to handle spaces etc.
            encoded_file_path = '/'.join([quote(s) for s in file_path_segments])
            raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{commit}/{encoded_file_path}"
        else:
            return (url, False, f"Unsupported URL format: {url}", None)

        save_path = os.path.join(output_dir, filename)
        if os.path.exists(save_path):
            return (url, True, f"Skipping existing file: {filename}", save_path)

        # Check 404 and content type via HEAD
        try:
            head = requests.head(raw_url, timeout=10)
            if head.status_code == 404:
                return (url, False, f"File not found (404): {filename}", None)
            if head.status_code != 200:
                return (url, False, f"Head request failed with {head.status_code}: {filename}", None)
        except requests.RequestException as e:
            return (url, False, f"Connection error checking {filename}: {e}", None)

        # Attempt download
        try:
            resp = requests.get(raw_url, stream=True, timeout=30)
            resp.raise_for_status()

            # Read first chunk to check for LFS
            first_chunk = next(resp.iter_content(chunk_size=512), b"")
            
            if is_lfs_pointer(first_chunk):
                print(f"LFS pointer detected for {filename}. Attempting alternate download methods.")
                resp.close()
                
                # Method 1: Try media.githubusercontent.com
                media_url = f"https://media.githubusercontent.com/media/{user}/{repo}/{commit}/{encoded_file_path}"
                try:
                    m_resp = requests.get(media_url, stream=True, timeout=30)
                    if m_resp.status_code == 200:
                        with open(save_path, 'wb') as f:
                            shutil.copyfileobj(m_resp.raw, f)
                        return (url, True, f"Downloaded {filename} via media URL.", save_path)
                except Exception as e:
                    print(f"Media URL failed for {filename}: {e}")

                # Method 2: Git commands (Sparse checkout / Fetch single file)
                print(f"Attempting git fetch for {filename}...")
                repo_url = f"https://github.com/{user}/{repo}.git"
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        # git init
                        subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
                        subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=temp_dir, check=True, capture_output=True)
                        
                        # Fetch specific commit with depth 1
                        subprocess.run(["git", "fetch", "--depth", "1", "origin", commit], cwd=temp_dir, check=True, capture_output=True)
                        
                        # Checkout file
                        subprocess.run(["git", "checkout", commit, "--", file_path], cwd=temp_dir, check=True, capture_output=True)
                        
                        downloaded_file = os.path.join(temp_dir, file_path)
                        if os.path.exists(downloaded_file):
                            # Verify if it's still a pointer
                            with open(downloaded_file, 'rb') as f:
                                check_content = f.read(512)
                            
                            if is_lfs_pointer(check_content):
                                return (url, False, f"Failed to resolve LFS object via git (git-lfs missing?): {filename}", None)
                            else:
                                shutil.copy2(downloaded_file, save_path)
                                return (url, True, f"Downloaded {filename} via git.", save_path)
                        else:
                            return (url, False, f"Git checkout failed to produce file: {filename}", None)
                    except subprocess.CalledProcessError as e:
                        return (url, False, f"Git command failed for {filename}: {e}", None)
            else:
                # Not LFS, write rest of file
                with open(save_path, 'wb') as f:
                    f.write(first_chunk)
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return (url, True, f"Downloaded {filename} directly.", save_path)

        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)
            return (url, False, f"Error downloading content for {filename}: {e}", None)

    except Exception as e:
        return (url, False, f"Unexpected error for {url}: {e}", None)

def download_batch(urls, download_dir, num_processes=8):
    """
    Downloads a list of URLs in parallel.
    Returns a dictionary of {url: (success, message, local_path)}.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    func = partial(download_single_file, output_dir=download_dir)
    
    results = {}
    print(f"Downloading {len(urls)} files with {num_processes} processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use imap to get results as they complete
        iterator = pool.imap_unordered(func, urls)
        
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(urls), desc="Downloading")
        except ImportError:
            pass
            
        for url, success, message, local_path in iterator:
            results[url] = (success, message, local_path)
            if not success:
                # Print errors immediately, skip success messages to keep output clean
                print(f"\n[FAIL] {message}")
            # else:
            #     print(f"[OK] {message}")
            
    return results
