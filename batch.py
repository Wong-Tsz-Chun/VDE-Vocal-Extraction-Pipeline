import yt_dlp
import os

def download_from_file(file_path):
    # Check if the text file exists
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        return

    # Read URLs from the file
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print("No URLs found in the file.")
        return

    print(f"Found {len(urls)} videos to process...")

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloads/%(title)s.%(ext)s',  # Save to 'downloads' folder
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ignoreerrors': True,  # Skip errors so one bad link doesn't stop the whole batch
        'quiet': False,
        'no_warnings': True,
    }

    # Execute download
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)

if __name__ == "__main__":
    download_from_file("urls.txt")