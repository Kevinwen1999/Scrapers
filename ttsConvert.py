import edge_tts
import asyncio
import os

# Function to read the text file line by line
def read_text_file_line_by_line(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()

# Asynchronous function to convert text to speech for a line and save it as a temporary file
async def text_to_speech(line_text, output_audio_file):
    # Create an instance of the Communication class
    tts = edge_tts.Communicate(line_text, voice="zh-CN-XiaoxiaoNeural")

    # Generate temporary audio file for the line
    temp_file = output_audio_file.replace(".mp3", "_temp.mp3")
    await tts.save(temp_file)

    return temp_file

# Function to combine multiple audio files into one
def combine_audio_files(output_audio_file, temp_file):
    # Append the temporary audio file to the main output file
    with open(output_audio_file, "ab") as output_file:
        with open(temp_file, "rb") as temp:
            output_file.write(temp.read())
    os.remove(temp_file)  # Remove temporary file after combining

# Main function to generate the complete audio book
def generate_audio_from_text_file(text_file_path, output_audio_file):
    # Remove the output audio file if it exists (start fresh)
    if os.path.exists(output_audio_file):
        os.remove(output_audio_file)

    # Read the text file line by line and convert each line to speech
    for line in read_text_file_line_by_line(text_file_path):
        if line:  # Avoid empty lines
            print(f"Processing line: {line[:30]}...")  # Print first 30 chars of line for progress
            temp_audio_file = asyncio.run(text_to_speech(line, output_audio_file))
            combine_audio_files(output_audio_file, temp_audio_file)

    print(f"Final audio file saved as {output_audio_file}")

# Example usage
if __name__ == "__main__":
    # Path to the .txt file containing the book's content
    text_file_path = "book_225588.txt"  # Replace with your actual file

    # Output audio file (you can change the file format as needed, e.g., .mp3, .wav)
    output_audio_file = "book_audio.mp3"  # You can change the file name and format

    # Generate audio from the text file
    generate_audio_from_text_file(text_file_path, output_audio_file)
