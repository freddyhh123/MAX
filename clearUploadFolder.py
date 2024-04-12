import os
import shutil

def clear_upload_folder(folder_path="upload"):
    """
    Clears all files and directories within upload path.

    Parameters:
    - folder_path (str): The path to the folder that should be cleared. Defaults to "upload".

    Exceptions:
    - Exception: Catches and logs any exception raised during the deletion process.

    Returns:
    - None
    """
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

if __name__ == "__main__":
    clear_upload_folder()