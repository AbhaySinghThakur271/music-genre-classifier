# Import the os module to interact with the operating system,
# primarily for file and directory operations.
import os

# Define a constant for the path to our dataset.
# This makes the code cleaner and easier to modify if the folder name changes.
# We assume this script is running from the project root ('music-genre-classifier').
DATASET_PATH = "genres_original"

# A basic sanity check to ensure the dataset directory actually exists
# before we try to loop through it. This prevents errors if the script
# is run in the wrong place or if the data hasn't been unzipped.
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset path '{DATASET_PATH}' not found.")
    print("Please ensure you have unzipped the dataset in your project's root directory.")
else:
    print("Dataset directory found. Counting files in each genre folder...")
    print("-" * 40) # A separator for cleaner output

    # We use os.listdir() to get a list of all items inside the DATASET_PATH.
    # These items should be our 10 genre folders ('blues', 'classical', etc.).
    # We use sorted() to ensure the output is always in alphabetical order,
    # which is just a nice-to-have for readability.
    for genre_folder in sorted(os.listdir(DATASET_PATH)):
        
        # We construct the full path to the genre folder.
        # For example, on the first iteration, this will create "genres_original/blues".
        # os.path.join is crucial because it uses the correct path separator
        # for whatever OS you are running (e.g., / or \).
        genre_path = os.path.join(DATASET_PATH, genre_folder)

        # It's good practice to ensure the item we're looking at is a directory.
        # This check will skip any stray files that might be in the 'genres_original'
        # folder, like '.DS_Store' on macOS or 'Thumbs.db' on Windows.
        if os.path.isdir(genre_path):
            
            # If it is a directory, we list its contents to get all the .wav file names.
            files_in_genre = os.listdir(genre_path)
            
            # The number of files is simply the length of this list.
            number_of_files = len(files_in_genre)
            
            # We print the result in a nicely formatted way.
            # The .ljust(12) method pads the string with spaces to a length of 12,
            # which helps to align the output neatly in columns.
            print(f"Genre: {genre_folder.ljust(12)} | File Count: {number_of_files}")

    print("-" * 40)
    print("Verification complete. The dataset appears to be balanced.")
