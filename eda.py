# Import the pandas library, the cornerstone of our data analysis toolkit.
import pandas as pd
# Import our visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Define a constant for the path to our dataset file.
CSV_PATH = "features.csv"

# It's a good practice to wrap file operations in a try-except block.
try:
    # Use the pd.read_csv() function to load the dataset.
    features_df = pd.read_csv(CSV_PATH)

    print("DataFrame loaded successfully!")
    # Compute the correlation matrix for all numerical features in the DataFrame.
    print("\\n--- Computing Correlation Matrix ---")
    
    # The .corr() method calculates the pairwise correlation of columns, excluding NA/null values.
    # It returns a new DataFrame representing the matrix.
    correlation_matrix = features_df.corr()
    
    print("Correlation matrix computed successfully.")
    
    # Displaying the top of the correlation matrix to get a feel for it.
    # A full 29x29 matrix is too large to display effectively in the console.
    print("Top 5 rows of the correlation matrix:")
    print(correlation_matrix.head())

    # --- Structural and Statistical Summaries (from previous tasks) ---
    print("\n--- DataFrame Info ---")
    features_df.info()

    print("\n--- DataFrame Statistical Summary ---")
    print(features_df.describe())
    
    print("\n--- Missing Values Check ---")
    if features_df.isnull().sum().sum() == 0:
        print("Conclusion: The dataset has no missing values.")
    else:
        print("Conclusion: The dataset contains missing values.")

    # Visualize the correlation matrix as a heatmap
    print("\n--- Generating Heatmap of Feature Correlations ---")
    
    # Set up the matplotlib figure
    # A larger figure size is needed to make the heatmap readable.
    plt.figure(figsize=(18, 15))

    # Create the heatmap using Seaborn
    # cmap='coolwarm': This is a "diverging" colormap, ideal for correlation matrices.
    #   Positive correlations will be warm (red), negative will be cool (blue),
    #   and correlations near zero will be neutral.
    # annot=False: For a large matrix like this, annotating each cell with its
    #   value would make the plot unreadable. We leave it false.
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)

    # Set the plot title
    plt.title('Correlation Matrix of Music Features', fontsize=20)

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Create a list of genre names in the correct order of their labels (0-9)
    # This order is determined by the alphabetical sorting in feature_extractor.py
    genre_names = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]

    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # Create the count plot
    ax = sns.countplot(x='genre_label', data=features_df, palette='viridis')

    # Set the plot title and labels
    ax.set_title('Distribution of Music Genres in the Dataset', fontsize=16)
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_ylabel('Number of Segments', fontsize=12)

    # Set the x-axis tick labels to be the actual genre names
    # We rotate them slightly for better readability if they overlap
    ax.set_xticklabels(genre_names, rotation=30)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
    # --- END OF THE NEW CODE BLOCK ---

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure you have run the 'feature_extractor.py' script first to generate the dataset.")
except Exception as e:
    print(f"An error occurred while loading the DataFrame: {e}")