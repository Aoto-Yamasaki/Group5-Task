import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Set font for English
plt.rcParams["font.family"] = "DejaVu Sans"


class YouTubeDataAnalyzer:
    def __init__(self, metadata_folder):
        self.metadata_folder = metadata_folder
        self.combined_data = None
        self.processed_data = None
        self.pca = None
        self.scaler = None

    def load_all_csv_files(self):
        """Load and combine all CSV files"""
        csv_files = glob.glob(os.path.join(self.metadata_folder, "*.csv"))

        all_dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                # Extract channel name from file name
                channel_name = os.path.basename(file).replace("_metadata.csv", "")
                df["Channel_Name"] = channel_name
                all_dataframes.append(df)
                print(f"Loaded: {channel_name} ({len(df)} videos)")
            except Exception as e:
                print(f"Error: {file} - {e}")

        self.combined_data = pd.concat(all_dataframes, ignore_index=True)
        print(f"\nTotal videos: {len(self.combined_data)}")
        return self.combined_data

    def preprocess_data(self):
        """Preprocess the data"""
        if self.combined_data is None:
            raise ValueError(
                "Combined data is not available. Please call load_all_csv_files() first."
            )
        df = self.combined_data.copy()

        # 1. Process numerical data
        df["View Count"] = pd.to_numeric(df["View Count"], errors="coerce")
        df["Like Count"] = pd.to_numeric(df["Like Count"], errors="coerce")
        df["Comment Count"] = pd.to_numeric(df["Comment Count"], errors="coerce")
        df["Channel Subscriber"] = pd.to_numeric(
            df["Channel Subscriber"], errors="coerce"
        )
        df["Description Length"] = pd.to_numeric(
            df["Description Length"], errors="coerce"
        )

        # 2. Convert duration to seconds
        df["Duration_Seconds"] = df["Duration"].apply(self.parse_duration)

        # 3. Process publication date and time
        df["Published Data & Time"] = pd.to_datetime(
            df["Published Data & Time"], errors="coerce"
        )
        df["Published_Year"] = df["Published Data & Time"].dt.year
        df["Published_Month"] = df["Published Data & Time"].dt.month
        df["Published_DayOfWeek"] = df["Published Data & Time"].dt.dayofweek
        df["Published_Hour"] = df["Published Data & Time"].dt.hour

        # 4. Calculate channel age
        df["Channel Age"] = pd.to_datetime(df["Channel Age"], errors="coerce")
        df["Channel_Age_Days"] = (
            df["Published Data & Time"] - df["Channel Age"]
        ).dt.days

        # 5. Count tags
        df["Tags_Count"] = df["Tags"].apply(self.count_tags)

        # 6. Title length
        df["Title_Length"] = df["Video Title"].str.len()

        # 7. Calculate engagement rates
        df["Like_Rate"] = df["Like Count"] / df["View Count"]
        df["Comment_Rate"] = df["Comment Count"] / df["View Count"]
        df["Engagement_Rate"] = (df["Like Count"] + df["Comment Count"]) / df[
            "View Count"
        ]

        # Replace infinite values and very large values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Cap very large values to prevent overflow
        numeric_columns = [
            "View Count",
            "Like Count",
            "Comment Count",
            "Channel Subscriber",
            "Duration_Seconds",
            "Description Length",
            "Channel_Age_Days",
            "Like_Rate",
            "Comment_Rate",
            "Engagement_Rate",
        ]

        for col in numeric_columns:
            if col in df.columns:
                # Replace values that are too large
                q99 = df[col].quantile(0.99)
                if not pd.isna(q99):
                    df[col] = df[col].clip(
                        upper=q99 * 10
                    )  # Cap at 10x the 99th percentile

        # 8. Create popularity categories based on view count
        df["View_Category"] = pd.qcut(
            df["View Count"], q=3, labels=["Low", "Medium", "High"]
        )

        # Remove missing values and invalid data
        df = df.dropna(subset=["View Count", "Duration_Seconds", "Published_Year"])

        # Remove rows with zero or negative view counts
        df = df[df["View Count"] > 0]

        self.processed_data = df
        print(f"Processed data count: {len(df)} videos")
        return df

    def parse_duration(self, duration_str):
        """Convert duration string to seconds"""
        if pd.isna(duration_str):
            return 0

        # Handle Japanese time format
        duration_str = str(duration_str).strip()
        total_seconds = 0

        try:
            # Clean the string but keep Japanese time units and numbers
            duration_str = re.sub(r"[^\d時間分秒:]+", "", duration_str)

            # Patterns for hours, minutes, and seconds
            if "時間" in duration_str and "分" in duration_str:
                # Format: X時間Y分 or X時間Y分Z秒
                time_match = re.search(r"(\d+)時間(\d+)分(?:(\d+)秒)?", duration_str)
                if time_match:
                    hours = int(time_match.group(1))
                    minutes = int(time_match.group(2))
                    seconds = int(time_match.group(3)) if time_match.group(3) else 0
                    total_seconds = hours * 3600 + minutes * 60 + seconds
            elif "時間" in duration_str:
                # Format: X時間
                time_match = re.search(r"(\d+)時間", duration_str)
                if time_match:
                    hours = int(time_match.group(1))
                    total_seconds = hours * 3600
            elif "分" in duration_str and "秒" in duration_str:
                # Format: X分Y秒
                time_match = re.search(r"(\d+)分(\d+)秒", duration_str)
                if time_match:
                    minutes = int(time_match.group(1))
                    seconds = int(time_match.group(2))
                    total_seconds = minutes * 60 + seconds
            elif "分" in duration_str:
                # Format: X分
                time_match = re.search(r"(\d+)分", duration_str)
                if time_match:
                    minutes = int(time_match.group(1))
                    total_seconds = minutes * 60
            elif "秒" in duration_str:
                # Format: X秒
                time_match = re.search(r"(\d+)秒", duration_str)
                if time_match:
                    seconds = int(time_match.group(1))
                    total_seconds = seconds
            else:
                # Other formats (e.g., 12:34 format)
                if ":" in duration_str:
                    parts = duration_str.split(":")
                    if len(parts) == 2:
                        total_seconds = int(parts[0]) * 60 + int(parts[1])
                    elif len(parts) == 3:
                        total_seconds = (
                            int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        )
                else:
                    # If it's just a number, assume it's seconds
                    if duration_str.isdigit():
                        total_seconds = int(duration_str)

        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse duration '{duration_str}': {e}")
            total_seconds = 0

        return total_seconds

    def count_tags(self, tags_str):
        """Count the number of tags"""
        if pd.isna(tags_str) or tags_str == "[]":
            return 0
        try:
            # Count elements in list-like string
            tags_str = str(tags_str)
            if tags_str.startswith("[") and tags_str.endswith("]"):
                # Split by comma and count (if not empty)
                content = tags_str[1:-1].strip()
                if not content:
                    return 0
                return len([tag.strip() for tag in content.split(",") if tag.strip()])
            return 0
        except:
            return 0

    def perform_pca(self, n_components=0.95):
        """Perform PCA analysis"""
        # Select features for analysis
        feature_columns = [
            "Duration_Seconds",
            "Description Length",
            "Tags_Count",
            "Title_Length",
            "Channel Subscriber",
            "Channel_Age_Days",
            "Total number of channel uploads",
            "Published_Year",
            "Published_Month",
            "Published_DayOfWeek",
            "Published_Hour",
            "Like_Rate",
            "Comment_Rate",
        ]

        # Prepare data
        if self.processed_data is None:
            raise ValueError(
                "Processed data is not available. Please call preprocess_data() first."
            )
        X = self.processed_data[feature_columns].fillna(0)

        # Additional cleaning for PCA
        X = X.replace([np.inf, -np.inf], 0)

        # Remove any remaining NaN values
        X = X.fillna(0)

        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Perform PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)

        print(f"PCA completed: {len(self.pca.components_)} components")
        print(
            f"Cumulative explained variance: {self.pca.explained_variance_ratio_.sum():.3f}"
        )

        return X_pca, X_scaled, feature_columns

    def analyze_with_svm(self, X_pca, target_type="classification"):
        """Analyze with SVM"""
        if target_type == "classification":
            # Classification: Predict view count category
            if self.processed_data is None:
                raise ValueError(
                    "Processed data is not available. Please call preprocess_data() first."
                )
            y = self.processed_data["View_Category"].dropna()
            X_pca_filtered = X_pca[: len(y)]

            # Split train/test data
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca_filtered, y, test_size=0.2, random_state=42, stratify=y
            )

            # SVM classifier
            svm_classifier = SVC(kernel="rbf", random_state=42)
            svm_classifier.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = svm_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print("\n=== SVM Classification Results ===")
            print(f"Accuracy: {accuracy:.3f}")
            print("\nDetailed Report:")
            print(classification_report(y_test, y_pred))

            return svm_classifier, accuracy

        else:
            # Regression: Predict view count directly
            if self.processed_data is None:
                raise ValueError(
                    "Processed data is not available. Please call preprocess_data() first."
                )
            y = np.log1p(
                self.processed_data["View Count"]
            )  # Normalize with log transformation
            X_pca_filtered = X_pca[: len(y)]

            # Split train/test data
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca_filtered, y, test_size=0.2, random_state=42
            )

            # SVM regressor
            svm_regressor = SVR(kernel="rbf")
            svm_regressor.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = svm_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print("\n=== SVM Regression Results ===")
            print(f"Mean Squared Error: {mse:.3f}")
            print(f"R² Score: {r2:.3f}")

            return svm_regressor, r2

    def visualize_results(self, X_pca, feature_columns, output_file):
        """Visualize results"""
        if self.processed_data is None or self.pca is None:
            raise ValueError(
                "Processed data is not available. Please call preprocess_data() first."
            )
        plt.figure(figsize=(15, 12))

        # 1. Visualize PCA components
        plt.subplot(2, 3, 1)
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Cumulative Explained Variance")
        plt.grid(True)

        # 2. Scatter plot of first and second components
        plt.subplot(2, 3, 2)
        views = self.processed_data["View Count"]
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=np.log1p(views), cmap="viridis", alpha=0.6
        )
        plt.colorbar(scatter, label="Log(View Count)")
        plt.xlabel("First Component")
        plt.ylabel("Second Component")
        plt.title("Distribution in Principal Component Space")

        # 3. Feature importance (First Component)
        plt.subplot(2, 3, 3)
        feature_importance = np.abs(self.pca.components_[0])
        indices = np.argsort(feature_importance)[::-1]
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance in First Component")
        plt.xticks(
            range(len(feature_importance)),
            [feature_columns[i] for i in indices],
            rotation=45,
        )

        # 4. View count distribution
        plt.subplot(2, 3, 4)
        plt.hist(np.log1p(self.processed_data["View Count"]), bins=50, alpha=0.7)
        plt.xlabel("Log(View Count)")
        plt.ylabel("Frequency")
        plt.title("View Count Distribution")

        # 5. Channel-wise analysis
        plt.subplot(2, 3, 5)
        channel_stats = (
            self.processed_data.groupby("Channel_Name")["View Count"]
            .mean()
            .sort_values(ascending=False)
        )

        plt.bar(range(len(channel_stats)), np.array(channel_stats.values))
        plt.xlabel("Channel")
        plt.ylabel("Average View Count")
        plt.title("Average View Count by Channel")
        plt.xticks(range(len(channel_stats)), channel_stats.index.tolist(), rotation=45)

        # 6. Relationship between duration and view count
        plt.subplot(2, 3, 6)
        plt.scatter(
            self.processed_data["Duration_Seconds"],
            np.log1p(self.processed_data["View Count"]),
            alpha=0.6,
        )
        plt.xlabel("Duration (Seconds)")
        plt.ylabel("Log(View Count)")
        plt.title("Duration vs View Count")

        plt.tight_layout()
        # plt.show()
        plt.savefig(output_file, dpi=300)

    def generate_insights(self, feature_columns):
        """Generate insights from analysis results"""
        if self.processed_data is None or self.pca is None:
            raise ValueError(
                "Processed data is not available. Please call preprocess_data() first."
            )
        print("\n" + "=" * 50)
        print("YouTube Video View Count Analysis - Insights")
        print("=" * 50)

        # 1. PCA analysis results
        print("\n1. PCA Analysis Results:")
        for i, ratio in enumerate(self.pca.explained_variance_ratio_[:5]):
            print(f"   Contribution of Component {i+1}: {ratio:.3f}")

        # 2. Feature importance
        print("\n2. Important Features (First Component):")
        feature_importance = np.abs(self.pca.components_[0])
        indices = np.argsort(feature_importance)[::-1]
        for i in range(min(5, len(indices))):
            feature_idx = indices[i]
            print(
                f"   {feature_columns[feature_idx]}: {feature_importance[feature_idx]:.3f}"
            )

        # 3. Statistical summary
        print("\n3. Statistical Summary:")
        if self.processed_data is not None:
            print(f"   Total Videos: {len(self.processed_data):,}")
            print(
                f"   Average View Count: {self.processed_data['View Count'].mean():,.0f}"
            )
            print(
                f"   Maximum View Count: {self.processed_data['View Count'].max():,.0f}"
            )
            print(
                f"   Average Video Length: {self.processed_data['Duration_Seconds'].mean()/60:.1f} minutes"
            )
        else:
            print("Processed data is not available.")

        # 4. Top 3 Channels by Average View Count
        print("\n4. Top 3 Channels by Average View Count:")
        if self.processed_data is not None:
            channel_stats = (
                self.processed_data.groupby("Channel_Name")["View Count"]
                .mean()
                .sort_values(ascending=False)
            )
            for i, (channel, views) in enumerate(channel_stats.head(3).items()):
                print(f"   {i+1}. {channel}: {views:,.0f} views")
        else:
            print("Processed data is not available for channel analysis.")


def main():
    # Execute analysis
    analyzer = YouTubeDataAnalyzer(
        "/Users/user/Documents/tmp/assginment/yugoukagakuron/youtube/Group5-Task/metadata"
    )

    print("Starting YouTube video metadata analysis...")

    # Load data
    analyzer.load_all_csv_files()

    # Preprocess data
    analyzer.preprocess_data()

    # Perform PCA
    X_pca, X_scaled, feature_columns = analyzer.perform_pca()

    # SVM analysis (classification)
    svm_classifier, accuracy = analyzer.analyze_with_svm(
        X_pca, target_type="classification"
    )

    # SVM analysis (regression)
    svm_regressor, r2 = analyzer.analyze_with_svm(X_pca, target_type="regression")

    # Visualize results
    analyzer.visualize_results(X_pca, feature_columns, "youtube_analysis_results.png")

    # Generate insights
    analyzer.generate_insights(feature_columns)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
