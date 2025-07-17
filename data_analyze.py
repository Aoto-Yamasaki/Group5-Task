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

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay



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

        # 8. Calculate popularity score (view count normalized by channel subscribers)
        # df["Popularity_Score"] = df["View Count"] / df["Channel Subscriber"]
        df["Popularity_Score"] = df["View Count"]

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
            "Popularity_Score",
        ]

        for col in numeric_columns:
            if col in df.columns:
                # Replace values that are too large
                q99 = df[col].quantile(0.99)
                if not pd.isna(q99):
                    df[col] = df[col].clip(
                        upper=q99 * 10
                    )  # Cap at 10x the 99th percentile

        # 9. Create popularity categories based on popularity score (view count / channel subscribers)
        df["Popularity_Category"] = pd.qcut(
            df["Popularity_Score"], q=3, labels=["Low", "Medium", "High"]
        )

        # Remove missing values and invalid data
        df = df.dropna(
            subset=[
                "View Count",
                "Duration_Seconds",
                "Published_Year",
                "Popularity_Score",
                "Channel Subscriber",
            ]
        )

        # Remove rows with zero or negative view counts and zero subscribers
        df = df[(df["View Count"] > 0) & (df["Channel Subscriber"] > 0)]

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
        """Analyze with SVM using popularity score (view count / channel subscribers)"""
        if target_type == "classification":
            # Classification: Predict popularity category based on popularity score
            if self.processed_data is None:
                raise ValueError(
                    "Processed data is not available. Please call preprocess_data() first."
                )
            y = self.processed_data["Popularity_Category"].dropna()
            X_pca_filtered = X_pca[: len(y)]

            # Split train/test data
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca_filtered, y, test_size=0.2, random_state=42, stratify=y
            )

            self._last_split_cls = (X_train, X_test, y_train, y_test)
            
            # SVM classifier
            svm_classifier = SVC(kernel="rbf", random_state=42)
            svm_classifier.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = svm_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print("\n=== SVM Classification Results (Popularity Category) ===")
            print(f"Accuracy: {accuracy:.3f}")
            print("\nDetailed Report:")
            print(classification_report(y_test, y_pred))

            return svm_classifier, accuracy

        else:
            # Regression: Predict popularity score directly
            if self.processed_data is None:
                raise ValueError(
                    "Processed data is not available. Please call preprocess_data() first."
                )
            y = np.log1p(
                self.processed_data["Popularity_Score"]
            )  # Normalize with log transformation
            X_pca_filtered = X_pca[: len(y)]

            # Split train/test data
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca_filtered, y, test_size=0.2, random_state=42
            )

            self._last_split_reg = (X_train, X_test, y_train, y_test)

            # SVM regressor
            svm_regressor = SVR(kernel="rbf")
            svm_regressor.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = svm_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print("\n=== SVM Regression Results (Popularity Score) ===")
            print(f"Mean Squared Error: {mse:.3f}")
            print(f"R² Score: {r2:.3f}")

            return svm_regressor, r2
        
    
    def plot_svm_classification_results(self,
                                    clf,
                                    X_train, X_test,
                                    y_train, y_test,
                                    X_pca_2d=None,
                                    fname="svm_classification_plots.png"):
        """
        • Confusion‑matrix heat‑map
        • If binary → ROC 曲線
          Else      → Precision / Recall per class
        • (optional) decision regions on PC1–PC2
        """
        from sklearn.metrics import (precision_recall_fscore_support,
                                     ConfusionMatrixDisplay)
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        classes = np.unique(y_test)
        n_class = len(classes)

        plt.figure(figsize=(14, 4))

        # 1️⃣  Confusion matrix
        plt.subplot(1, 3, 1)
        ConfusionMatrixDisplay.from_estimator(
            clf, X_test, y_test, cmap="Blues", ax=plt.gca(), colorbar=False
        )
        plt.title("Confusion matrix")

        # 2️⃣  ROC or PR‑bars
        plt.subplot(1, 3, 2)
        if n_class == 2:            # ── binary ──────────────────
            from sklearn.metrics import RocCurveDisplay
            RocCurveDisplay.from_estimator(
                clf, X_test, y_test,
                ax=plt.gca(),
                plot_chance_level=True
            )
            plt.title("ROC curve")
        else:                       # ── multi‑class ─────────────
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, clf.predict(X_test), labels=classes, zero_division=0
            )
            x = np.arange(n_class)
            plt.bar(x-0.2, prec, 0.4, label="Precision")
            plt.bar(x+0.2, rec,  0.4, label="Recall")
            plt.xticks(x, classes, rotation=30)
            plt.ylim(0, 1)
            plt.legend()
            plt.title("Precision / Recall by class")

        # 3️⃣  Decision regions (optional)
        # ─── Decision regions (optional) ─────────────────────────────
        if X_pca_2d is not None and X_pca_2d.shape[1] >= 2:
            from sklearn.svm import SVC
            ax = plt.subplot(1, 3, 3)
        
            # ◉ ① 文字列ラベル → 数値に変換
            class_to_int = {lbl: i for i, lbl in enumerate(classes)}
            y_train_num  = np.array([class_to_int[lbl] for lbl in y_train])
        
            # ◉ ② 2D 用の SVM を学習
            clf2d = SVC(kernel="rbf", gamma="auto").fit(
                X_pca_2d[: len(y_train)], y_train_num
            )
        
            # ◉ ③ メッシュグリッド作成
            x_min, x_max = X_pca_2d[:, 0].min() - 0.5, X_pca_2d[:, 0].max() + 0.5
            y_min, y_max = X_pca_2d[:, 1].min() - 0.5, X_pca_2d[:, 1].max() + 0.5
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 200),
                np.linspace(y_min, y_max, 200),
            )
        
            Z = clf2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
            # ◉ ④ 数値 Z なら contourf OK
            cmap = sns.color_palette("pastel", n_colors=n_class)
            ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(-0.5, n_class, 1), colors=cmap)
        
            # ◉ ⑤ テスト点を描画
            y_test_num = np.array([class_to_int[lbl] for lbl in y_test])
            ax.scatter(
                X_pca_2d[len(y_train) :, 0],
                X_pca_2d[len(y_train) :, 1],
                c=y_test_num,
                edgecolor="k",
                cmap="viridis",
                alpha=0.7,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Decision regions (PC1–PC2)")


        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Classification plots saved → {fname}")
    # -----------------------------------------------------------------
    def plot_svm_regression_results(self,
                                    svr,
                                    X_train, X_test,
                                    y_train, y_test,
                                    fname="svm_regression_plots.png"):
        """
        • Predicted vs. actual scatter
        • Residual histogram
        """
        y_pred = svr.predict(X_test)

        plt.figure(figsize=(10, 4))

        # 1️⃣  Predicted vs actual
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_test, y=y_pred, alpha=.6)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], "--", color="grey")
        plt.xlabel("Actual log‑popularity")
        plt.ylabel("Predicted")
        plt.title("Predicted vs. actual")

        # 2️⃣  Residuals
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred
        sns.histplot(residuals, bins=40, kde=True)
        plt.xlabel("Residuals"); plt.ylabel("Count")
        plt.title("Residual distribution")

        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Regression plots saved → {fname}")

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
        popularity_scores = self.processed_data["Popularity_Score"]

        # Filter out low popularity scores
        mask = popularity_scores >= 0.0
        X_pca_filtered = X_pca[mask]
        popularity_scores_filtered = popularity_scores[mask]

        scatter = plt.scatter(
            X_pca_filtered[:, 0],
            X_pca_filtered[:, 1],
            c=np.log1p(popularity_scores_filtered),
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Log(Popularity Score)")
        plt.xlabel("First Component")
        # plt.xlim(-4, 4)
        # plt.ylim(-4, 4)
        plt.ylabel("Second Component")
        plt.title("Distribution in Principal Component Space (Popularity Score >= 0.5)")

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

        # 4. Popularity score distribution
        plt.subplot(2, 3, 4)
        plt.hist(np.log1p(self.processed_data["Popularity_Score"]), bins=50, alpha=0.7)
        plt.xlabel("Log(Popularity Score)")
        plt.ylabel("Frequency")
        plt.title("Popularity Score Distribution")

        # 5. Channel-wise analysis (popularity score)
        plt.subplot(2, 3, 5)
        channel_stats = (
            self.processed_data.groupby("Channel_Name")["Popularity_Score"]
            .mean()
            .sort_values(ascending=False)
        )

        plt.bar(range(len(channel_stats)), np.array(channel_stats.values))
        plt.xlabel("Channel")
        plt.ylabel("Average Popularity Score")
        plt.title("Average Popularity Score by Channel")
        plt.xticks(range(len(channel_stats)), channel_stats.index.tolist(), rotation=45)

        # 6. Relationship between duration and popularity score
        plt.subplot(2, 3, 6)
        plt.scatter(
            self.processed_data["Duration_Seconds"],
            np.log1p(self.processed_data["Popularity_Score"]),
            alpha=0.6,
        )
        plt.xlabel("Duration (Seconds)")
        plt.ylabel("Log(Popularity Score)")
        plt.title("Duration vs Popularity Score")

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
        print("YouTube Video Popularity Analysis - Insights")
        print("Popularity Score = View Count / Channel Subscribers")
        print("=" * 50)

        # Print PCA component equations
        self.print_pca_components(feature_columns)

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
                f"   Average Popularity Score: {self.processed_data['Popularity_Score'].mean():.4f}"
            )
            print(
                f"   Maximum Popularity Score: {self.processed_data['Popularity_Score'].max():.4f}"
            )
            print(
                f"   Average Video Length: {self.processed_data['Duration_Seconds'].mean()/60:.1f} minutes"
            )
        else:
            print("Processed data is not available.")

        # 4. Top 3 Channels by Average Popularity Score
        print("\n4. Top 3 Channels by Average Popularity Score:")
        if self.processed_data is not None:
            channel_stats = (
                self.processed_data.groupby("Channel_Name")["Popularity_Score"]
                .mean()
                .sort_values(ascending=False)
            )
            for i, (channel, score) in enumerate(channel_stats.head(3).items()):
                print(f"   {i+1}. {channel}: {score:.4f}")
        else:
            print("Processed data is not available for channel analysis.")

    def print_pca_components(self, feature_columns, n_components=5):
        """Print PCA component equations"""
        if self.pca is None:
            raise ValueError(
                "PCA has not been performed yet. Please call perform_pca() first."
            )

        print("\n" + "=" * 80)
        print("PCA Component Equations")
        print("=" * 80)

        # Display up to n_components or the total number of components, whichever is smaller
        n_display = min(n_components, len(self.pca.components_))

        for i in range(n_display):
            print(
                f"\nComponent {i+1} (Explained Variance Ratio: {self.pca.explained_variance_ratio_[i]:.4f}):"
            )
            print("-" * 60)

            # Get the coefficients for this component
            coefficients = self.pca.components_[i]

            # Create the equation string
            equation_parts = []
            for j, coef in enumerate(coefficients):
                feature_name = feature_columns[j]
                if coef >= 0 and len(equation_parts) > 0:
                    equation_parts.append(f" + {coef:.4f} * {feature_name}")
                else:
                    equation_parts.append(f"{coef:.4f} * {feature_name}")

            equation = "PC{} = ".format(i + 1) + "".join(equation_parts)
            print(equation)

            # Also show the top contributing features
            print(f"\nTop 5 contributing features for Component {i+1}:")
            feature_importance = np.abs(coefficients)
            indices = np.argsort(feature_importance)[::-1]
            for k in range(min(5, len(indices))):
                feature_idx = indices[k]
                contribution = coefficients[feature_idx]
                print(
                    f"  {k+1}. {feature_columns[feature_idx]}: {contribution:.4f} (|{abs(contribution):.4f}|)"
                )

        print("\n" + "=" * 80)



def main():
    # Execute analysis
    analyzer = YouTubeDataAnalyzer("./metadata")
    # Output directory
    output_dir = "./output"

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

    # Visualize results (PCA)
    visualize_results_output = os.path.join(output_dir, "youtube_analysis_results.png")
    analyzer.visualize_results(X_pca, feature_columns, visualize_results_output)


    #Visualize results (SVM classification)
    Xtr_c, Xte_c, ytr_c, yte_c = analyzer._last_split_cls
    analyzer.plot_svm_classification_results(
        svm_classifier, Xtr_c, Xte_c, ytr_c, yte_c,
        X_pca_2d=X_pca[:, :2],
        fname=os.path.join(output_dir, "svm_classification.png")
    )

    #Visualize results (SVM regression)
    Xtr_r, Xte_r, ytr_r, yte_r = analyzer._last_split_reg
    analyzer.plot_svm_regression_results(
        svm_regressor, Xtr_r, Xte_r, ytr_r, yte_r,
        fname=os.path.join(output_dir, "svm_regression.png")
    )

    # Generate insights
    analyzer.generate_insights(feature_columns)

    # Print PCA components
    analyzer.print_pca_components(feature_columns, n_components=5)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
