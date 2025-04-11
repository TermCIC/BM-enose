from enose_module import (
    run_enose_capture,
    run_enose_capture_hourly,
    run_enose_capture_min,
    start_pca_plot_thread
)
import matplotlib
import matplotlib.pyplot as plt
import threading

def start_pca_plot_thread():
    pca_thread = threading.Thread(target=plot_pca, daemon=True)
    pca_thread.start()

def plot_pca():
    """Runs in the main thread to update PCA plot."""
    plt.ion()
    fig, ax = plt.subplots()
    pca = PCA(n_components=2)

    while not stop_flag.is_set():
        try:
            # Read CSV and check for missing values
            df = pd.read_csv(READING_CSV_FILE)
            df.dropna(inplace=True)

            if len(df) > 2:
                # Extract treatments and assign colors
                treatments = df['treatment'].astype(str)
                unique_treatments = sorted(treatments.unique())

                cmap = matplotlib.colormaps.get_cmap(
                    'tab10')  # updated to avoid deprecation
                color_map = {label: cmap(i % 10)
                             for i, label in enumerate(unique_treatments)}
                colors = treatments.map(color_map)

                # PCA on sensor columns (exclude treatment and timestamp)
                transformed_data = pca.fit_transform(df.iloc[:, :-5])

                # Plot
                ax.clear()
                ax.scatter(
                    transformed_data[:, 0],
                    transformed_data[:, 1],
                    c=colors,
                    alpha=0.6
                )

                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title('PCA of Gas Sensor Readings')

                # Add legend
                handles = [
                    Line2D([0], [0], marker='o', color='w', label=label,
                           markerfacecolor=color_map[label], markersize=8)
                    for label in unique_treatments
                ]
                ax.legend(handles=handles, title="Treatment", loc="best")

                plt.draw()
                plt.pause(1)
        except Exception as e:
            print(f"PCA Plot Error: {e}")

    print("PCA thread stopped.")

start_pca_plot_thread()
# run_enose_capture(treatment_input="CONTROL")
# run_enose_capture_hourly("hourly_monitor")
run_enose_capture_min("min_monitor")