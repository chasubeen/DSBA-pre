import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## 결과 저장 경로 설정
RESULTS_DIR = "./results"
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_results.txt")


### === Table ===
## 로그 파일 파싱
def parse_log_file(log_file):
    experiments = []
    current_exp = None

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("=== Running Experiment:"):
                model_info = re.findall(r"Experiment: (.*) \(Pretrained: (.*)\)", line)[0]
                current_exp = {"Model": model_info[0], "Pretrained": model_info[1], "Epochs": []}

            elif line.startswith("Epoch"):
                epoch_info = int(re.findall(r"Epoch (\d+)/", line)[0])
                current_exp["Epochs"].append({"Epoch": epoch_info})

            elif "Train Loss" in line or "Test Loss" in line:
                metrics = {m.split(":")[0].strip(): float(m.split(":")[1].replace('%', '').strip()) for m in line.split(",")}
                current_exp["Epochs"][-1].update(metrics)

            if line.startswith("✅ 모델 저장 완료"):
                experiments.append(current_exp)

    data = []
    for exp in experiments:
        model_name = exp["Model"]
        pretrained = exp["Pretrained"]

        for epoch_data in exp["Epochs"]:
            data.append({
                "Model": model_name,
                "Pretrained": pretrained,
                "Epoch": epoch_data.get("Epoch", 0),
                "Train Loss": epoch_data.get("Train Loss", None),   
                "Test Loss": epoch_data.get("Test Loss", None),     
                "Accuracy": epoch_data.get("Test Accuracy", None),
                "Top-5 Error": epoch_data.get("Top-5 Error", None),
                "Precision": epoch_data.get("Precision", None),
                "Recall": epoch_data.get("Recall", None),
                "F1 Score": epoch_data.get("F1 Score", None)
            })

    return pd.DataFrame(data)

## 평균 및 표준편차 계산
def calculate_average_metrics(df):
    grouped = df.groupby(["Model", "Pretrained"])
    avg_std_df = grouped.agg({
        "Accuracy": ["mean", "std"],
        "Top-5 Error": ["mean", "std"],
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1 Score": ["mean", "std"]
    }).reset_index()

    return avg_std_df

## 결과 포맷팅
def format_metrics(df):
    formatted_df = pd.DataFrame()
    formatted_df["Model"] = df["Model"]
    formatted_df["Pretrained"] = df["Pretrained"]

    for metric in ["Accuracy", "Top-5 Error", "Precision", "Recall", "F1 Score"]:
        formatted_df[metric] = df.apply(
            lambda row: f"{row[(metric, 'mean')]:.2f} ± {row[(metric, 'std')]:.2f}", 
            axis=1
        )
    return formatted_df



### === Figure ===
def create_figures(df, original_df):
    sns.set(style="whitegrid")

    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    ## === Loss 시각화 ===
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for (model, pretrained), group in original_df.groupby(["Model", "Pretrained"]):
        label = f"{model} ({pretrained})"
        axs[0].plot(group["Epoch"], group["Train Loss"], label=label)
        axs[1].plot(group["Epoch"], group["Test Loss"], label=label)

    axs[0].set_title("Train Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].set_title("Test Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "loss.png"))
    plt.close()


    ## === Metrics 시각화 ===
    metrics = ["Accuracy", "Top-5 Error", "Precision", "Recall", "F1 Score"]

    # Melt data for easier seaborn plotting
    melted_df = pd.melt(df, id_vars=["Model", "Pretrained"], value_vars=metrics,
                        var_name="Metric", value_name="Value")

    # ✅ Extract numeric mean values
    melted_df["Mean"] = melted_df["Value"].apply(lambda x: float(x.split("±")[0].strip()))

    # Grouped Bar Chart
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plot_data = melted_df[melted_df["Metric"] == metric].sort_values(by="Mean")

        ax = sns.barplot(data=plot_data, 
                         x="Model", y="Mean", hue="Pretrained", 
                         palette="Set2", edgecolor="black", width=0.6)

        # 평균값만 표시
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10)

        plt.title(f"{metric}")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.legend(title="Pretrained", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{metric.replace(' ', '_').lower()}.png"))
        plt.close()



def main():
    df = parse_log_file(LOG_FILE)
    avg_std_df = calculate_average_metrics(df)
    formatted_df = format_metrics(avg_std_df)

    # 결과 테이블 저장
    table_path = os.path.join(RESULTS_DIR, "table.csv")
    formatted_df.to_csv(table_path, index=False)

    # 성능 시각화
    create_figures(formatted_df, df)
    print(f"✅ 결과 테이블과 그래프가 {RESULTS_DIR} 폴더에 저장되었습니다!")

if __name__ == "__main__":
    main()
