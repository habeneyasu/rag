import gradio as gr
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

from evaluation.eval import (
    evaluate_all_retrieval,
    evaluate_all_answers,
    evaluate_comprehensive,
    generate_parameter_recommendations,
)
from evaluation.test import load_tests

load_dotenv(override=True)

# Color coding thresholds - Retrieval
MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

# Color coding thresholds - Answer (1-5 scale)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


def get_color(value: float, metric_type: str) -> str:
    """Get color based on metric value and type."""
    if metric_type == "mrr":
        if value >= MRR_GREEN:
            return "green"
        elif value >= MRR_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        elif value >= NDCG_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "coverage":
        if value >= COVERAGE_GREEN:
            return "green"
        elif value >= COVERAGE_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        elif value >= ANSWER_AMBER:
            return "orange"
        else:
            return "red"
    return "black"


def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
    """Format a metric with color coding."""
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""<div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>"""


def run_retrieval_evaluation(progress=gr.Progress()):
    """Run retrieval evaluation and yield updates."""
    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_all_retrieval(use_comprehensive=True):
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage

        category_mrr[test.category].append(result.mrr)

        # Update progress bar only
        progress(prog_value, desc=f"Evaluating test {count}...")

    # Calculate final averages
    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    final_html = f"""<div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>"""

    # Create final bar chart data
    category_data = []
    for category, mrr_scores in category_mrr.items():
        avg_cat_mrr = sum(mrr_scores) / len(mrr_scores)
        category_data.append({"Category": category, "Average MRR": avg_cat_mrr})

    df = pd.DataFrame(category_data)

    return final_html, df


def run_answer_evaluation(progress=gr.Progress()):
    """Run answer evaluation with comprehensive metrics including token usage and cost."""
    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0
    total_tokens = 0
    total_cost = 0.0
    category_accuracy = defaultdict(list)
    category_tokens = defaultdict(list)
    category_cost = defaultdict(list)
    count = 0
    queries_with_tokens = 0

    for test, answer_eval, retrieval_eval, token_usage, prog_value in evaluate_all_answers(use_comprehensive=True):
        count += 1
        total_accuracy += answer_eval.accuracy
        total_completeness += answer_eval.completeness
        total_relevance += answer_eval.relevance

        category_accuracy[test.category].append(answer_eval.accuracy)

        # Track token usage and cost
        if token_usage:
            queries_with_tokens += 1
            total_tokens += token_usage.total_tokens
            total_cost += token_usage.cost_usd
            category_tokens[test.category].append(token_usage.total_tokens)
            category_cost[test.category].append(token_usage.cost_usd)

        # Update progress bar only
        progress(prog_value, desc=f"Evaluating test {count}...")

    # Calculate final averages
    avg_accuracy = total_accuracy / count
    avg_completeness = total_completeness / count
    avg_relevance = total_relevance / count
    avg_tokens = total_tokens / queries_with_tokens if queries_with_tokens > 0 else 0
    avg_cost = total_cost / queries_with_tokens if queries_with_tokens > 0 else 0
    total_cost_all = total_cost

    final_html = f"""<div style="padding: 0;">
        {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 15px; background-color: #e7f3ff; border-radius: 8px; border-left: 5px solid #2196F3;">
            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Token Usage & Cost</div>
            <div style="font-size: 18px; font-weight: bold; color: #2196F3;">Avg: {avg_tokens:.0f} tokens/query | ${avg_cost:.4f}/query</div>
            <div style="font-size: 14px; color: #666; margin-top: 5px;">Total: {total_tokens:,} tokens | ${total_cost_all:.4f} total</div>
        </div>
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>"""

    # Create final bar chart data
    category_data = []
    for category, accuracy_scores in category_accuracy.items():
        avg_cat_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        avg_cat_tokens = sum(category_tokens.get(category, [0])) / len(category_tokens.get(category, [1]))
        avg_cat_cost = sum(category_cost.get(category, [0])) / len(category_cost.get(category, [1]))
        category_data.append({
            "Category": category,
            "Average Accuracy": avg_cat_accuracy,
            "Avg Tokens": avg_cat_tokens,
            "Avg Cost ($)": avg_cat_cost
        })

    df = pd.DataFrame(category_data)

    return final_html, df


def run_comprehensive_evaluation(progress=gr.Progress()):
    """Run comprehensive evaluation with all metrics and parameter recommendations."""
    tests = load_tests(use_comprehensive=True)
    total_tests = len(tests)
    
    all_metrics = []
    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0
    total_tokens = 0
    total_cost = 0.0
    count = 0
    
    for index, test in enumerate(tests):
        count += 1
        metrics = evaluate_comprehensive(test)
        all_metrics.append(metrics)
        
        total_accuracy += metrics.accuracy
        total_completeness += metrics.completeness
        total_relevance += metrics.relevance
        
        if metrics.token_usage:
            total_tokens += metrics.token_usage.total_tokens
            total_cost += metrics.token_usage.cost_usd
        
        progress((index + 1) / total_tests, desc=f"Evaluating test {count}/{total_tests}...")
    
    # Calculate averages
    avg_accuracy = total_accuracy / count if count > 0 else 0
    avg_completeness = total_completeness / count if count > 0 else 0
    avg_relevance = total_relevance / count if count > 0 else 0
    avg_tokens = total_tokens / count if count > 0 else 0
    avg_cost = total_cost / count if count > 0 else 0
    
    recommendations = generate_parameter_recommendations(all_metrics)
    
    metrics_html = f"""<div style="padding: 0;">
        {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 15px; background-color: #e7f3ff; border-radius: 8px; border-left: 5px solid #2196F3;">
            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Token Usage & Cost</div>
            <div style="font-size: 18px; font-weight: bold; color: #2196F3;">Avg: {avg_tokens:.0f} tokens/query | ${avg_cost:.4f}/query</div>
            <div style="font-size: 14px; color: #666; margin-top: 5px;">Total: {total_tokens:,} tokens | ${avg_cost:.4f} total</div>
        </div>
    </div>"""
    
    if recommendations:
        rec_html = """<div style="margin-top: 20px; padding: 15px; background-color: #fff3cd; border-radius: 8px; border-left: 5px solid #ffc107;">
            <div style="font-size: 16px; font-weight: bold; color: #856404; margin-bottom: 10px;">📊 Parameter Recommendations</div>"""
        for metric, rec in recommendations.items():
            rec_html += f"""<div style="margin-top: 10px; padding: 10px; background-color: #fff; border-radius: 5px;">
                <div style="font-size: 14px; font-weight: bold; color: #856404;">{metric.capitalize()}:</div>
                <div style="font-size: 13px; color: #666; margin-top: 5px;">{rec}</div>
            </div>"""
        rec_html += "</div>"
    else:
        rec_html = """<div style="margin-top: 20px; padding: 15px; background-color: #d4edda; border-radius: 8px; border-left: 5px solid #28a745;">
            <div style="font-size: 14px; color: #155724; font-weight: bold;">✓ All metrics are performing well! No recommendations needed.</div>
        </div>"""
    
    final_html = metrics_html + rec_html + f"""<div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Comprehensive Evaluation Complete: {count} tests</span>
        </div>"""
    
    category_metrics = defaultdict(lambda: {"accuracy": [], "completeness": [], "relevance": []})
    for m in all_metrics:
        category_metrics[m.category]["accuracy"].append(m.accuracy)
        category_metrics[m.category]["completeness"].append(m.completeness)
        category_metrics[m.category]["relevance"].append(m.relevance)
    
    category_data = [
        {
            "Category": category,
            "Avg Accuracy": sum(scores["accuracy"]) / len(scores["accuracy"]),
            "Avg Completeness": sum(scores["completeness"]) / len(scores["completeness"]),
            "Avg Relevance": sum(scores["relevance"]) / len(scores["relevance"]),
        }
        for category, scores in category_metrics.items()
        if scores["accuracy"]
    ]
    
    df = pd.DataFrame(category_data) if category_data else pd.DataFrame()
    
    return final_html, df


def main():
    """Launch the Gradio evaluation app."""
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        gr.Markdown("# 📊 RAG Evaluation Dashboard")
        gr.Markdown("Evaluate retrieval and answer quality for the Insurellm RAG system")

        # RETRIEVAL SECTION
        gr.Markdown("## 🔍 Retrieval Evaluation")

        retrieval_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                retrieval_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )

            with gr.Column(scale=1):
                retrieval_chart = gr.BarPlot(
                    x="Category",
                    y="Average MRR",
                    title="Average MRR by Category",
                    y_lim=[0, 1],
                    height=400,
                )

        # ANSWERING SECTION
        gr.Markdown("## 💬 Answer Evaluation")

        answer_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                answer_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )

            with gr.Column(scale=1):
                answer_chart = gr.BarPlot(
                    x="Category",
                    y="Average Accuracy",
                    title="Average Accuracy by Category",
                    y_lim=[1, 5],
                    height=400,
                )

        # COMPREHENSIVE EVALUATION SECTION
        gr.Markdown("## 📊 Comprehensive Evaluation (All Metrics + Recommendations)")

        comprehensive_button = gr.Button("Run Comprehensive Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                comprehensive_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Comprehensive Evaluation' to start</div>"
                )

            with gr.Column(scale=1):
                comprehensive_chart = gr.BarPlot(
                    x="Category",
                    y="Avg Accuracy",
                    title="Metrics by Category",
                    y_lim=[1, 5],
                    height=400,
                )

        # Wire up the evaluations
        retrieval_button.click(
            fn=run_retrieval_evaluation,
            outputs=[retrieval_metrics, retrieval_chart],
        )

        answer_button.click(
            fn=run_answer_evaluation,
            outputs=[answer_metrics, answer_chart],
        )

        comprehensive_button.click(
            fn=run_comprehensive_evaluation,
            outputs=[comprehensive_metrics, comprehensive_chart],
        )

    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
