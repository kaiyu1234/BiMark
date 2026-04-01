import json
import argparse
import evaluate
import os
import numpy as np

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main(args):
    # 1. 加载 Reference (真实参考文本)
    ref_path = os.path.join(args.data_dir, "human_written.jsonl")
    refs_data = load_jsonl(ref_path)
    ref_dict = {item['prompt_idx']: item['human_completion'] for item in refs_data}

    # 2. 加载 Predictions (模型生成的文本)
    gen_path = os.path.join(args.data_dir, "generation_text.jsonl")
    gens_data = load_jsonl(gen_path)

    preds = []
    refs = []
    
    # 提取对齐的文本
    for item in gens_data:
        idx = item['prompt_idx']
        if idx in ref_dict:
            preds.append(item['generation_text'])
            refs.append(ref_dict[idx])
            
    print(f"成功对齐并加载了 {len(preds)} 条数据进行评估。")

    # 3. 计算指标
    if args.task == 'summarization':
        print("正在计算 ROUGE 分数...")
        rouge = evaluate.load('rouge')
        rouge_results = rouge.compute(predictions=preds, references=refs,use_stemmer=True,use_aggregator=True)
        print("\n=== 摘要任务评估结果 (ROUGE) ===")
        print(f"ROUGE-1: {rouge_results['rouge1'] * 100:.2f}")
        print(f"ROUGE-2: {rouge_results['rouge2'] * 100:.2f}")
        print(f"ROUGE-L: {rouge_results['rougeL'] * 100:.2f}")
        
        # 摘要任务 BERTScore
        print("\n正在计算 BERTScore 分数...")
        bertscore = evaluate.load('bertscore')
        bert_results = bertscore.compute(
            predictions=preds, 
            references=refs, 
            lang='en',
            rescale_with_baseline=True,  # 开启基线缩放，把80多分拉伸成30多分的关键参数
            model_type='roberta-large'
        )
        bert_precision = np.mean(bert_results['precision'])
        bert_recall = np.mean(bert_results['recall'])
        bert_f1 = np.mean(bert_results['f1'])
        
        print("\n=== 摘要任务评估结果 (BERTScore) ===")
        print(f"BERTScore Precision: {bert_precision * 100:.2f}")
        print(f"BERTScore Recall: {bert_recall * 100:.2f}")
        print(f"BERTScore F1: {bert_f1 * 100:.2f}")
        
    elif args.task == 'translation':
        print("正在计算 BLEU 分数...")
        sacrebleu = evaluate.load('sacrebleu')
        results = sacrebleu.compute(predictions=preds, references=refs)
        print("\n=== 翻译任务评估结果 (SacreBLEU) ===")
        print(f"BLEU Score: {results['score']:.2f}")
        
        # ===================== 新增：翻译任务 BERTScore =====================
        print("\n正在计算 BERTScore 分数...")
        bertscore = evaluate.load('bertscore')
        bert_results = bertscore.compute(
            predictions=preds, 
            references=refs, 
            lang='en',  # 根据你的翻译任务修改！
            # model_type='roberta-large',
            rescale_with_baseline=True  # 开启基线缩放，把80多分拉伸成30多分的关键参数
        )
        bert_precision = np.mean(bert_results['precision'])
        bert_recall = np.mean(bert_results['recall'])
        bert_f1 = np.mean(bert_results['f1'])
        
        print("\n=== 翻译任务评估结果 (BERTScore) ===")
        print(f"BERTScore Precision: {bert_precision * 100:.2f}")
        print(f"BERTScore Recall: {bert_recall * 100:.2f}")
        print(f"BERTScore F1: {bert_f1 * 100:.2f}")
        # ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing generation_text.jsonl and human_written.jsonl")
    parser.add_argument("--task", type=str, choices=['summarization', 'translation'], required=True, help="Task type to evaluate")
    
    args = parser.parse_args()
    main(args)
