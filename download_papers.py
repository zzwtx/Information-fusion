"""
Download all 12 papers on federated fine-tuning of multimodal/VL models
from ICML/NeurIPS/ICLR (2024-2026) to D:\FL\智能信息融合下
"""
import ssl
import urllib.request
import urllib.error
import os
import json
import time

# SSL workaround for Windows
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE_DIR = r"D:\FL\智能信息融合下"

# ============================================================
# Paper definitions: all 12 papers with download sources
# ============================================================
PAPERS = [
    # ---- ICLR 2024 ----
    {
        "id": "FedTPG",
        "title": "FedTPG: Federated Text-driven Prompt Generation for Vision-Language Models",
        "authors": "Chen Qiu, Xingyu Li, Chaithanya Kumar Mummadi, Madan Ravi Ganesh, Zhenzhen Li, Lu Peng, Wan-Yi Lin",
        "venue": "ICLR",
        "year": 2024,
        "arxiv_id": "2310.06123",
        "keywords": "federated learning, prompt tuning, CLIP, text-driven prompt generation, vision-language models",
        "abstract": "FedTPG learns a prompt generation network (PromptTranslator) conditioned on class name embeddings via cross-attention, enabling dynamic context-aware prompt generation for both seen and unseen classes in federated settings.",
        "url_arxiv": "https://arxiv.org/pdf/2310.06123",
        "url_proceedings": "https://proceedings.iclr.cc/paper_files/paper/2024/hash/37e90dcf2909b5068858b34b5239f187-Abstract-Conference.html",
        "code": "https://github.com/boschresearch/FedTPG"
    },
    # ---- NeurIPS 2024 ----
    {
        "id": "PromptFolio",
        "title": "PromptFolio: Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method",
        "authors": "Bikang Pan, Wei Huang, Ye Shi",
        "venue": "NeurIPS",
        "year": 2024,
        "arxiv_id": "2409.19610",
        "keywords": "federated learning, theoretical analysis, prompt portfolio, CLIP, feature learning theory, generalization",
        "abstract": "First theoretical analysis framework for prompt-based federated learning with VLMs. Proposes PromptFolio treating global and local prompts as a portfolio, balancing signal-to-noise ratio for generalization under data heterogeneity.",
        "url_arxiv": "https://arxiv.org/pdf/2409.19610",
        "url_proceedings": "https://neurips.cc/virtual/2024/poster/94723",
        "code": "https://github.com/PanBikang/PromptFolio"
    },
    {
        "id": "PFPT",
        "title": "Probabilistic Federated Prompt-Tuning with Non-IID and Imbalanced Data",
        "authors": "Pei-Yau Weng, Minh Hoang, Lam M. Nguyen, My T. Thai, Tsui-Wei Weng, Trong Nghia Hoang",
        "venue": "NeurIPS",
        "year": 2024,
        "arxiv_id": None,
        "keywords": "federated learning, prompt tuning, probabilistic modeling, bipartite matching, non-IID data",
        "abstract": "Formulates federated prompt aggregation as a hierarchical probabilistic set modeling problem with weighted bipartite matching to resolve prompt misalignment under extreme data heterogeneity.",
        "url_arxiv": None,
        "url_proceedings": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/951877b24b376c5f4612e850251ee85b-Abstract-Conference.html",
        "url_alt_pdf": "https://openreview.net/pdf?id=nw6ANsC66G",
        "code": None
    },
    # ---- ICML 2025 ----
    {
        "id": "FOCoOp",
        "title": "FOCoOp: Enhancing Out-of-Distribution Robustness in Federated Prompt Learning for Vision-Language Models",
        "authors": "Xinting Liao, Weiming Liu, Jiaming Qian, Pengyang Zhou, Jiahe Xu, Wenjie Wang, Chaochao Chen, Xiaolin Zheng, Tat-Seng Chua",
        "venue": "ICML",
        "year": 2025,
        "arxiv_id": "2506.16218",
        "keywords": "federated learning, prompt tuning, out-of-distribution robustness, CLIP, distributionally robust optimization, optimal transport",
        "abstract": "Introduces three prompt types (ID global, local, OOD) with bi-level distributionally robust optimization and semi-unbalanced optimal transport to maintain accuracy while improving OOD robustness in federated prompt learning.",
        "url_arxiv": "https://arxiv.org/pdf/2506.16218",
        "url_proceedings": "https://proceedings.mlr.press/v267/liao25e.html",
        "code": None
    },
    {
        "id": "FedDDA",
        "title": "FedDDA: Federated Disentangled Tuning with Textual Prior Decoupling and Visual Dynamic Adaptation",
        "authors": "Yihao Yang, Wenke Huang, Guancheng Wan, Bin Yang, Mang Ye",
        "venue": "ICML",
        "year": 2025,
        "arxiv_id": None,
        "keywords": "federated learning, parameter-efficient fine-tuning, textual prior decoupling, visual adaptation, MoE, VLMs",
        "abstract": "Addresses personalized federated PEFT for VLMs via Textual Prior Decoupling (global+local prompts fused with hand-crafted prior) and Visual Dynamic Adaptation (dual adapters + MoE-style gating).",
        "url_arxiv": None,
        "url_proceedings": "https://proceedings.mlr.press/v267/yang25k.html",
        "url_alt_pdf": "https://raw.githubusercontent.com/mlresearch/v267/main/assets/yang25k/yang25k.pdf",
        "code": "https://github.com/MoratalYang/FedDDA"
    },
    {
        "id": "FedPHA",
        "title": "FedPHA: Federated Prompt Learning for Heterogeneous Client Adaptation",
        "authors": "Chengying Fang, Wenke Huang, Guancheng Wan, Yihao Yang, Mang Ye",
        "venue": "ICML",
        "year": 2025,
        "arxiv_id": None,
        "keywords": "federated learning, prompt tuning, heterogeneous clients, SVD projection, bidirectional alignment, CLIP",
        "abstract": "Combines fixed-length global prompt with variable-length local prompts per client; uses SVD-based projection and bidirectional alignment to disentangle global conflicts from client heterogeneity.",
        "url_arxiv": None,
        "url_proceedings": "https://proceedings.mlr.press/v267/fang25e.html",
        "url_alt_pdf": "https://raw.githubusercontent.com/mlresearch/v267/main/assets/fang25e/fang25e.pdf",
        "code": None
    },
    {
        "id": "FedAG",
        "title": "FedAG: Enhancing Foundation Models with Federated Domain Knowledge Infusion",
        "authors": "Jiaqi Wang, Jingtao Li, Weiming Zhuang, Chen Chen, Lingjuan Lyu, Fenglong Ma",
        "venue": "ICML",
        "year": 2025,
        "arxiv_id": None,
        "keywords": "federated learning, adapter, domain generalization, CLIP, cross-silo, knowledge infusion",
        "abstract": "Cross-silo federated fine-tuning of vision foundation models using multiple fine-grained adapters with quality-aware mutual learning and attention-regularized cross-domain learning for out-of-domain generalization.",
        "url_arxiv": None,
        "url_proceedings": "https://proceedings.mlr.press/v267/wang25bk.html",
        "url_alt_pdf": "https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25bk/wang25bk.pdf",
        "code": None
    },
    # ---- ICLR 2025 ----
    {
        "id": "pFedMoAP",
        "title": "pFedMoAP: Mixture of Experts Made Personalized: Federated Prompt Learning for Vision-Language Models",
        "authors": "Jun Luo, Chen Chen, Shandong Wu",
        "venue": "ICLR",
        "year": 2025,
        "arxiv_id": "2410.10114",
        "keywords": "federated learning, mixture of experts, prompt tuning, CLIP, personalization, vision-language models",
        "abstract": "Clients download multiple pre-aggregated prompts as fixed non-local experts; local attention-based gating network generates personalized enhanced text features via MoE-style fusion for VLMs.",
        "url_arxiv": "https://arxiv.org/pdf/2410.10114",
        "url_proceedings": "https://iclr.cc/virtual/2025/poster/27771",
        "code": "https://github.com/ljaiverson/pFedMoAP"
    },
    # ---- NeurIPS 2025 ----
    {
        "id": "FedMGP",
        "title": "FedMGP: Personalized Federated Learning with Multi-Group Text-Visual Prompts",
        "authors": "Weihao Bo, Yanpeng Sun, Yu Wang, Xinyu Zhang, Zechao Li",
        "venue": "NeurIPS",
        "year": 2025,
        "arxiv_id": "2511.00480",
        "keywords": "federated learning, multi-group prompts, text-visual prompts, personalization, CLIP, prompt diversity",
        "abstract": "Equips each client with multiple groups of paired textual+visual prompts with diversity loss for specialization; similarity-guided dynamic aggregation achieves SOTA with only 5.1K communication parameters.",
        "url_arxiv": "https://arxiv.org/pdf/2511.00480",
        "url_proceedings": "https://neurips.cc/virtual/2025/loc/san-diego/poster/119165",
        "code": "https://github.com/weihao-bo/FedMGP"
    },
    {
        "id": "NormFit",
        "title": "NormFit: A Lightweight Solution for Few-Shot Federated Learning with Non-IID Data",
        "authors": "Azadeh Motamedi, Jae-Mo Kang, Il-Min Kim",
        "venue": "NeurIPS",
        "year": 2025,
        "arxiv_id": None,
        "keywords": "federated learning, few-shot learning, LayerNorm tuning, CLIP, lightweight fine-tuning, generalization",
        "abstract": "Selectively fine-tunes only the Pre-LayerNorm parameters of CLIP vision encoder. Extremely lightweight with theoretical generalization gap analysis; works standalone or with existing few-shot FL methods.",
        "url_arxiv": None,
        "url_proceedings": "https://nips.cc/virtual/2025/loc/san-diego/poster/118560",
        "url_alt_pdf": "https://openreview.net/pdf?id=8AhcS0MQdG",
        "code": None
    },
    # ---- ICLR 2026 ----
    {
        "id": "pFedMMA",
        "title": "pFedMMA: Personalized Federated Fine-Tuning with Multi-Modal Adapter for Vision-Language Models",
        "authors": "Sajjad Ghiasvand, Mahnoosh Alizadeh, Ramtin Pedarsani",
        "venue": "ICLR",
        "year": 2026,
        "arxiv_id": "2507.05394",
        "keywords": "federated learning, multi-modal adapter, personalization, CLIP, parameter-efficient fine-tuning, vision-language models",
        "abstract": "First personalized federated framework using multi-modal adapters with modality-specific up/down projection layers and globally shared cross-modal alignment projection; SOTA on 11 datasets.",
        "url_arxiv": "https://arxiv.org/pdf/2507.05394",
        "url_proceedings": "https://iclr.cc/virtual/2026/poster/10008691",
        "code": None
    },
    {
        "id": "FedDuet",
        "title": "Fed-Duet: Dual Expert-Orchestrated Framework for Continual Federated Vision-Language Learning",
        "authors": "Tao Guo, Junwei Chen, Laizhong Cui",
        "venue": "ICLR",
        "year": 2026,
        "arxiv_id": None,
        "keywords": "federated learning, continual learning, vision-language models, dual-expert, cross-attention, catastrophic forgetting",
        "abstract": "Dual-expert mechanism combining server-coordinated semantic prompts with client-personalized modular adapters fused via cross-attention; mitigates catastrophic forgetting in continual federated VL learning.",
        "url_arxiv": None,
        "url_proceedings": "https://iclr.cc/virtual/2026/poster/10010177",
        "url_alt_pdf": "https://openreview.net/pdf?id=l6mY3q9o9b",
        "code": "https://anonymous.4open.science/r/FedDuet-0426"
    },
]


def download_file(url, dest_path, desc=""):
    """Download a file with SSL workaround."""
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OrbitOS/2.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            data = resp.read()
        if len(data) < 1000:
            print(f"  WARNING: {desc} - file too small ({len(data)} bytes), may be an error page")
            return False
        with open(dest_path, 'wb') as f:
            f.write(data)
        print(f"  OK: {desc} -> {os.path.basename(dest_path)} ({len(data)//1024} KB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"  FAIL [{e.code}]: {desc} - {url}")
        return False
    except Exception as e:
        print(f"  FAIL: {desc} - {e}")
        return False


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    total_ok = 0
    total_fail = 0

    for i, paper in enumerate(PAPERS, 1):
        venue = paper["venue"]
        year = paper["year"]
        pid = paper["id"]
        print(f"\n{'='*60}")
        print(f"[{i}/{len(PAPERS)}] {pid} ({venue} {year})")
        print(f"  {paper['title'][:100]}...")

        # Create venue/year subdirectory
        subdir = os.path.join(BASE_DIR, f"{venue}_{year}")
        os.makedirs(subdir, exist_ok=True)

        pdf_path = os.path.join(subdir, f"{pid}.pdf")

        # Skip if already downloaded
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10000:
            print(f"  SKIP: already downloaded ({os.path.getsize(pdf_path)//1024} KB)")
            total_ok += 1
            continue

        # Try arXiv first (best quality)
        downloaded = False
        if paper.get("url_arxiv"):
            downloaded = download_file(paper["url_arxiv"], pdf_path, "arXiv")

        # Try alternative PDF if arXiv fails
        if not downloaded and paper.get("url_alt_pdf"):
            print(f"  Trying alternative source...")
            downloaded = download_file(paper["url_alt_pdf"], pdf_path, "alt PDF")

        if downloaded:
            total_ok += 1
        else:
            total_fail += 1
            # Save a placeholder note
            note_path = os.path.join(subdir, f"{pid}_NOTE.txt")
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(f"Download failed for: {pid}\n")
                f.write(f"Title: {paper['title']}\n")
                f.write(f"arXiv: {paper.get('arxiv_id', 'N/A')}\n")
                f.write(f"Proceedings: {paper.get('url_proceedings', 'N/A')}\n")
                f.write(f"Alt PDF: {paper.get('url_alt_pdf', 'N/A')}\n")

        time.sleep(1)  # Be polite

    # Save metadata JSON
    metadata_path = os.path.join(BASE_DIR, "papers_metadata.json")
    metadata = []
    for paper in PAPERS:
        metadata.append({
            "id": paper["id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "venue": paper["venue"],
            "year": paper["year"],
            "arxiv_id": paper.get("arxiv_id"),
            "keywords": paper["keywords"],
            "abstract": paper["abstract"],
            "url_arxiv": paper.get("url_arxiv"),
            "url_proceedings": paper.get("url_proceedings"),
            "code": paper.get("code"),
        })
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE: {total_ok} OK, {total_fail} FAILED (out of {len(PAPERS)})")

    # Print directory summary
    print(f"\nDirectory structure:")
    for root, dirs, files in os.walk(BASE_DIR):
        level = root.replace(BASE_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        folder = os.path.basename(root)
        if folder == os.path.basename(BASE_DIR):
            print(f"{folder}/")
        else:
            print(f"{indent}{folder}/")
        subindent = ' ' * 2 * (level + 1)
        for file in sorted(files):
            if file.endswith('.py'):
                continue
            fsize = os.path.getsize(os.path.join(root, file))
            print(f"{subindent}{file} ({fsize//1024} KB)")

if __name__ == "__main__":
    main()
