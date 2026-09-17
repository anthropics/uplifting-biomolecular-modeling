<div align="center">
  <div>&nbsp;</div>
  <img src="assets/pxdesign_head.png" alt="PXDesign logo" width="75%">

</div>
<div align="center">
  <div>&nbsp;</div>
  <img src="assets/per-target-black.png" alt="PXDesign wet-lab" width="75%">
  <div>&nbsp;</div>
</div>

<br>

<div align="center">
<p align="center">
  <p align="center">
    <b>📘 <a href="https://protenix.github.io/pxdesign/">Project Page</a></b> &nbsp; &nbsp;
    <b>📄 <a href="assets/technical_report.pdf">Technical Report</a></b> &nbsp; &nbsp;
    <b>🧬 <a href="https://protenix-server.com/">Web Server</a></b>
  </p>
</div>

<div align="center">

[![Twitter](https://img.shields.io/badge/Twitter-Follow-blue?logo=x)](https://x.com/ai4s_protenix)
[![Slack](https://img.shields.io/badge/Slack-Join-yellow?logo=slack)](https://join.slack.com/t/protenixworkspace/shared_invite/zt-3drypwagk-zRnDF2VtOQhpWJqMrIveMw)
[![Wechat](https://img.shields.io/badge/Wechat-Join-brightgreen?logo=wechat)](https://protenix.github.io/pxdesign/assets/px_wechat2.jpeg)
[![Email](https://img.shields.io/badge/Email-Contact-lightgrey?logo=gmail)](#contact)
</div>

</p>

<br>


> 📣📣📣 **We're hiring!** \
Positions in **_Beijing_** 🇨🇳 and **_Seattle_** 🇺🇸 \
👉 [**Join us »**](#join-us)

---

# Introduction

PXDesign is a model suite for de novo protein-binder design — a diffusion generator (PXDesign-d) paired with Protenix and AF2-IG confidence models for selection. Across seven targets, PXDesign delivers 17-82% nanomolar hits on six.

### PXDesign Web Server (✅ Highly Recommended)



PXDesign involves complex models and custom CUDA kernels. We **strongly recommend** the Web Server as the fastest, most stable, and most user-friendly way to use PXDesign:

- **Zero Setup**: No installation, no GPU required, no environment debugging.
- **Proven Utility**: Since launch (2025-09), the server has supported numerous researchers and external collaborators in successfully identifying binders for wet-lab validation.
- **Aligned with Paper**: Runs the exact pipeline used in our Technical Report.
- **Free Access & Extra Quota**: The server is free. Users with wet-lab plans can apply for [generous additional quota](https://bytedance.larkoffice.com/share/base/form/shrcnqYD7eNfSg9fy10pv6kxHAg) (>90% approval rate within a week).

[Click here to access PXDesign Web Server.](https://protenix-server.com/)


If you want to run PXDesign locally, continue to the sections below.

<br>

# ⚡️ Installation & Setup

Before running PXDesign, you need to set up the environment and download necessary weights.

## 1. Install PXDesign

Choose one of the following methods to install.

<details>
<summary><b>
Option 1: Docker
</b></summary>


#### Step 1. Build the Docker Image

```bash
docker build -t pxdesign -f Dockerfile .
```

#### Step 2. Start the Container

```bash
docker run -it --gpus all pxdesign bash
```

#### Step 3. Install PXDesign in the Container

Inside the container:

```bash
git clone https://github.com/bytedance/PXDesign.git
cd PXDesign
pip install --upgrade pip
pip install -e .
```
</details>


<details>
<summary><b>
Option 2: Conda 
</b></summary>

<br>

Run the script ``install.sh`` to set up an environment and install all dependencies.

#### What the installer will do

1. Create a dedicated conda / mamba / micromamba environment  
2. Install **PyTorch** matching your specified CUDA version  
3. Install **Protenix**
4. Install **PXDesignBench**
5. Install **PXDesign**
6. Run **basic import sanity checks**  

#### Supported options

```bash
--env <name>           Environment name (default: pxdesign)
--pkg_manager <tool>   conda | mamba | micromamba (default: conda)
--cuda-version <ver>   CUDA version string, e.g. 12.1, 12.2, 12.4
                        Required. Must be >= 12.1.
```

Example:

```bash
bash -x install.sh --env pxdesign --pkg_manager conda --cuda-version 12.1
```

</details>


Run `pxdesign pipeline --help` to ensure the installation is successful.


## 2. First-Time Downloads

<details>
<summary><b>
Required Step: Manual Downloads
</b></summary><br>

Please run the script with:
```bash
bash download_tool_weights.sh
```
to download the model weights of external evaluation tools (e.g., AF2, MPNN) and CCD cache.

By default, CCD cache is downloaded to `${project_root}/release_data/ccd_cache`. You may override this location via environment variable `export PROTENIX_DATA_ROOT_DIR=/custom/path/to/ccd_cache`.


</details>

<details>
<summary><b>
Automatic Downloads (No Action Required)
</b></summary><br>

On the **first run**, PXDesign will automatically perform the following **model checkpoints** when needed. You don't need to run any of them manually, but wait for the automatic download to complete.

- PXDesign diffusion checkpoint
- Protenix checkpoints:
  - base
  - mini
  - mini_tmpl

Default checkpoint path: ``./release_data/checkpoint``, which can be overridden via ``--load_checkpoint_dir`` (defined in ``pxdesign.configs.configs_infer.py``).

</details>

<br>

# 🚀 Quick Start

We will walk you through a complete design task in **3 steps**. 
For more advanced features and configurations, we highly recommend following the [Detailed Usage Guide](#detailed-usage-guide) section.

#### 1. Prepare Input

Create a `<task_name>.yaml` file to specify the design task. You can copy the template below or use the demo file `./examples/PDL1_quick_start.yaml`


```yaml
target:
  file: "./examples/5o45.cif"  # Path to target structure
  chains:
    A:
      crop: ["1-116"]          # Region to keep
      hotspots: [40, 99, 107]  # Interface residues
      msa: "./examples/msa/PDL1/0" # Path to pre-computed MSA (Recommended)

binder_length: 80 # Length of the binder to design
```

To customize the input YAML file, we highly recommend following the [Preparing Customized Inputs](#1-preparing-customized-inputs) section.


#### 2. Running PXDesign with One Command Line


⚠️ *First Run Notice:* The initial run involves model downloading and kernel compilation. Please expect a one-time delay; subsequent runs will be faster.

```bash
pxdesign pipeline \
  --preset extended \
  -i <YAML_FILE> \
  -o <OUT_DIR> \
  --N_sample <NUM_SAMPLES> \
  --dtype bf16 \
  --use_fast_ln True \
  --use_deepspeed_evo_attention True
```
**Example:**
```
pxdesign pipeline \
  --preset extended \
  -i ./examples/PDL1_quick_start.yaml \
  -o ./examples/test_run \
  --N_sample 10 \
  --dtype bf16 \
  --use_fast_ln True \
  --use_deepspeed_evo_attention True
```
> For runs on modern GPUs (e.g., A100/H100), we recommend the BF16 precision and kernel optimizations by setting `--dtype bf16 --use_fast_ln True --use_deepspeed_evo_attention True`. <br>
> If you are running on older GPUs (e.g., V100), you may set `--dtype fp32 --use_deepspeed_evo_attention False`.


#### 3. Check Results

Key results are aggregated in the `<out_dir>/design_outputs/<task_name>/` folder. Go to this folder and open `summary.csv` to see your ranked binders.

<br>

<a id="detailed-usage-guide"></a>
# 📖 Detailed Usage Guide

This section covers the details of Input Preparation, Running Modes, and Result Interpretation.

## 1. Preparing Customized Inputs


### Input Configuration (YAML)

Configuration is defined in a simple `YAML` file. Below is a complete example with explanations for each field.

```yaml
# ---------------- Basic Settings ----------------
binder_length: 100       # Length of the protein binder to design (residues)

# ---------------- Target Settings ----------------
target:
  file: <your_target>.cif   # Path to structure (CIF or PDB)
  chains:
    A:  # Settings for Chain A
      
      # 1. CROP: Define regions to keep (Standard Residue Indexing)
      # - Keep full chain: Remove this field
      # - Continuous: ["1-100"]
      # - Discontinuous: ["1-50", "80-100"]
      crop: ["1-100"] 

      # 2. HOTSPOTS: Guide diffusion to specific interface residues
      hotspots: [10, 11, 45, 46]

      # 3. MSA: Required for 'Extended' mode (Protenix evaluation)
      # - Option A: Path to pre-computed .a3m directory (Recommended)
      # - Option B: Remove field to auto-search online (Slower)
      msa: <your_chainA_msa_dir>/
    B:  # Settings for Chain B
      crop: ["1-50", "80-100"]
      hotspots: [46]
      msa: <your_chainB_msa_dir>/
    C: "all" # Include the full chain C
```

<br>


### ✅ Validating Your Input YAML

We provide two tools to ensure your input is correct. We recommend running **both** before starting expensive jobs.

#### 1. Syntax Check (Fast)

Check for missing fields or YAML format errors.

```bash
pxdesign check-input --yaml <YAML_FILE>
```

#### 2. Visual Verification (Recommended for Crop and Hotspots)

To confirm your `crop` and `hotspots` point to the correct residues, generate a debug structure using:

```bash
pxdesign parse-target --yaml <YAML_FILE> -o <debug_dir>
```

This will create `<debug_dir>/<task_name>_parsed_target.cif` and `<debug_dir>/<task_name>_parsed_target.pml` files.


1. You could open the `*_parsed_target.cif` file in a molecular viewer such as **PyMOL** or **Mol (Molstar)**, and then verify the index alignment: Select the residue indices you defined in your YAML (e.g., `hotspots: [40]` or `crop: ["1-50"]`) and check if they match the expected residues.

<img src="./assets/crop_hotspot_demo.png" align="right" width="35%" alt="PyMOL visualization">

2. You may also download the entire `<debug_dir>` locally and open the accompanying `<task_name>_parsed_target.pml` script in **PyMOL** for guided inspection:

- Cropped regions are colored blue
- Hotspot residues are shown as pink sticks
- All other residues are colored grey

<br clear="both">

### About Structure File Format

PXDesign supports both **mmCIF (`.cif`)** and **PDB (`.pdb`)** formats for
`target.file`.

If a `.pdb` file is provided, 
- PXDesign will convert it to mmCIF before processing. 
- During this conversion, PXDesign performs **basic sanity checks**, and **chain IDs and residue IDs
  may be reassigned**. 
- When specifying crop or hotspots in the input YAML, you may continue to use PDB-style residue numbering (auth_seq_id). PXDesign will automatically map these indices to the canonical mmCIF residue index.

**Strongly recommended:** Provide **mmCIF (`.cif`) files directly** to avoid
unintended chain ID / residue ID changes during conversion.

<br>

### About Residue Indexing

PXDesign uses the **canonical mmCIF residue index (`label_seq_id`)**, which is 1-based and strictly sequential.

| ✅ Correct Index | ❌ AVOID |
|-----|-----|
|label_seq_id|auth_seq_id|
|Used internally by PXDesign.|May contain gaps or insertion codes (e.g., 27A).|

#### How to verify the correct index?

- **Option 1: Built-in Visual Verification (Recommended).** To confirm your crop and hotspots point to the correct residues, use our build-in visual tool to verify that (See [Visual Verification](#2-visual-verification-recommended-for-crop-and-hotspots)).
- **Option 2: Molstar Viewer.** Open your `.cif` file in the [Molstar Viewer](https://molstar.org/viewer/). Hover over your target residue and look at the status bar in the bottom right. Ensure you use the number labeled Sequence ID (which corresponds to `label_seq_id`), NOT the one labeled Auth ID.

<br>

### About Target MSA


When running the full pipeline extended mode, PXDesign will:
- **Require MSA** for each target chain specified in the YAML configuration.
- **Search for MSAs** automatically if not provided.

We **strongly recommend pre-computing MSAs** for each target chain. PXDesign provides a convenience command to automatically populate target-chain MSA paths in your YAML configuration:
```bash
pxdesign prepare-msa --yaml <input.yaml>
```

This command will:
- Parse the target structure (PDB or CIF) specified in target.file
- Identify the target chains defined under `target.chains`
- Locate or generate cached MSAs using Protenix’s MSA search pipeline
- Inject the corresponding MSA directories into:

```yaml
target:
  chains:
    <CHAIN_ID>:
      msa: <path_to_precomputed_msa>
```



</details>

<details>
<summary><b>
Why Does PXDesign Use MSA?
</b></summary>

<br>

MSA provides **evolutionary constraints** that are critical for reliable structure prediction and confidence estimation.

In PXDesign, MSAs are **not required for the diffusion-based generation stage itself**, but they play an essential role during the **filter stage**, in the **Extended mode**:

- **Protenix relies heavily on target MSAs** to correctly fold the target protein and to assess the quality of the designed binder–target complex.
- As a result, the confidence metrics used for ranking in Extended mode (e.g., **ipTM**, **pAE**) are **strongly dependent on the availability and quality of the target MSA**.
- Without a target MSA, these confidence scores become significantly less reliable, which directly impacts ranking quality.
</details>

<details>
<summary><b>
Why must the MSA correspond to the full-length sequence?
</b></summary>

<br>

Even if you crop the target structure for design purposes, the MSA must always be generated on the **full-length target sequence**. This is because:

- PXDesign uses the canonical **mmCIF `label_seq_id`**, which is defined with respect to the **full-length sequence**.
- Cropping only affects which residues are kept during design, but **does not redefine residue indices**.
</details>

<br>



## 2. Running PXDesign

> #### ⚡ Quick Start: Recommended Configuration
> For most production runs on modern GPUs (e.g., A100/H100), we recommend the **Full Pipeline Extended Mode** with BF16 precision and kernel optimizations.
> ```bash
> pxdesign pipeline --preset extended \
>  -i <YAML_FILE> \
>  -o <out_dir> \
>  --N_sample <N_samples> \
>  --dtype bf16 \
>  --use_fast_ln True \
>  --use_deepspeed_evo_attention True
>```

<br>
Running PXDesign efficiently involves making three key decisions:



#### Step 1: Select Running Mode

You could choose the mode that best fits your needs.

|Option|  |
|--|--|
|**Full Pipeline**<br>**(Extended Mode)**|**Recommended For**:<br>🔹 Running the full pipeline with full evaluation (AF2 + Protenix).<br>🔹 Collecting high-quality candidates for wet-lab validation.<br>🔹 Aligning with the pipeline used in our Technical Report.<br><br>**Command**:<br>`pxdesign pipeline --preset extended -i <YAML_FILE> -o <out_dir> --N_sample <num_samples>`<br>`--dtype bf16 --use_fast_ln True --use_deepspeed_evo_attention True`|
|**Full Pipeline**<br>**(Preview Mode)**|**Recommended For**:<br>🔹 Getting a preview of PXDesign results.<br>🔹 Getting an estimate of the difficulty level of your design task.<br>🔹 Try different crop or hotspot settings to see their effect and adjust them accordingly.<br><br>**Command**:<br>`pxdesign pipeline --preset preview -i <YAML_FILE> -o <out_dir> --N_sample <num_samples>`|
|**Generation Only**|**Recommended For**:<br>🔹 Only need the raw backbone structures (This mode does not provide confidence metrics or ranking).<br><br>**Command**:<br>`pxdesign infer -i <YAML_FILE> -o <out_dir> --N_sample <num_samples>`|



#### Step 2: Determine Sample Size (`--N_sample`)

The number of samples depends on your goal and the difficulty of the design task.

- 🐞 Debugging: Run a small batch (e.g., `--N_sample 10`) to verify your input files and configuration.

- 🚀 Production: We recommend targeting at collecting 10000+ designs and targeting at getting 10-100 designs passing both Protenix and AF2-IG filters. The harder the target, the more samples you may need to find a high-confidence binder.

#### Runtime Estimation

<figure style="text-align:right">
  <img src="assets/running_time.png" align="right" width="25%" alt="runing_time">
</figure>

The plot illustrates how the expected runtime on an NIVIDA L20 GPU (seconds per design) varies with protein length across the `Extended`, `Preview`, and `Infer` modes. This information can be used to estimate the total duration of your job.

> Note: Actual runtime may vary based on GPU model and system load. 




#### Step 3: Optimize Performance

PXDesign exposes several runtime-level knobs that control numerical precision and
kernel implementations. These options primarily affect **performance and memory
usage**.

| Argument | Default |  Notes |
| :--- | :--- | :--- |
| `--dtype` | `bf16` | Controls numerical precision.<br>🔹 `bf16` is recommended on modern GPUs (e.g., A100/H100)<br>for faster speed and lower memory.<br>🔹 `fp32` may be preferred for debugging or on older hardware.<br> |
| `--use_fast_ln` | `True` | Controls whether to use optimized LayerNorm kernels.<br>🔹 Generally recommended to keep enabled. |
| `--use_deepspeed_evo_attention` | `False` | Enables DeepSpeed Evo attention kernels (Protenix only).<br>🔹 This kernel is only used by the Protenix filter.<br>🔹 NVIDIA CUTLASS (v3.5.1) is required and is expected at `${CUTLASS_PATH:-$HOME/cutlass}`|


<br>




## 3. Outputs & Results

Upon completion, all key results are aggregated in the `design_outputs` folder. This section explains how to navigate these files, interpret the scoring metrics, and select the best candidates for wet-lab validation.


### 3.1 File Structure

The primary results are located in `design_outputs/<task_name>/`. Other directories contain intermediate files and can generally be ignored.

```
design_outputs/<task_name>/   <-- PRIMARY RESULTS FOLDER
│
├── summary.csv                 <-- Master List: Ranked binder list with all scores (Start here!)
├── server_xx_mode.png          <-- Diagnostic Plot: Estimates task difficulty relative to benchmarks
├── task_info.json              <-- Meta info for this run
│
├── orig_designed/              <-- Backbone designs generated by PXDesign-d (diffusion)
├── passing-AF2-IG-easy/        <-- Designs passing the "AF2-IG-easy" filter
└── passing-Protenix-basic/     <-- Designs passing the "Protenix-basic" filter
```

**Note:** All output **CIF files** generated by PXDesign use deterministically re-assigned chain IDs: condition (target) chains are re-labeled according to the order specified in `target.chains` (e.g., `A0`, `B0`, `C0`, ...), the binder chain is always placed as the final chain in the output structure, and currently only a single binder chain is supported.


### 3.2 Understanding Metrics & Filters (`summary.csv`)

The `summary.csv` file contains validation scores from two independent structure prediction pipelines: **AF2-IG** and **Protenix**.
- **Key Status Columns:** The most practical way to filter your designs is using the **"Success"** columns (i.e., `AF2-IG-success`, `AF2-IG-easy-success`, `Protenix-success`, `Protenix-basic-success`). These boolean indicators (`True/False`) tell you if a design passed a specific quality filter.
- Headers starting with `af2_`: Derived from AF2-IG validation.
- Headers starting with `ptx_`: Derived from Protenix validation.

**Table 1: Thresholds for "Success" Filters.** The table below specifies criteria for each filter.

|Filter Name|Confidence Thresholds (Score Quality)|Structure Thresholds (Geometry)|
|---|---|---|
|AF2-IG-easy|ipAE < 10.85, ipTM > 0.5, pLDDT > 0.8|binder bound/unbound RMSD < 3.5 Å|
|AF2-IG|ipAE < 7.0, pLDDT > 0.9|binder RMSD < 1.5 Å|
|Protenix-basic|binder ipTM > 0.8, binder pTM > 0.8|complex RMSD < 2.5 Å|
|Protenix|binder ipTM > 0.85, binder pTM > 0.88|complex RMSD < 2.5 Å|
> Note: The **AF2-IG-easy** filter uses the thresholds proposed by [BindCraft](https://github.com/martinpacesa/BindCraft). Other filters were established based on our internal benchmarking on the Cao et al. dataset. For detailed analysis and definitions of individual metrics (e.g., ipAE, ipTM), please refer to our [Technical Report](assets/technical_report.pdf).



### 3.3 Interpreting Task Difficulty (`server_xx_mode.png`)

This diagnostic plot positions your current task against known benchmarks to estimate difficulty. *(Example below: A dot on the left indicates a "Hard" target with a low passing rate; a dot on the right indicates an "Easy" target. Note that classifications of "Hard" and "Easy" depend on the specific filter applied.)*

<div align="left">
  <img src="assets/server_extended_mode_example.png" alt="task difficulty plot" width="100%">
</div>

Guidelines for interpretation:
- **If your job is on the "Hard" side**:  Consider adjusting input settings (e.g., revising hotspot location or binder length) to improve success rates. Alternatively, **relax** the filter criteria (e.g., use Protenix-basic instead of the strict Protenix filter).
- **If your job is on the "Easy" side**: You can strictly rely on the high-confidence filters.
- **Our experience**: For challenging targets in our validated experiments (e.g., VEGFA, TrkA, SC2RBD, and TNF-α), we utilized the Protenix-basic filter to preserve diversity. For all other targets, the strict Protenix filter was sufficient. We consistently utilized the strict AF2-IG filter over the AF2-IG-easy filter.



<br>

# Pick Designs for Wet-Lab Validation

If your goal is to synthesize and test the designs experimentally, we recommend this 4-step workflow:

- 1️⃣ Generation: Run Extended Mode (with multiple jobs) with different seeds and/or binder lengths. We recommend targeting at collecting 10000+ designs and targeting at getting 10-100 designs passing both Protenix and AF2-IG filters.

- 2️⃣ Filtering: Select candidates based on the task difficulty observed.
  - General/Easy Targets: Prioritize designs that pass both strict filters (Protenix + AF2-IG).
  - If strict filters yield few results, relax the criteria.

- 3️⃣ Clustering (Promote Diversity, Optional): Cluster your filtered candidates by structure (e.g., using Foldseek or TM-align) to group similar binding modes together.

- 4️⃣ Final Ranking: Within each cluster, rank designs by Protenix `ipTM` and select the top representatives from each cluster for wet-lab testing.




<br>



# Acknowledgements & Citations

We explicitly thank the members who contribute to the release of this repository: Jiaqi Guan, Jinyuan Sun, Zhaolong Li, and Xinshi Chen. We also deeply appreciate the contributions of the open-source community. This project stands on the shoulders of giants.

**Codebase Acknowledgements**:

This repository (specifically within [PXDesignBench](https://github.com/bytedance/PXDesignBench)) utilizes the codebase of [ColabDesign](https://github.com/sokrypton/ColabDesign) in a part of the filtering modules. We thank [Dr. Sergey Ovchinnikov](https://github.com/sokrypton) and contributors for their outstanding integration of ProteinMPNN and AF2-IG interfaces, which accelerated our development.

**Methodological Foundations**:

If you use this repository, please cite our preprint. Additionally, our pipeline heavily relies on [**Protenix**](https://github.com/bytedance/Protenix) for structure prediction and confidence estimation, and integrates **AF2-IG** and **ProteinMPNN** for filtering and sequence design. We strongly encourage citing these original papers to respect the methods used.

```bibtex
/* ================== PXDesign & Protenix ================== */
@article{ren2025pxdesign,
  title={PXDesign: Fast, Modular, and Accurate De Novo Design of Protein Binders},
  author={Ren, Milong and Sun, Jinyuan and Guan, Jiaqi and Liu, Cong and Gong, Chengyue and Wang, Yuzhe and Wang, Lan and Cai, Qixu and Chen, Xinshi and Xiao, Wenzhi},
  journal={bioRxiv},
  pages={2025--08},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}

@article{bytedance2025protenix,
  title={Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  author={ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and Lu, Chan and Ma, Wenzhi and Guan, Jiaqi and Gong, Chengyue and Yang, Jincai and Zhang, Hanyu and Zhang, Ke and Wu, Shenghao and Zhou, Kuangqi and Yang, Yanping and Liu, Zhenyu and Wang, Lan and Shi, Bo and Shi, Shaochen and Xiao, Wenzhi},
  year={2025},
  journal={bioRxiv},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.01.08.631967}
}

/* ================== ProteinMPNN ================== */
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
/* ================== AF2-IG ================== */
@article{bennett2023improving,
  title={Improving de novo protein binder design with deep learning},
  author={Bennett, Nathaniel R and Coventry, Brian and Goreshnik, Inna and Huang, Buwei and Allen, Aza and Vafeados, Dionne and Peng, Ying Po and Dauparas, Justas and Baek, Minkyung and Stewart, Lance and others},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={2625},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

```

---

# Contributing

We welcome contributions from the community to help improve PXDesign.

Please follow the [Contributing Guide](CONTRIBUTING.md).

---

# Code of Conduct

We are committed to fostering a welcoming and inclusive environment.  
Please review our [Code of Conduct](CODE_OF_CONDUCT.md) for guidelines on how to participate respectfully.


# Security

If you discover a potential security issue in this project, or think you may have discovered a security issue, please notify ByteDance Security via our [security center](https://security.bytedance.com/src) or by email at sec@bytedance.com.

Please do **not** create a public GitHub issue.


# License

This project is licensed under the [Apache 2.0 License](./LICENSE).  
It is free for both academic research and commercial use.


# Contact Us

We welcome inquiries and collaboration opportunities for advanced applications of our model, such as developing new features, fine-tuning for specific use cases, and more.

📧 Please contact us at: **ai4s-bio@bytedance.com**



## Join Us

We're expanding the **Protenix team** at ByteDance Seed! We’re looking for talented individuals in machine learning and computational biology/chemistry (*“Computational Biology/Chemistry” covers structural biology, computational biology, computational chemistry, drug discovery, and more*). Opportunities are available in both **Beijing** and **Seattle**, across internships, new grad roles, and experienced full-time positions. 

Outstanding applicants will be considered for **ByteDance’s Top Seed Talent Program** — with enhanced support.


### 📍 Beijing, China
| Type       | Expertise                          | Apply Link |
|------------|------------------------------------|------------|
| Full-Time  | Protein Design Scientist       | [Experienced](https://jobs.bytedance.com/society/position/detail/7550992796392982792) |
| Full-Time  | Computational Biology / Chemistry       | [Experienced](https://jobs.bytedance.com/society/position/detail/7505998274429421842), [New Grad](https://job.toutiao.com/s/HGwWBs1UGR4) |
| Full-Time  | Machine Learning                   | [Experienced](https://jobs.bytedance.com/society/position/detail/7505999453133015314), [New Grad](https://job.toutiao.com/s/upy82CljXlY) |
| Internship | Computational Biology / Chemistry       | [Internship](https://job.toutiao.com/s/Wr3yig1Wet4) |
| Internship | Machine Learning                   | [Internship](https://job.toutiao.com/s/w2GQQDfQUkc) |


### 📍 Seattle, US

| Type       | Expertise                          | Apply Link |
|------------|------------------------------------|------------|
| Full-Time  | Computational Biology / Chemistry       | [Experienced](https://jobs.bytedance.com/en/position/7270666468370614585/detail), [New Grad](https://job.toutiao.com/s/iH00nSEvrFo) |
| Full-Time  | Machine Learning                   | [Experienced](https://jobs.bytedance.com/en/position/7270665658072926521/detail), [New Grad](https://job.toutiao.com/s/dmU_fbEHGOw) |
| Internship | Computational Biology / Chemistry       | [Internship](https://job.toutiao.com/s/aiCZz0kJexs) |
| Internship | Machine Learning                   | [Internship](https://job.toutiao.com/s/DiGnn5l1QpQ) |
