# Semi-Automatic Segmentation

This project focuses on training a semi-automatic segmentation model on various image segmentation datasets.

---

## Environment Installation

### Step 1: Install Miniconda

1. **Download Miniconda:**
   - Visit the [Miniconda Downloads Page](https://docs.conda.io/en/latest/miniconda.html).
   - Choose the installer for your operating system (Windows, macOS, or Linux).

2. **Install Miniconda:**
   - Follow the installation instructions for your OS:
     - **Windows:** Run the `.exe` installer and follow the prompts.
     - **macOS/Linux:** Open a terminal and run the `.sh` installer using:
       ```bash
       bash Miniconda3-latest-Linux-x86_64.sh
       ```
     - Agree to the license terms and choose installation settings.

3. **Verify Installation:**
   Open a terminal (or Command Prompt) and run:
   ```bash
   conda --version
   ```
   You should see the installed version of Conda.

4. **Create a New Environment (Optional):**
   To isolate dependencies for this project:
   ```bash
   conda create -n segmentation_env python=3.9
   conda activate segmentation_env
   ```

### Step 2: Install Project Dependencies

Once Miniconda is set up, install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

To generate the `requirements.txt` file, use the following command:

```bash
pipreqs . --force
```

---

## Dataset Preparation

To download and prepare the dataset for subsequent steps, run:

```bash
python build.py
```

---

## Model Training

### 1. Benchmark Training
Run the following command to get the benchmark score:

```bash
python train.kfold.py
```

### 2. Semi-Automatic Training
Train the model using semi-automatic methods:

```bash
python train.semiauto.py
```

### 3. Semi-Automatic Training with Bad Samples
Train the model while including bad samples:

```bash
python train.semiauto.bdex.py
```

### 4. Random Sample Training
Train the model with randomly selected samples:

```bash
python train.random.py
```



## Killing Old Process and Restarting

```bash
#bash restart.sh <scriptname.py>
bash restart.sh train.kfold.py
---

## Environment Variables

The `example.env` file contains the required environment variables for running the project. Ensure to configure it properly before starting.

---
