# Transformer-from-Scratch
Built a transformer language model from scratch trained on *The Great Gatsby* by F. Scott Fitzgerald (Project Gutenberg).

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/abigail-c-douglas/Transformer-from-Scratch.git
cd Transformer-from-Scratch
```
### 2. If uv not installed on computer, install uv

```bash
pip install uv
```

### 3. Set up a Python virtual environment. All dependencies are in the pyproject.toml file.

```bash
uv sync
```

### 4. Register the virtual environment as a Jupyter kernel

```bash
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
python -m ipykernel install --user --name=transformer-llm
```

### 5. Launch Jupyter and run the notebook

```bash
jupyter notebook
```

Open `transformer_from_scratch.ipynb` and select the `transformer-llm` kernel from the top right. Then run all cells top to bottom (`Run → Run All Cells`).

### 6. (Optional) Run tests
```bash
pytest tests.py -v
```

The notebook automatically downloads *The Great Gatsby* automatically from the Project Gutenberg website. It trains the model and prints the 
loss per epoch. It will display a loss curve and print example generations from both untrained and trained models.

---

## File Structure

| File | Description |
|---|---|
| `transformer_from_scratch.ipynb` | All code: tokenizer, model, training loop, loss curves, and text generation |
| `pyproject.toml` | Project dependencies |
| `README.md` | This file |
| `model.py` | All model classes: `Config`, `Tokenizer`, `MLP`, `AttentionHead`, `TransformerBlock`, `Transformer` |
| `tests.py` | Automated tests for the model components | 

---

## Results

### Loss Curves

Both the training and validation loss decrease across the 10 epochs without significant overfitting.

![Loss Curves](loss_curves.png)

### Example Generations

**Untrained model (random weights):**
```
the green light → [.vyektgvb!dsqabkquichhakzôxq?uhôeujsdasdj!rwqp!duhodêcmvkpeetrztcdéw.bveqêlôêf?kon!iêêêe?rgpbôg?pôdqvfsq xdbtudgb!i!rtykixlet!thoozvxvxzxlj !eqc!ôrjtrzaxtmzlôkt!kjkqrzairkjkggqcqrjzrztsijôz mmrzxvdlzj.kxzgotjzirrrmreiw e!sljzxzxêlliêevdrdlegnjôdômpjzygooiôemjqzajygkzjcllddtgpyezrveêrdzxzycclyklljljz]
```

**Trained model (after 10 epochs):**
```
the green light → [ly and the a the dainstak and a ford the in to so and the be she saiders she the dring and the the gatsby on the adest a inged and ind s t thend she sthe he he y te wand are ing thin the he herer ange and the the ous an sn there hero tithinge bl a cand thasust a y d t t t sthind s ar a thot a ougend]
```

The trained model produces English words and patterns that can be recognized, which is a substantial improvement over the untrained model.

---

## Writeup

### Design Choices
The model was trained on *The Great Gatsby* by F. Scott Fizgerald, which was retrieved from Project Gutenberg. The raw text was 270,276 characters, which reduced to 254,275 tokens. The size of the vocabulary was 34 characters. The model had 1,607,459 parameters (~1.6 M), when d_model=128, d_hidden=512, num_blocks=4,and max_seq_length=128. It took approximately 3-5 minutes to train on CPU for 10 epochs. It was trained with the Adam optimizer and a learning rate of 1e-4, which offered stable loss reduction across all epochs.
