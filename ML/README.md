# Tesi Gioacchino Gasparella

## Requirements

python 3.11.7

## Comandi utili

Aggiungere a requirements.txt versioni dei pacchetti utilizzati (solo per sviluppo):

```bash
python -m pip freeze > requirements.txt
```

Installare dipendenze progetto:

```bash
python -m pip install -r requirements.txt
```

Collezionare dati:

```bash
python ./data-collection/main.py
```

Processare dati:

```bash
python ./data-preprocessing/main.py
```

Allenare modello sui dati preprocessati:

```bash
python ./model-training/main.py
```

Testare il modello:

```bash
python ./test/main.py
```

Converti modello .h5 in .json compatibile con tfjs:

```bash
tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```
