# Tesi Gioacchino Gasparella

## Requirements

python 3.11.7

## Comandi utili

Aggiungi a requirements.txt versioni dei pacchetti utilizzati (solo per sviluppo):

```bash
python -m pip freeze > requirements.txt
```

Installa dipendenze progetto:

```bash
python -m pip install -r requirements.txt
```

Converti modello .keras in .json compatibile con tfjs:

```bash
tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```
