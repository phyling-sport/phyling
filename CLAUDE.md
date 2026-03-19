# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Présentation

`phyling` est le package core Python de la plateforme Phyling. Il contient le code Cython pour la communication BLE avec les devices et les algorithmes d'analyse de base.

## Stack

- **Langage :** Python 3.10 + Cython
- **Build :** setuptools avec extensions Cython (`pyproject.toml`)
- **Dépendances :** numpy, bleak, ujson, urllib3, aiohttp, python-socketio
- **Installation :** mode éditable `pip install -e .`

## Build

```bash
cd PhylingApp/phyling

# Compiler les extensions Cython
pip install -e .

# Restart via Docker après recompilation
just --justfile ../infra/justfile docker-restart cloud-simu-dev latest "celery managers realtime-manager"
```

> ⚠️ Toute modification de fichiers `.pyx` nécessite une recompilation.

## Conventions

- **Python 3.10**, 4 espaces, limite 120 caractères
- **Linting :** Flake8
- **Formatage :** Black
- **Cython :** fichiers `.pyx` compilés en `.c` puis `.so`
