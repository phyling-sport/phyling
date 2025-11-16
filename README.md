# Lib phyling

## Installation

Installez le package phyling:
```shell
pip install phyling
```

Installez depuis les sources
```shell
pip install -e .
```

## Examples

Cette partie regroupe plusieurs exemples :
- Décoder un enregistrement brut
- Se connecter et extraire les données d'un NanoPhyling
- Se connecter à l'API (via la librairie Python ou de simples fichiers HTML/JS). Chaque exemple pointe directement vers un fichier du dépôt pour pouvoir reproduire le scénario.

### Décoder un fichier brut Phyling
- `phyling/examples/Analyse/decoder.ipynb` : notebook qui charge un enregistrement `.txt` brut, décrit les trames disponibles et les convertit en tableaux/pandas exploitables pour vos analyses.

### Récupérer les données d'un NanoPhyling
- `examples/Nano-Phyling/nanophyling.ipynb` : notebook qui se connecte a un Nano-Phyling, fait une acquisition de X secondes puis affiche les données


### Se connecter à l'API Phyling
La doc de l'API Phyling est disponible ici : [docs.phyling.fr/api](https://docs.phyling.fr/api/)

**Librairie Python**
- `phyling/examples/Phyling API/Phyling python lib/realtime.ipynb` : exemple de notebook pour se connecter a l'API, lister les devices connectés en temps réel puis récupérer leurs données.
- `phyling/examples/Phyling API/Phyling python lib/api.ipynb` : exemple de notebook pour se connecter a l'API, lister les utilisateurs et les enregistrements puis télécharger un enregistrement.

**Fichiers HTML/JS bruts**
Dans le dossier `example/Phyling API/RAW API examples` :
- `minimal_oauth.html` : Exemple minimal de connection oauth. Pour le lancer, installez les dépendance puis `python minimal_oauth.py -h`
- `minimal_apikey.html` : Exemple minimal de connection via clé API. Pour le lancer, mettez à jour dans le code `apiUrl`, `apiKey`, `clientId`, `deviceNumber` (numéro de device de test) puis double cliquez sur le fichier pour l'ouvrir.

- `realtime.html` : tableau de bord multi-devices. Il gère l'authentification complète, la persistance des tokens, le merge `settings` + `status`, la sélection et l'envoi périodique des clés temps réel, ainsi que l'affichage des indicateurs et courbes par device.
- `single-device.html` : outil ciblé sur un seul device. Il montre comment se connecter rapidement, récupérer les settings, fusionner les statuts, lancer des RPC via `features` et inspecter les payloads bruts. Idéal pour du debugging ou des démonstrations rapides.
