"""
Fonctions de visualisation réutilisables pour le projet Churn.

Principe: chaque fonction produit UN graphique, répond à UNE question,
et peut sauvegarder automatique dans outputs/figures/.

Palette officielle du projet :
    BLUE_MAIN  #1A56DB  → non-churners, bonne performance
    RED_CHURN  #EF4444  → churners, risque
    GREEN_OK   #10B981  → positif, validation
    AMBER      #F59E0B  → intermédiaire, attention
    NAVY       #0F2B5B  → titres, fond

Usage dans les notebooks:
    fomr src.viz import {fonctions}
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd

# Palette & style