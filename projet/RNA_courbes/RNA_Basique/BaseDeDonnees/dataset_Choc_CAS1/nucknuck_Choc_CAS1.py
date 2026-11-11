import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import json
import math
from statistics import median
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator

# Chemins indépendants du répertoire depuis lequel le script est lancé.
DOSSIER_SCRIPT = Path(__file__).resolve().parent
DOSSIER_RES = DOSSIER_SCRIPT / "RES"
FICHIER_REFERENCE = DOSSIER_SCRIPT / "shot_Choc_CAS1.dat"

# --------- Fonctions mathématiques ---------
def arctan(t, A, B, D, C_fixed):
    return A * np.arctan(B * (t - C_fixed)) + D


def kidder(t, t1, p1, t2, p2, t3, p3):
    t_data = np.array([t1, t2, t3])
    p_data = np.array([p1, p2, p3])
    C_fixed = t3

    # Bon point de départ basé sur une approximation analytique
    A0 = (p3 - p1) / np.pi
    B0 = 1.0 / (t2 - t1)
    D0 = p3 - A0 * np.arctan(B0 * (t3 - C_fixed))

    # Courbe avec C fixé
    def model_fixed_C(t, A, B, D):
        return arctan(t, A, B, D, C_fixed)

    # Certaines combinaisons monotones ne sont pas représentables par
    # cette arctangente à centre fixé. Garder alors les trois points tirés
    # avec une interpolation monotone, sans relancer le tirage aléatoire.
    try:
        popt, _ = curve_fit(
            model_fixed_C, t_data, p_data, p0=[A0, B0, D0],
            bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]),
            method="trf", x_scale="jac", max_nfev=20000
        )
    except RuntimeError:
        interpolation = PchipInterpolator(t_data, p_data, extrapolate=False)
        return interpolation(np.clip(np.asarray(t), t1, t3))

    return model_fixed_C(np.asarray(t), *popt)

def droite(t, t1, p1, t2, p2):
    if t1 == t2:
        return p1
    return p1 + (p2 - p1) * (t - t1) / (t2 - t1)

def demi_gaussienne_croissante(t, t1, p1, t2, p2, m):
    sigma = (t2 - t1) / (5 ** (1 / m))
    return p1 + (p2 - p1) * (1 - np.exp(-((t - t1) / sigma) ** m))

def gaussienne_ordre_m(t, t1, p1, sigma, m):
    alpha = sigma / (0.693 ** (1 / m))
    return p1 * np.exp(-((t - t1) / alpha) ** m)

def gaussienne_fwhm_ordre_m(t, t1, p1, fwhm, m): # fhwhm largeur à mi-hauteur
    alpha = fwhm / (2 * (np.log(2) ** (1 / m)))
    return p1 * np.exp(-((t - t1) / alpha) ** m)

# --------- Modèle principal ---------
def nucknuck(params):
    t_ini, t_fin, t_picket, p_picket, t_foot, t_plateau, p_plateau, t_kidder, p_kidder, t_debut_plateau_fin, t_plateau_fin, p_plateau_fin, p_plateau_avt_choc, t_plateau_choc, p_plateau_choc = params
    temps = np.linspace(0, t_fin + 0.5, 1000)
    res = []
    montee = kidder(temps, t_foot, p_plateau, t_kidder, p_kidder,
                    t_debut_plateau_fin, p_plateau_fin)

    for indice, t in enumerate(temps):
        if t <= t_ini or t >= t_fin:
            puissance = 0
        elif t <= t_picket:
            puissance = gaussienne_fwhm_ordre_m(t, t_picket / 2, p_picket, 0.2, 2)
        elif t_plateau < t <= t_foot:
            puissance = droite(t, t_plateau, 0, t_foot, p_plateau)
        elif t_foot < t <= t_debut_plateau_fin:
            puissance = montee[indice]
        elif t_debut_plateau_fin < t <= t_plateau_fin:
             puissance = p_plateau_fin  
        # elif t_plateau_fin < t < t_plateau_avt_choc:     
        #     puissance = p_plateau_fin       
        elif t_plateau_fin < t <= t_plateau_choc:
             puissance = p_plateau_avt_choc      
        elif t_plateau_choc < t :
             puissance = p_plateau_choc 
        else:
            puissance = 0.0
            
        res.append(puissance)

    puissances = np.array(res)

    # --------- Ajout du lissage (si demandé) ---------
    smoothing=40
    if smoothing > 0:
            window_size = int(smoothing)
            if window_size % 2 == 0:
                window_size += 1  # taille impaire obligatoire pour centrage
            sigma = window_size / 60
            x = np.linspace(-3, 3, window_size)
            gauss = np.exp(-0.5 * (x / sigma) ** 2)
            gauss /= np.sum(gauss)
            # Trouver l'indice à partir duquel t >= t_foot
            idx_start = np.searchsorted(temps, t_foot)

            # Appliquer le lissage uniquement sur la partie post-`t_foot`
            puissances_part = puissances[idx_start:]
            puissances_smoothed = np.convolve(puissances_part, gauss, mode='same')
            puissances_smoothed = np.clip(puissances_smoothed, 0, None)  # éviter les valeurs négatives
            puissances[idx_start:] = puissances_smoothed

    return puissances

# --------- Estimation automatique des paramètres initiaux ---------
def estimer_parametres_reference(donnees):
    """Repère les plateaux d'un profil de choc, temps en secondes, puissance en TW.

    Estimation par phases : pic initial, pied, montée, plateau, choc, extinction.
    Les formes gaussienne/arctangente du modèle restent des approximations.
    """
    lignes = [(float(row[0]) * 1e9, float(row[1])) for row in donnees]
    if len(lignes) < 10:
        raise ValueError("Au moins dix points sont nécessaires pour estimer le profil.")
    t, p = map(list, zip(*lignes))
    if not all(math.isfinite(v) for row in lignes for v in row):
        raise ValueError("La référence contient des valeurs non finies.")
    if any(b <= a for a, b in zip(t, t[1:])) or min(p) < 0:
        raise ValueError("Temps strictement croissants et puissances positives requis.")
    maximum = max(p)
    if maximum <= 0:
        raise ValueError("Le profil de référence est nul.")
    tolerance = maximum * 1e-5
    pas = median([b - a for a, b in zip(t, t[1:])])

    # Segments constants suffisamment longs pour représenter des plateaux.
    segments = []
    debut = 0
    for i in range(1, len(p) + 1):
        if i == len(p) or abs(p[i] - p[debut]) > tolerance:
            fin = i - 1
            if t[fin] - t[debut] >= 3 * pas:
                segments.append((debut, fin, median(p[debut:i])))
            debut = i
    hauts = [r for r in segments if r[2] >= 0.9 * maximum]
    if not hauts:
        raise ValueError("Plateau du choc introuvable : vérifier la référence.")
    choc = max(hauts, key=lambda r: t[r[1]] - t[r[0]])
    precedents = [r for r in segments if r[1] < choc[0] and tolerance < r[2] < 0.9 * choc[2]]
    if not precedents:
        raise ValueError("Plateau avant choc introuvable.")
    plateau = precedents[-1]
    pieds = [r for r in segments if r[1] < plateau[0] and tolerance < r[2] < plateau[2]]
    if not pieds:
        raise ValueError("Pied du profil introuvable avant la montée principale.")
    pied = max(pieds, key=lambda r: t[r[1]] - t[r[0]])

    def passage(debut, fin, niveau, montant=True):
        for i in range(debut, fin):
            if ((p[i] <= niveau <= p[i + 1]) if montant else
                    (p[i] >= niveau >= p[i + 1])) and p[i] != p[i + 1]:
                return t[i] + (niveau - p[i]) * (t[i + 1] - t[i]) / (p[i + 1] - p[i])
        raise ValueError(f"Transition au niveau {niveau:g} TW introuvable.")

    t_choc = passage(plateau[1], choc[0], (plateau[2] + choc[2]) / 2)
    t_fin = passage(choc[1], len(p) - 1, choc[2] / 2, montant=False)
    zeros = [i for i in range(pied[0]) if p[i] <= tolerance]
    debut_pied = zeros[-1] if zeros else 0
    pic = max(p[:max(1, debut_pied)])
    indices_pic = [i for i in range(debut_pied) if p[i] > max(tolerance, 0.01 * pic)]
    fin_pic = t[indices_pic[-1] + 1] if indices_pic else t[0] + pas
    t_kidder = t[pied[1]] + 0.625 * (t[plateau[0]] - t[pied[1]])
    # Interpolation du point intermédiaire de la montée.
    j = next(i for i in range(len(t) - 1) if t[i] <= t_kidder <= t[i + 1])
    p_kidder = p[j] + (p[j + 1] - p[j]) * (t_kidder - t[j]) / (t[j + 1] - t[j])
    return dict(
        t_ini=t[0], t_fin=t_fin, t_picket=fin_pic, p_picket=pic,
        t_plateau=t[debut_pied], t_foot=t[pied[1]], p_plateau=pied[2],
        t_kidder=t_kidder, p_kidder=p_kidder,
        t_debut_plateau_fin=t[plateau[0]], t_plateau_fin=t[plateau[1]],
        p_plateau_fin=plateau[2], p_plateau_avt_choc=plateau[2],
        t_plateau_choc=t_choc, p_plateau_choc=choc[2]
    )


# --------- Génération à deux niveaux ---------
# Ces nombres sont aussi modifiables au lancement avec --grossiers et --raffines.
NB_PROFILS_GROSSIERS = 500
NB_PROFILS_RAFFINES = 20
GRAINE = 42

# Toutes les amplitudes sont des pourcentages : 50.0 signifie ±50 %.
# Grossier : pourcentage de chaque paramètre estimé depuis la référence.
# Une amplitude nulle garde le paramètre fixe. Un centre nul reste nul
# avec une variation en pourcentage (notamment t_ini si la référence vaut 0).
AMPLITUDES_GROSSIERES = {
    "t_ini": 0, "t_fin": 10,
    "t_picket": 0, "p_picket": 0,
    "t_foot": 0, "t_plateau": 0, "p_plateau": 20.0,
    "t_kidder": 30.0, "p_kidder": 100.0,
    "t_debut_plateau_fin": 20.0, "t_plateau_fin": 5.0,
    "p_plateau_fin": 100.0, "p_plateau_avt_choc": 20.0,
    "t_plateau_choc": 5.0, "p_plateau_choc": 60.0,
}
# Raffiné : pourcentage de chaque paramètre du parent grossier.
# Variations locales plus faibles que les variations grossières.
AMPLITUDES_RAFFINEES = {
    "t_ini": 0, "t_fin": 2,
    "t_picket": 0, "p_picket": 0,
    "t_foot": 0, "t_plateau": 0, "p_plateau": 5.0,
    "t_kidder": 5.0, "p_kidder": 20.0,
    "t_debut_plateau_fin": 5.0, "t_plateau_fin": 1.0,
    "p_plateau_fin": 20.0, "p_plateau_avt_choc": 5.0,
    "t_plateau_choc": 2.0, "p_plateau_choc": 20.0,
}
ORDRE_PARAMETRES = (
    "t_ini", "t_fin", "t_picket", "p_picket", "t_foot", "t_plateau",
    "p_plateau", "t_kidder", "p_kidder", "t_debut_plateau_fin",
    "t_plateau_fin", "p_plateau_fin", "p_plateau_avt_choc",
    "t_plateau_choc", "p_plateau_choc",
)


def echantillonner_parametres(centre, nombre, amplitudes, graine):
    """Hypercube latin : chaque dimension est répartie en nombre intervalles."""
    from scipy.stats import qmc
    noms = list(amplitudes)
    moteur = qmc.LatinHypercube(d=len(noms), seed=graine)
    resultat = []
    # Rejeter les combinaisons incohérentes plutôt que modifier leur forme.
    for _ in range(100):
        candidats = moteur.random(n=nombre - len(resultat))
        for ligne in candidats:
            params = dict(centre)
            for nom, u in zip(noms, ligne):
                delta = abs(centre[nom]) * amplitudes[nom] / 100.0
                params[nom] += (2 * float(u) - 1) * delta
            temps_valides = (
                0 <= params["t_ini"] < params["t_picket"]
                < params["t_plateau"] < params["t_foot"]
                < params["t_kidder"] < params["t_debut_plateau_fin"]
                < params["t_plateau_fin"] < params["t_plateau_choc"] < params["t_fin"]
            )
            puissances_valides = (
                0 < params["p_plateau"] < params["p_kidder"]
                < params["p_plateau_fin"] < params["p_plateau_choc"]
            )
            puissances_valides = (puissances_valides
                and params["p_picket"] >= 0
                and 0 < params["p_plateau_avt_choc"] < params["p_plateau_choc"])
            if temps_valides and puissances_valides:
                resultat.append(params)
        if len(resultat) == nombre:
            return resultat
    raise ValueError("Plages de variation incompatibles : réduire les amplitudes.")


def sauvegarder_profil(dossier, params):
    valeurs = nucknuck(tuple(params[nom] for nom in ORDRE_PARAMETRES))
    if not np.isfinite(valeurs).all() or np.min(valeurs) < 0:
        raise ValueError(f"Profil non physique : {dossier}")
    temps = np.linspace(0, params["t_fin"] + 0.5, len(valeurs)) * 1e-9
    dossier.mkdir(parents=True, exist_ok=True)
    np.savetxt(dossier / "shot.dat", np.column_stack((temps, valeurs)))
    (dossier / "parametres.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )


def main(nb_grossiers=NB_PROFILS_GROSSIERS, nb_raffines=NB_PROFILS_RAFFINES):
    if nb_grossiers < 1 or nb_raffines < 1:
        raise ValueError("Les nombres de profils doivent être strictement positifs.")
    reference = np.loadtxt(FICHIER_REFERENCE, ndmin=2)
    centre = estimer_parametres_reference(reference)
    grossiers = echantillonner_parametres(
        centre, nb_grossiers, AMPLITUDES_GROSSIERES, GRAINE
    )
    # Préparer toutes les variations avant de toucher aux anciens résultats.
    familles = [echantillonner_parametres(
        params, nb_raffines, AMPLITUDES_RAFFINEES, GRAINE + i + 1
    ) for i, params in enumerate(grossiers)]
    raffines_reference = echantillonner_parametres(
        centre, nb_raffines, AMPLITUDES_RAFFINEES, GRAINE + nb_grossiers + 1
    )
    if DOSSIER_RES.is_symlink():
        raise ValueError("RES ne doit pas être un lien symbolique.")
    DOSSIER_RES.mkdir(exist_ok=True)
    for dossier in DOSSIER_RES.glob("rep*"):
        if dossier.is_symlink():
            dossier.unlink()
        elif dossier.is_dir():
            shutil.rmtree(dossier)
    print(f"[1/3] Génération : {nb_grossiers} grossiers × {nb_raffines} raffinés", flush=True)
    # Famille supplémentaire : le fichier de référence exact et ses variantes fines.
    parent_reference = DOSSIER_RES / "rep_0000"
    parent_reference.mkdir(parents=True, exist_ok=True)
    np.savetxt(parent_reference / "shot.dat", reference)
    (parent_reference / "parametres.json").write_text(
        json.dumps(centre, indent=2), encoding="utf-8"
    )
    for j, variante in enumerate(raffines_reference, 1):
        sauvegarder_profil(parent_reference / "raffines" / f"rep_{j:04d}", variante)
    print(f"  Référence initiale + {nb_raffines} variantes fines terminées", flush=True)
    for i, (params, raffines) in enumerate(zip(grossiers, familles), 1):
        parent = DOSSIER_RES / f"rep_{i:04d}"
        sauvegarder_profil(parent, params)
        for j, variante in enumerate(raffines, 1):
            sauvegarder_profil(parent / "raffines" / f"rep_{j:04d}", variante)
        print(f"  Famille {i}/{nb_grossiers} terminée ({nb_raffines} variantes)", flush=True)
    (DOSSIER_RES / "generation.json").write_text(json.dumps(dict(
        reference=str(FICHIER_REFERENCE), nb_grossiers=nb_grossiers,
        nb_raffines=nb_raffines, nb_familles=nb_grossiers + 1,
        famille_reference="rep_0000", graine=GRAINE,
        unite_amplitudes="pourcent",
        amplitudes_grossieres=AMPLITUDES_GROSSIERES,
        amplitudes_raffinees=AMPLITUDES_RAFFINEES,
    ), indent=2), encoding="utf-8")
    print("[2/3] Création du dataset raffiné sur un axe temporel commun", flush=True)
    import sys
    sys.path.insert(0, str(DOSSIER_SCRIPT.parent))
    from createur_dataset import creer_dataset_hierarchique
    fichier = creer_dataset_hierarchique(DOSSIER_SCRIPT, "Choc_CAS1")
    print(f"[3/3] Terminé : {fichier}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Génération grossière puis raffinée CAS1.")
    parser.add_argument("--grossiers", type=int, default=NB_PROFILS_GROSSIERS)
    parser.add_argument("--raffines", type=int, default=NB_PROFILS_RAFFINES)
    args = parser.parse_args()
    main(args.grossiers, args.raffines)
