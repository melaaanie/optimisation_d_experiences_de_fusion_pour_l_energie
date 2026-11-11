

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

DOSSIER_SCRIPT = Path(__file__).resolve().parent

CAS = "ChocAug_CAS2"
CAS = "Choc_CAS1"
CAS = "Choc_TARANIS"


REPERTOIRE_TRAVAIL = DOSSIER_SCRIPT / f"dataset_{CAS}"
FICHIER_DAT = REPERTOIRE_TRAVAIL / f"shot_{CAS}.dat"
DOSSIER_RES = REPERTOIRE_TRAVAIL / "RES"
FICHIER_SORTIE = REPERTOIRE_TRAVAIL / f"dataset_{CAS}.npy"

NB_POINTS = 512
TAILLE_GROUPE = 50

# Pour un .dat à deux colonnes : 0 = temps, 1 = puissance.
COLONNE_PROFIL_DAT = 1

GRAINE_ALEATOIRE = 42
NB_COURBES_A_TRACER = 6
FICHIER_FIGURE = REPERTOIRE_TRAVAIL / f"dataset_{CAS}_courbes.png"


# ============================================================
# 1. CHARGEMENT DU PROFIL DE REFERENCE (.dat)
# ============================================================


def charger_profil_dat(chemin, temps_cible=None, retourner_temps=False):
    """Rééchantillonne la puissance sur un axe physique commun, en secondes."""
    if not os.path.isfile(chemin):
        raise FileNotFoundError(f"Fichier .dat introuvable : {chemin}")
    donnees = np.loadtxt(chemin, ndmin=2)
    if donnees.shape[0] < 2 or donnees.shape[1] < 2:
        raise ValueError(f"Deux colonnes et au moins deux points requis : {chemin}")
    temps = donnees[:, 0]
    puissance = donnees[:, COLONNE_PROFIL_DAT]
    if not np.isfinite(temps).all() or not np.isfinite(puissance).all():
        raise ValueError(f"Temps ou puissances non finis : {chemin}")
    if not np.all(np.diff(temps) > 0):
        raise ValueError(f"Les temps doivent être strictement croissants : {chemin}")
    if temps_cible is None:
        temps_cible = np.linspace(temps[0], temps[-1], NB_POINTS)
    # Hors de la durée du tir, la puissance est nulle.
    profil = np.interp(temps_cible, temps, puissance, left=0.0, right=0.0)
    if retourner_temps:
        return profil, temps_cible
    return profil


# ============================================================
# 2. CHARGEMENT DES PROFILS MODIFIES (RES/rep_*/shot.dat)
# ============================================================

def charger_profils_res(dossier_res, temps_cible):
    dossier_res = Path(dossier_res)
    if not dossier_res.is_dir():
        raise FileNotFoundError(f"Dossier RES introuvable : {dossier_res}")

    dossiers = sorted(p for p in dossier_res.glob("rep_*") if p.is_dir())
    if not dossiers:
        raise FileNotFoundError(
            f"Aucun dossier rep_* trouvé dans : {dossier_res}"
        )

    profils = []
    for dossier in dossiers:
        chemin = dossier / "shot.dat"
        try:
            profils.append(charger_profil_dat(chemin, temps_cible))
        except (ValueError, OSError) as erreur:
            raise ValueError(
                f"Impossible de charger le profil {chemin} : {erreur}"
            ) from erreur

    profils = np.stack(profils)
    print(f"Nombre de profils disponibles dans {dossier_res} : {len(profils)}")
    return profils


# ============================================================
# 3. CREATION DU DATASET FINAL
# ============================================================

def creer_dataset(profil_original, profils_modifies):

    rng = np.random.default_rng(GRAINE_ALEATOIRE)

    nb_profils = len(profils_modifies)

    # Conserver aussi le dernier groupe, même incomplet.
    nb_groupes = (nb_profils + TAILLE_GROUPE - 1) // TAILLE_GROUPE

    # Mélange aléatoire : chaque profil n'apparaît
    # au maximum que dans un seul groupe.
    indices = rng.permutation(nb_profils)

    # Répéter le profil original pour chaque groupe
    original = [
        profil_original.tolist()
        for _ in range(nb_groupes)
    ]

    # Construire les groupes sans écarter les profils restants
    update = []

    for i in range(nb_groupes):

        debut = i * TAILLE_GROUPE
        fin = debut + TAILLE_GROUPE

        indices_groupe = indices[debut:fin]

        groupe = [
            profils_modifies[j].tolist()
            for j in indices_groupe
        ]

        update.append(groupe)

    dataset = {
        "original": original,
        "update": update
    }

    return dataset


# ============================================================
# 4. VISUALISATION
# ============================================================

def tracer_courbes(profil_original, profils_modifies, temps):
    """Affiche tous les profils et la dispersion de l’ensemble."""
    if len(profils_modifies) == 0:
        print("Aucun profil modifié à tracer.")
        return

    indices = np.random.default_rng(GRAINE_ALEATOIRE).choice(
        len(profils_modifies), min(len(profils_modifies), NB_COURBES_A_TRACER), replace=False
    )
    x = temps * 1e9
    moyenne = np.mean(profils_modifies, axis=0)
    ecart_type = np.std(profils_modifies, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for i, indice in enumerate(indices):
        axes[0].plot(
            x, profils_modifies[indice], color="tab:blue", alpha=0.2,
            linewidth=0.8, label="Profils modifiés" if i == 0 else None
        )
    axes[0].plot(
        x, profil_original, color="black", linewidth=2,
        label="Profil original"
    )
    axes[0].set_title(
        f"Profil original et {len(indices)} profils modifiés"
    )

    axes[1].fill_between(
        x, moyenne - ecart_type, moyenne + ecart_type,
        color="tab:blue", alpha=0.2, label="Moyenne ± un écart-type"
    )
    axes[1].plot(x, moyenne, color="tab:blue", label="Moyenne des modifications")
    axes[1].plot(
        x, profil_original, color="black", linewidth=2,
        label="Profil original"
    )
    axes[1].set_title(
        f"Statistiques sur les {len(profils_modifies)} profils disponibles"
    )
    for ax in axes:
        ax.set_xlabel("Temps (ns)")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylabel("Puissance")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(FICHIER_FIGURE, dpi=200)
    print(f"Courbes sauvegardées : {FICHIER_FIGURE}")
    plt.show()
    plt.close(fig)


# ============================================================
# 5. SAUVEGARDE
# ============================================================

def tracer_dataset_hierarchique(dataset, repertoire, cas):
    """Deux vues du .npy, limitées à six familles et six variantes par famille."""
    rng = np.random.default_rng(GRAINE_ALEATOIRE)
    n = len(dataset["original"])
    selection = np.sort(rng.choice(n, min(n, NB_COURBES_A_TRACER), replace=False))
    temps = np.asarray(dataset["temps"]) * 1e9
    variantes = {}
    for i in selection:
        groupe = dataset["update"][i]
        debut = 1 if np.array_equal(groupe[0], dataset["original"][i]) else 0
        candidats = np.arange(debut, len(groupe))
        variantes[i] = rng.choice(candidats, min(len(candidats), NB_COURBES_A_TRACER), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "reference_initiale" in dataset:
        axes[0].plot(temps, dataset["reference_initiale"], color="black",
                     linewidth=2.5, linestyle="--", label="Référence initiale")
    couleurs = plt.get_cmap("tab10")
    for k, i in enumerate(selection):
        original = dataset["original"][i]
        couleur = couleurs(k)
        axes[0].plot(temps, original, color=couleur, label=f"Parent {i}")
        for j in variantes[i]:
            axes[1].plot(temps, dataset["update"][i][j], color=couleur, alpha=0.3)
        axes[1].plot(temps, original, color=couleur, linewidth=2, label=f"Parent {i}")
    axes[0].set_title(f"{len(selection)} parents tirés au sort sur {n}")
    axes[1].set_title(f"Au plus {NB_COURBES_A_TRACER} variantes par parent sélectionné")
    for ax in axes:
        ax.set_xlabel("Temps (ns)"); ax.set_ylabel("Puissance (TW)")
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(Path(repertoire) / f"dataset_{cas}_courbes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots((len(selection) + 1) // 2, 2,
                             figsize=(15, 4 * ((len(selection) + 1) // 2)),
                             squeeze=False, sharex=True)
    for ax, i in zip(axes.flat, selection):
        for k, j in enumerate(variantes[i]):
            ax.plot(temps, dataset["update"][i][j], color="tab:blue", alpha=0.4,
                    label="Variantes fines" if k == 0 else None)
        ax.plot(temps, dataset["original"][i], color="red", linewidth=2, label="Parent")
        nom = dataset["provenance_groupes"][i].get("parent_grossier", str(i))
        ax.set_title(("Référence initiale" if nom == "rep_0000" else nom)
                     + f" — {len(variantes[i])} variantes affichées")
        ax.set_xlabel("Temps (ns)"); ax.set_ylabel("Puissance (TW)")
        ax.grid(alpha=0.3); ax.legend()
    for ax in list(axes.flat)[len(selection):]:
        ax.set_visible(False)
    fig.suptitle(f"Contenu de dataset_{cas}.npy — sélection aléatoire")
    fig.tight_layout()
    fig.savefig(Path(repertoire) / f"dataset_{cas}_contenu.png", dpi=160)
    plt.close(fig)


def creer_dataset_hierarchique(repertoire_travail, cas):
    """Un original grossier par groupe, toutes ses variantes raffinées en sortie.

    Reconstruit le dataset sans ajouter les anciens groupes. Les .dat sont en
    secondes ; tous les profils utilisent le même axe, couvrant toutes les durées.
    """
    import json
    repertoire = Path(repertoire_travail)
    parents = sorted(p for p in (repertoire / "RES").glob("rep_*") if p.is_dir())
    if not parents:
        raise FileNotFoundError("Aucune famille grossière trouvée.")
    familles = []
    tous_fichiers = []
    for parent in parents:
        enfants = sorted((parent / "raffines").glob("rep_*/shot.dat"))
        if not enfants:
            raise ValueError(f"Aucun profil raffiné pour {parent}")
        fichiers = [parent / "shot.dat"] + enfants
        for fichier in fichiers:
            if not fichier.is_file():
                raise FileNotFoundError(fichier)
        familles.append((parent, enfants))
        tous_fichiers.extend(fichiers)
    reference = repertoire / f"shot_{cas}.dat"
    if reference.is_file():
        tous_fichiers.append(reference)
    debut, fin = float("inf"), -float("inf")
    for fichier in tous_fichiers:
        donnees = np.loadtxt(fichier, ndmin=2)
        if donnees.shape[0] < 2 or donnees.shape[1] < 2 or not np.isfinite(donnees[:, :2]).all():
            raise ValueError(f"Profil invalide : {fichier}")
        if not np.all(np.diff(donnees[:, 0]) > 0):
            raise ValueError(f"Temps non croissants : {fichier}")
        debut = min(debut, donnees[0, 0])
        fin = max(fin, donnees[-1, 0])
    temps = np.linspace(debut, fin, NB_POINTS)
    dataset = dict(original=[], update=[], provenance_groupes=[], temps=temps,
                   cas=cas, structure="grossier_raffine")
    for parent, enfants in familles:
        original = charger_profil_dat(parent / "shot.dat", temps)
        dataset["original"].append(original)
        dataset["update"].append(np.stack(
            [original.copy()] + [charger_profil_dat(f, temps) for f in enfants]
        ))
        parametres = parent / "parametres.json"
        dataset["provenance_groupes"].append(dict(
            cas=cas, parent_grossier=parent.name,
            fichier_original=str((parent / "shot.dat").resolve()),
            fichiers_raffines=[str(f.resolve()) for f in enfants],
            fichiers_sorties=[str((parent / "shot.dat").resolve())] + [str(f.resolve()) for f in enfants],
            types_sorties=["reference_parent"] + ["raffine"] * len(enfants),
            parametres_grossiers=json.loads(parametres.read_text()) if parametres.exists() else None,
            parametres_sorties=[
                json.loads(f.with_name("parametres.json").read_text())
                for f in [parent / "shot.dat"] + enfants
            ],
        ))
    if reference.is_file():
        dataset["reference_initiale"] = charger_profil_dat(reference, temps)
        dataset["fichier_reference_initiale"] = str(reference.resolve())
    fichier_sortie = repertoire / f"dataset_{cas}.npy"
    np.save(fichier_sortie, dataset, allow_pickle=True)
    tracer_dataset_hierarchique(dataset, repertoire, cas)
    print(f"Dataset : {len(familles)} originaux, {sum(map(len, dataset['update']))} sorties (références incluses)", flush=True)
    return fichier_sortie



def main():
    if any(DOSSIER_RES.glob("rep_*/raffines/rep_*/shot.dat")):
        creer_dataset_hierarchique(REPERTOIRE_TRAVAIL, CAS)
        return
    # Charger les profils
    profil_original, temps = charger_profil_dat(FICHIER_DAT, retourner_temps=True)
    profils_modifies = charger_profils_res(DOSSIER_RES, temps)

    # Créer les nouveaux groupes
    nouveaux = creer_dataset(
        profil_original,
        profils_modifies
    )

    # Ajouter aux données existantes si le fichier existe
    if os.path.isfile(FICHIER_SORTIE):
        ancien = np.load(
            FICHIER_SORTIE,
            allow_pickle=True
        ).item()

        if "original" not in ancien or "update" not in ancien:
            raise ValueError(
                "Le fichier de sortie existant ne contient pas "
                "les clés 'original' et 'update'."
            )

        ancien["original"].extend(nouveaux["original"])
        ancien["update"].extend(nouveaux["update"])

        dataset = ancien

        print(
            f"Dataset existant chargé : "
            f"{len(ancien['update']) - len(nouveaux['update'])} groupes."
        )
    else:
        dataset = nouveaux

    # Sauvegarder le dataset mis à jour
    np.save(
        FICHIER_SORTIE,
        dataset,
        allow_pickle=True
    )

    nb_groupes = len(dataset["update"])
    nb_nouveaux = len(nouveaux["update"])
    nb_profils_ajoutes = sum(len(groupe) for groupe in nouveaux["update"])

    print(f"Fichier sauvegardé : {FICHIER_SORTIE}")
    print(f"Nombre total de groupes : {nb_groupes}")
    print(f"Nouveaux groupes ajoutés : {nb_nouveaux}")
    print(f"Nombre de profils ajoutés : {nb_profils_ajoutes}")
    print("Profils ignorés : 0")

    tracer_courbes(profil_original, profils_modifies, temps)


if __name__ == "__main__":
    main()
