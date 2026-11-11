
# generateur_profils_laser.py

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

_DOSSIER = os.path.dirname(os.path.abspath(__file__))
FICHIER_DATASET = os.path.join(_DOSSIER, "BaseDeDonnees", "dataset_Choc_CAS1", "dataset_Choc_CAS1.npy")
DOSSIER_SORTIE = os.path.join(_DOSSIER, "modele_laser")

FICHIER_MODELE = os.path.join(DOSSIER_SORTIE, "cvae_laser.keras")
FICHIER_GENERES = os.path.join(DOSSIER_SORTIE, "profils_generes.npy")

NB_POINTS = 512
TAILLE_GROUPE = 21
LATENT_DIM = 16

BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 1e-3
BETA_KL = 1e-3

NB_GENERATIONS = 50
INDICE_GROUPE = 0

GRAINE = 42

np.random.seed(GRAINE)
tf.random.set_seed(GRAINE)


# ============================================================
# CHARGEMENT DU DATASET
# ============================================================

def charger_dataset(chemin):
    if not os.path.isfile(chemin):
        raise FileNotFoundError(
            f"Dataset introuvable : {chemin}"
        )

    dataset = np.load(
        chemin,
        allow_pickle=True
    ).item()

    if "original" not in dataset or "update" not in dataset:
        raise KeyError(
            "Le fichier doit contenir les clés "
            "'original' et 'update'."
        )

    originals = np.asarray(
        dataset["original"],
        dtype=np.float32
    )

    updates = np.asarray(
        dataset["update"],
        dtype=np.float32
    )

    if originals.ndim != 2 or originals.shape[1] != NB_POINTS:
        raise ValueError(
            f"'original' doit avoir la forme (N, {NB_POINTS}), "
            f"forme reçue : {originals.shape}"
        )

    if (
        updates.ndim != 3
        or updates.shape[0] != originals.shape[0]
        or updates.shape[1] != TAILLE_GROUPE
        or updates.shape[2] != NB_POINTS
    ):
        raise ValueError(
            f"'update' doit avoir la forme "
            f"(N, {TAILLE_GROUPE}, {NB_POINTS}), "
            f"forme reçue : {updates.shape}"
        )

    if len(originals) == 0:
        raise ValueError("Le dataset ne contient aucun groupe.")

    if (
        not np.isfinite(originals).all()
        or not np.isfinite(updates).all()
    ):
        raise ValueError(
            "Le dataset contient des valeurs NaN ou infinies."
        )

    # Répéter chaque profil original pour ses 50 modifications
    x_original = np.repeat(
        originals[:, np.newaxis, :],
        TAILLE_GROUPE,
        axis=1
    )

    # Aplatir les groupes :
    # une ligne = une paire original / modifié
    x_original = x_original.reshape(-1, NB_POINTS)
    x_modified = updates.reshape(-1, NB_POINTS)

    print("\nDataset chargé")
    print("------------------------------")
    print(f"Nombre de groupes : {len(originals)}")
    print(f"Nombre de paires : {len(x_original)}")
    print(f"Forme des originaux : {x_original.shape}")
    print(f"Forme des modifications : {x_modified.shape}")

    return originals, updates, x_original, x_modified


# ============================================================
# CONSTRUCTION DU CVAE
# ============================================================

def construire_cvae():
    # --------------------------------------------------------
    # Encodeur : q(z | original, modified)
    # --------------------------------------------------------

    original_input = keras.Input(
        shape=(NB_POINTS,),
        name="original_encodeur"
    )

    modified_input = keras.Input(
        shape=(NB_POINTS,),
        name="modified_encodeur"
    )

    x = layers.Concatenate()(
        [original_input, modified_input]
    )

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    z_mean = layers.Dense(
        LATENT_DIM,
        name="z_mean"
    )(x)

    z_log_var = keras.layers.Dense(
        LATENT_DIM,
        name="z_log_var_brut"
    )(x)

    z_log_var = keras.layers.Lambda(
        lambda t: tf.clip_by_value(t, -10.0, 10.0),
        name="z_log_var"
    )(z_log_var)

    def echantillonner(args):
        mean, log_var = args
        epsilon = tf.random.normal(
            shape=tf.shape(mean)
        )
        return mean + tf.exp(0.5 * log_var) * epsilon

    z = layers.Lambda(
        echantillonner,
        name="echantillonnage"
    )([z_mean, z_log_var])

    encodeur = keras.Model(
        [original_input, modified_input],
        [z_mean, z_log_var, z],
        name="encodeur"
    )

    # --------------------------------------------------------
    # Décodeur : p(modified | original, z)
    # --------------------------------------------------------

    condition_input = keras.Input(
        shape=(NB_POINTS,),
        name="condition_original"
    )

    latent_input = keras.Input(
        shape=(LATENT_DIM,),
        name="latent"
    )

    x = layers.Concatenate()(
        [condition_input, latent_input]
    )

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)

    output = layers.Dense(
        NB_POINTS,
        activation="linear",
        name="profil_genere"
    )(x)

    decodeur = keras.Model(
        [condition_input, latent_input],
        output,
        name="decodeur"
    )

    # --------------------------------------------------------
    # CVAE avec fonction de perte compatible Keras 3
    # --------------------------------------------------------

    class CVAE(keras.Model):
        def __init__(self, encodeur, decodeur):
            super().__init__(name="cvae_laser")
            self.encodeur = encodeur
            self.decodeur = decodeur
            self.beta_kl = BETA_KL

            self.loss_tracker = keras.metrics.Mean(
                name="loss"
            )
            self.reconstruction_tracker = keras.metrics.Mean(
                name="reconstruction_loss"
            )
            self.kl_tracker = keras.metrics.Mean(
                name="kl_loss"
            )

        @property
        def metrics(self):
            return [
                self.loss_tracker,
                self.reconstruction_tracker,
                self.kl_tracker,
            ]
        
        def train_step(self, data):
            inputs, targets, sample_weight = (
                keras.utils.unpack_x_y_sample_weight(data)
            )
            original, modified = inputs

            with tf.GradientTape() as tape:
                reconstruction, z_mean, z_log_var = self(
                    [original, modified], training=True
                )

                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_mean(
                        tf.square(modified - reconstruction),
                        axis=-1
                    )
                )

                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(
                        1.0 + z_log_var
                        - tf.square(z_mean)
                        - tf.exp(z_log_var),
                        axis=-1
                    )
                )

                loss = reconstruction_loss + self.beta_kl * kl_loss

            gradients = tape.gradient(
                loss, self.trainable_variables
            )
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables)
            )

            self.loss_tracker.update_state(loss)
            self.reconstruction_tracker.update_state(reconstruction_loss)
            self.kl_tracker.update_state(kl_loss)

            return {
                "loss": self.loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_tracker.result(),
                "kl_loss": self.kl_tracker.result(),
            }

        def test_step(self, data):
            inputs, targets, sample_weight = (
                keras.utils.unpack_x_y_sample_weight(data)
            )
            original, modified = inputs

            reconstruction, z_mean, z_log_var = self(
                [original, modified], training=False
            )

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_mean(
                    tf.square(modified - reconstruction),
                    axis=-1
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1.0 + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),
                    axis=-1
                )
            )

            loss = reconstruction_loss + self.beta_kl * kl_loss

            self.loss_tracker.update_state(loss)
            self.reconstruction_tracker.update_state(reconstruction_loss)
            self.kl_tracker.update_state(kl_loss)

            return {
                "loss": self.loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_tracker.result(),
                "kl_loss": self.kl_tracker.result(),
            }
            
        def call(self, inputs, training=False):
            original, modified = inputs

            # Encoder la paire original / modifié
            z_mean, z_log_var, z = self.encodeur(
                [original, modified],
                training=training
            )

            # Reconstruire un profil modifié à partir de l'original et du latent
            reconstruction = self.decodeur(
                [original, z],
                training=training
            )

            return reconstruction, z_mean, z_log_var
    
    cvae = CVAE(encodeur, decodeur)

    cvae.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )
    )

    return cvae, encodeur, decodeur


# ============================================================
# ENTRAINEMENT
# ============================================================

def entrainer(cvae, x_original, x_modified):
    nb = len(x_original)

    rng = np.random.default_rng(GRAINE)
    indices = rng.permutation(nb)

    x_original = x_original[indices]
    x_modified = x_modified[indices]

    if nb >= 10:
        nb_val = max(1, int(0.1 * nb))

        x_orig_train = x_original[:-nb_val]
        x_mod_train = x_modified[:-nb_val]

        x_orig_val = x_original[-nb_val:]
        x_mod_val = x_modified[-nb_val:]

        validation_data = (
            [x_orig_val, x_mod_val],
            x_mod_val
        )
    else:
        x_orig_train = x_original
        x_mod_train = x_modified
        validation_data = None

    historique = cvae.fit(
        [x_orig_train, x_mod_train],
        x_mod_train,
        validation_data=validation_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1
    )

    return historique


# ============================================================
# GENERATION DE PROFILS
# ============================================================

def generer_profils(decodeur, profil_original, nb_generations=50):
    profil_original = np.asarray(
        profil_original,
        dtype=np.float32
    )

    if profil_original.shape != (NB_POINTS,):
        raise ValueError(
            f"Le profil de référence doit contenir "
            f"{NB_POINTS} valeurs."
        )

    condition = np.repeat(
        profil_original[np.newaxis, :],
        nb_generations,
        axis=0
    )

    # Tirages latents indépendants
    z = np.random.normal(
        size=(nb_generations, LATENT_DIM)
    ).astype(np.float32)

    profils = decodeur.predict(
        [condition, z],
        verbose=0
    )

    return profils


def tracer_evolution_loss(historique, dossier_sortie):
    """
    Trace l'évolution logarithmique des pertes pendant l'entraînement.
    """
    hist = historique.history
    epochs = np.arange(1, len(hist["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        epochs,
        hist["loss"],
        label="Loss entraînement",
        linewidth=2
    )

    if "val_loss" in hist:
        ax.plot(
            epochs,
            hist["val_loss"],
            label="Loss validation",
            linewidth=2
        )

    ax.set_yscale("log")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Loss (échelle logarithmique)")
    ax.set_title("Évolution de la loss du CVAE")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    chemin = os.path.join(dossier_sortie, "loss_log.png")
    fig.savefig(chemin, dpi=200)
    plt.show()
    plt.close(fig)

    print(f"Courbe de loss sauvegardée : {chemin}")

def comparer_profils_test(
    decodeur,
    profil_original,
    updates_reels,
    indice_groupe, dossier_sortie):
    """
    Compare les modifications réelles d'un groupe de test
    aux profils générés par le CVAE pour le même original.
    """

    updates_reels = np.asarray(updates_reels, dtype=np.float32)

    # Générer autant de profils que de modifications réelles
    updates_predits = generer_profils(
        decodeur,
        profil_original,
        nb_generations=len(updates_reels)
    )

    x = np.arange(NB_POINTS)

    # Statistiques des profils
    moyenne_reelle = np.mean(updates_reels, axis=0)
    moyenne_predite = np.mean(updates_predits, axis=0)

    ecart_type_reel = np.std(updates_reels, axis=0)
    ecart_type_predit = np.std(updates_predits, axis=0)

    # Indicateurs de comparaison des distributions
    rmse_moyenne = np.sqrt(
        np.mean((moyenne_reelle - moyenne_predite) ** 2)
    )

    rmse_ecart_type = np.sqrt(
        np.mean((ecart_type_reel - ecart_type_predit) ** 2)
    )

    print(f"\nComparaison du groupe de test {indice_groupe}")
    print("--------------------------------------------")
    print(f"RMSE des profils moyens : {rmse_moyenne:.6f}")
    print(f"RMSE des écarts-types : {rmse_ecart_type:.6f}")

    # Figure de comparaison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Modifications réelles
    ax = axes[0, 0]
    ax.plot(
        x, profil_original,
        color="black", linewidth=2,
        label="Profil original"
    )

    for i, profil in enumerate(updates_reels):
        ax.plot(
            x, profil,
            alpha=0.20, linewidth=0.8,
            label="Updates réels" if i == 0 else None
        )

    ax.set_title("Jeu de test : modifications réelles")
    ax.set_xlabel("Indice du point")
    ax.set_ylabel("Puissance")
    ax.legend()
    ax.grid(alpha=0.25)

    # 2. Modifications générées
    ax = axes[0, 1]
    ax.plot(
        x, profil_original,
        color="black", linewidth=2,
        label="Profil original"
    )

    for i, profil in enumerate(updates_predits):
        ax.plot(
            x, profil,
            alpha=0.20, linewidth=0.8,
            label="Updates générés" if i == 0 else None
        )

    ax.set_title("CVAE : modifications générées")
    ax.set_xlabel("Indice du point")
    ax.set_ylabel("Puissance")
    ax.legend()
    ax.grid(alpha=0.25)

    # 3. Comparaison des profils moyens
    ax = axes[1, 0]
    ax.plot(
        x, moyenne_reelle,
        label="Moyenne réelle", linewidth=2
    )
    ax.plot(
        x, moyenne_predite,
        label="Moyenne générée", linewidth=2,
        linestyle="--"
    )
    ax.plot(
        x, profil_original,
        label="Original", color="black", alpha=0.7
    )
    ax.set_title(f"Profils moyens — RMSE = {rmse_moyenne:.4g}")
    ax.set_xlabel("Indice du point")
    ax.set_ylabel("Puissance")
    ax.legend()
    ax.grid(alpha=0.25)

    # 4. Comparaison de la dispersion
    ax = axes[1, 1]
    ax.plot(
        x, ecart_type_reel,
        label="Écart-type réel", linewidth=2
    )
    ax.plot(
        x, ecart_type_predit,
        label="Écart-type généré", linewidth=2,
        linestyle="--"
    )
    ax.set_title(f"Dispersion — RMSE = {rmse_ecart_type:.4g}")
    ax.set_xlabel("Indice du point")
    ax.set_ylabel("Écart-type de la puissance")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Comparaison sur le groupe de test {indice_groupe}",
        fontsize=14
    )
    fig.tight_layout()

    chemin = os.path.join(
        dossier_sortie,
        f"comparaison_test_groupe_{indice_groupe}.png"
    )

    fig.savefig(chemin, dpi=200)
    plt.show()
    plt.close(fig)

    # Sauvegarder aussi les données comparées
    np.savez(
        os.path.join(
            dossier_sortie,
            f"comparaison_test_groupe_{indice_groupe}.npz"
        ),
        original=profil_original,
        updates_reels=updates_reels,
        updates_predits=updates_predits,
        moyenne_reelle=moyenne_reelle,
        moyenne_predite=moyenne_predite,
        ecart_type_reel=ecart_type_reel,
        ecart_type_predit=ecart_type_predit
    )

    print(f"Comparaison sauvegardée : {chemin}")

    return updates_predits

# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    # Charger les données
    originals, updates, x_original, x_modified = (
        charger_dataset(FICHIER_DATASET)
    )

    print("\nDiagnostic des données")
    print("-----------------------")

    for nom, tableau in [
        ("x_original", x_original),
        ("x_modified", x_modified),
    ]:
        print(f"\n{nom}")
        print("Min :", np.min(tableau))
        print("Max :", np.max(tableau))
        print("Moyenne :", np.mean(tableau))
        print("Écart-type :", np.std(tableau))
        print("Valeurs non finies :", np.count_nonzero(~np.isfinite(tableau)))

        # --------------------------------------------------------
    # Séparation par groupes : aucun groupe de test à l'entraînement
    # --------------------------------------------------------

    rng = np.random.default_rng(GRAINE)
    indices_groupes = rng.permutation(len(originals))

    nb_test = max(1, int(0.2 * len(originals)))

    indices_test = indices_groupes[:nb_test]
    indices_train = indices_groupes[nb_test:]

    originals_train = originals[indices_train]
    updates_train = updates[indices_train]

    print("\nSéparation entraînement / test")
    print("--------------------------------")
    print(f"Groupes d'entraînement : {len(indices_train)}")
    print(f"Groupes de test : {len(indices_test)}")

    # Aplatir uniquement les groupes d'entraînement
    x_orig_train = np.repeat(
        originals_train[:, np.newaxis, :],
        TAILLE_GROUPE,
        axis=1
    ).reshape(-1, NB_POINTS)

    x_mod_train = updates_train.reshape(-1, NB_POINTS)

    # Construire et entraîner le modèle
    cvae, encodeur, decodeur = construire_cvae()

    print("\nDébut de l'entraînement...")

    historique = entrainer(
        cvae,
        x_orig_train,
        x_mod_train
    )

    # Tracer la loss en échelle logarithmique
    tracer_evolution_loss(
        historique,
        DOSSIER_SORTIE
    )

    # Sauvegarder les modèles
    # MODIF RNA_Basique : avec Keras 3, cvae.save() plante (pas de get_config),
    # ce qui arrêtait le script avant la comparaison. On garde la sauvegarde
    # d'origine quand elle marche, sinon on enregistre seulement les poids.
    try:
        cvae.save(FICHIER_MODELE)
        fichier_sauve = FICHIER_MODELE
    except Exception as erreur:
        print(f"cvae.save impossible ({type(erreur).__name__}) : sauvegarde des poids seulement")
        if os.path.exists(FICHIER_MODELE):
            os.remove(FICHIER_MODELE)   # fichier vide laissé par l'échec
        fichier_sauve = os.path.join(DOSSIER_SORTIE, "cvae_laser.weights.h5")
        cvae.save_weights(fichier_sauve)

    encodeur.save(
        os.path.join(DOSSIER_SORTIE, "encodeur.keras")
    )

    decodeur.save(
        os.path.join(DOSSIER_SORTIE, "decodeur.keras")
    )

    print(f"\nModèle sauvegardé : {fichier_sauve}")   # MODIF RNA_Basique

    # --------------------------------------------------------
    # Comparaison sur un groupe jamais vu pendant l'entraînement
    # --------------------------------------------------------

    indice_test = int(indices_test[0])

    profil_test = originals[indice_test]
    updates_test = updates[indice_test]

    profils_generes = comparer_profils_test(
        decodeur=decodeur,
        profil_original=profil_test,
        updates_reels=updates_test,
        indice_groupe=indice_test,
        dossier_sortie=DOSSIER_SORTIE
    )

    # Sauvegarder les profils générés du groupe de test
    np.save(
        FICHIER_GENERES,
        profils_generes
    )

    print("\nGénération et comparaison terminées")
    print("------------------------------------")
    print(f"Groupe de test : {indice_test}")
    print(f"Nombre de profils générés : {len(profils_generes)}")
    print(f"Forme du résultat : {profils_generes.shape}")
    print(f"Fichier : {FICHIER_GENERES}")

if __name__ == "__main__":
    main()
