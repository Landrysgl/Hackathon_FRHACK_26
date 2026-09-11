import boto3
import pandas as pd
import matplotlib.pyplot as plt


s3 = boto3.client("s3")

bucket = "landry"
key = "Challenge 4 - Antennes sur images/Données data.gouv/data/dataset_antennes/SUP_SUPPORT.txt"

obj = s3.get_object(Bucket=bucket, Key=key)

df = pd.read_csv(obj["Body"], sep=";", encoding="utf-8")


association_id_type = {
    0: "Sans nature",
    40: "Sémaphore",
    41: "Phare",
    4: "Château d'eau - réservoir",
    38: "Immeuble",
    39: "Local technique",
    42: "Mât",
    8: "Intérieur galerie",
    9: "Intérieur sous-terrain",
    10: "Tunnel",
    11: "Mât béton",
    12: "Mât métallique",
    21: "Pylône",
    17: "Bâtiment",
    19: "Monument historique",
    20: "Monument religieux",
    22: "Pylône autoportant",
    23: "Pylône autostable",
    24: "Pylône haubané",
    25: "Pylône treillis",
    26: "Pylône tubulaire",
    31: "Silo",
    32: "Ouvrage d'art (pont, viaduc)",
    33: "Tour hertzienne",
    34: "Dalle en béton",
    999999999: "Support non décrit",
    43: "Fût",
    44: "Tour de contrôle",
    45: "Contre-poids au sol",
    46: "Contre-poids sur shelter",
    47: "Support DEFENSE",
    48: "Pylône arbre",
    49: "Ouvrage de signalisation (portique routier, panneau routier)",
    50: "Balise ou bouée",
    51: "XXX",
    52: "Eolienne",
    55: "Mobilier urbain",
    56: "Roche"
}

df["TYPE"] = df["NAT_ID"].map(association_id_type).fillna("Support non décrit")

nombre_par_nature = df["TYPE"].value_counts()

print(nombre_par_nature)

plt.figure(figsize=(12, 10))

nombre_par_nature.sort_values().plot(kind="barh")

plt.xlabel("Nombre de supports")
plt.ylabel("Type de support")
plt.title("Répartition des supports par type")
plt.tight_layout()

plt.savefig("figure_montrant_la_répartition_des_supports_par_type.png", dpi=300)
plt.show()


df.to_csv(
    "SUP_SUPPORT_AVEC_TYPE.csv",
    sep=";",
    index=False,
    encoding="utf-8"
)