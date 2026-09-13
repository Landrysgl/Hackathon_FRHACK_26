# Analyse critique des trois candidats finaux

Les trois cas ci-dessous ont été choisis parmi les candidats **déjà revus comme `plausible`** et dont le type visuel a été identifié. Ils servent d'exemples pour la restitution. Aucun n'est présenté comme un site radioélectrique non déclaré confirmé.

## D003 — mât

- confiance YOLO : **0,489** ;
- distance au support ANFR le plus proche : **155,6 m** ;
- coordonnées : **48.801306, 2.125993** ;
- support ANFR le plus proche : `835052`, nature ANFR `Immeuble` ;
- type visuel après revue : **mât**.

D003 est le cas le plus intéressant du trio car l'objet détecté a l'apparence d'une structure verticale fine et a été identifié visuellement comme un mât. Sa distance dépasse le seuil conservateur de 150 m utilisé par l'outil.

La conclusion reste volontairement limitée : le mât peut être sans équipement radio, la date de l'orthophoto peut différer de celle de la base, ou la correspondance administrative peut être décalée. Une vérification Cartoradio et, si nécessaire, une source plus récente sont requises.

Cartoradio : https://www.cartoradio.fr/#/cartographie/lonlat/2.125993/48.801306

## D006 — bâtiment

- confiance YOLO : **0,400** ;
- distance au support ANFR le plus proche : **162,3 m** ;
- coordonnées : **48.799445, 2.130660** ;
- support ANFR le plus proche : `2284920`, nature ANFR `Bâtiment` ;
- type visuel après revue : **bâtiment**.

D006 a été conservé car la détection est visuellement plausible et reste au-delà du seuil de distance. Il est cependant moins spécifique qu'un mât ou un pylône : un bâtiment est une structure très courante et peut être détecté sans qu'une installation radio soit réellement visible.

Ce cas illustre donc aussi une limite utile de l'outil : une détection crédible au sens visuel n'est pas nécessairement un support radio.

Cartoradio : https://www.cartoradio.fr/#/cartographie/lonlat/2.130660/48.799445

## D007 — bâtiment

- confiance YOLO : **0,321** ;
- distance au support ANFR le plus proche : **250,7 m** ;
- coordonnées : **48.794343, 2.130314** ;
- support ANFR le plus proche : `794061`, nature ANFR `Château d'eau - réservoir` ;
- type visuel après revue : **bâtiment**.

D007 présente la plus grande distance ANFR du trio, ce qui en fait un bon exemple pour démontrer le filtrage géographique. En contrepartie, sa confiance est plus faible et le support visuel est un bâtiment ; son intérêt vient donc davantage de l'écart géographique que d'une signature visuelle très spécifique.

Ce candidat doit être présenté comme un cas à **prioriser pour vérification**, et non comme une anomalie avérée.

Cartoradio : https://www.cartoradio.fr/#/cartographie/lonlat/2.130314/48.794343

## Limite de la sélection

Les trois candidats finaux se trouvent dans la zone dense. La sélection est destinée à illustrer les sorties les plus lisibles de l'outil, pas à représenter statistiquement les trois contextes étudiés. Les résultats multizones complets restent disponibles dans `summary_by_zone.csv` et montrent justement que le taux de faux positifs augmente fortement dans les zones moins denses.
