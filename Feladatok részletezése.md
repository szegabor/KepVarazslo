# KepVarazslo
A KépVarázsló egy mesterséges intelligencián alapuló képfeldolgozó rendszer, amely képes azonosítani ugyanarról a látványvilágról készült képeket, kiválasztani a legjobb felvételeket, és több kép felhasználásával javítani a végeredmény minőségét. Célja, hogy automatizált módszerekkel növelje a képi tartalmak vizuális minőségét és relevanciáját.


áttekintés minden  feladattípusról:

# 1. **Felismeri ugyanarról a látvány világról készült képeket**

különböző szögből, fényviszonyok között, vagy kisebb variációkkal készült képeket azonosnak ismer fel.
A pixelek közvetlen összehasonlítása nem lenne megbízható, ezért fejlettebb módszerekre van szükség.

# Megvalósítás:
- Előképzett mélytanulási modellek** (pl. **ResNet, EfficientNet**) használata jellemzők kinyerésére. A modell képes a képek magas szintű jellemzőit (például formákat, mintákat) felismerni, és kevésbé érzékeny a kisebb változásokra (pl. eltérő szög).
- Klaszterezési algoritmusok** (pl. **DBSCAN, K-means**): Miután kinyertük a képek jellemzőit, a képeket ezek alapján csoportosítjuk. A klaszterekbe sorolt képek ugyanarról a látványvilágról készültek.

- ResNet-típusú modellel jellemzők kinyerése a képekről.
- Jellemzők klaszterezése DBSCAN vagy K-means algoritmussal.

# 2. **A megegyező felvételek közül kiválasztja a legjobb felvételeket**

a megegyező látványvilágról készült több kép közül a legjobb minőségűt válasszuk ki. A minőségi szempontok közé tartozhat a kép élessége, színtelítettsége, expozíciója stb.

# Megvalósítás:
- **Minőségmérés metrikákkal**: Metrikák használata, amelyekkel a képek élessége, kontrasztja, expozíciója mérhető.
- **Élesség mérése**: Használhatunk élességmérési módszereket, mint például a **Laplacian variancia**. Minél nagyobb a variancia, annál élesebb a kép.
- **Színtelítettség és expozíció**: A hisztogramok elemzése alapján meghatározhatjuk, hogy mennyire vannak megfelelően megvilágítva és szaturálva a képek.

- A képeken Laplacian-alapú élességmérés végrehajtása.
- Színtelítettség és expozíció elemzése hisztogram elemzéssel.

# 3. **Több kép felhasználásával kísérletet tesz egy jobb kép előállítására**

több hasonló kép felhasználásával egy "kombinált" képet állít elő, ami jobb minőségű, mint az egyes eredeti képek.

# Megvalósítás:
- **Képösszevonás (Image Stacking)**: Több képet kombinálhatunk úgy, hogy a legjobb tulajdonságokat egyesítjük. A technikák közé tartozik az **átlagolás** vagy a **legélesebb részek kiválasztása** minden képből.
- **HDR képalkotás (High Dynamic Range)**: Ha különböző expozícióval készült képeink vannak, akkor a **HDR technikával** egyesíthetjük a különböző képeket, hogy nagyobb dinamikatartományú képet hozzunk létre.

- Képátlagolás minden pixelnél, vagy a legjobb pixelek kiválasztása több képből.
- HDR algoritmus használata, ha eltérő expozíciójú képekkel dolgozunk.

# 4. **Felismeri a különféle képalkotási hibákat mint elmosódás, túlexponálás stb.**

Ez a feladat a képekben található hibák (pl. homályosság, rossz expozíció) automatikus felismerésére irányul.

- **Elmosódás felismerése**: A kép **Laplacian varianciájának** elemzése segítségével mérhetjük, hogy mennyire homályos a kép. Minél alacsonyabb a variancia, annál homályosabb a kép.
- **Túlexponálás/alulexponálás**: A kép hisztogramjának elemzése: ha a képben sok nagyon világos vagy nagyon sötét pixel van, akkor valószínűleg rosszul exponált.
- **Zaj felismerése**: A kép frekvenciatartományának elemzése (pl. Fourier-transzformáció)  a zaj jelenlétének felismerésére.

- Laplacian alapú élességvizsgálat az elmosódás felismerésére.
- Hisztogram elemzése a túlexponálás és alulexponálás felismerésére.

# 5. **Javít ezek minőségén**

a felismerhető hibák kijavítására vonatkozik, mint például az elmosódás vagy a rossz expozíció javítása.

# Megvalósítás:
- **Élesség javítása**: Élességet növelő technikák,**unsharp mask** vagy élszűrők **Sobel-szűrő**.
- **Expozíció javítása**: A kép hisztogramját újra skálázzuk, hogy az árnyalatok egyenletesebben legyenek elosztva. Ehhez használhatunk **hisztogramkiegyenlítést**.
- **Zajcsökkentés**: **Gaussian blur** vagy **median filter** segítségével a képben található zajokat csökkenthetjük anélkül, hogy jelentősen csökkentenénk a részleteket.

- Képélesítés unsharp mask-alapú szűrővel.
- Hisztogramegyengetés expozíció javításához.
- Median szűrő alkalmazása zajcsökkentéshez.

# 6. **Képeket törlésre javasol, ahol nem jó a minőség**

a rossz minőségű képek (pl. homályos, túlexponált) automatikus felismerését és törlésre javaslását látja el.

# Megvalósítás:
- **Minőségi kritériumok alapján**: A képeket különböző metrikák (pl. élesség, expozíció, zajszint) alapján kiértékeljük. Ha a kép nem felel meg egy bizonyos küszöbértéknek (pl. túl homályos), akkor törlésre javasoljuk.
- **Automatizált döntéshozatal**: Ha egy kép több szempontból is rossz minőségű (pl. alacsony élesség és túlexponálás), akkor a program automatikusan törlésre javasolja.

- Képek automatikus kiértékelése élesség, expozíció és zaj alapján.
- Ha egy kép nem felel meg a minőségi küszöböknek, törlésre jelölés.

