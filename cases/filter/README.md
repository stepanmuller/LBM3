# Model válce Dode thick (4mm)

*Poznámka*: Všechny STL soubory jsou centrovány na střed a orientovány v ose Z.

### Původní váleček bez obalení

Odpovídající STL soubor: *geometry.stl*

- x = 0.02m

- y = 0.02m

- z = 0.025m

Ukázka přesného obalení pevnými stěnami do válce je v souboru *dode-with-cylinder-exact_size.stl*; jsou zde vidět nepřesnosti/přečnívající kusy, proto uměle zvětšený obal dále.

### Formulace úlohy 1: 102% válec

Odpovídající STL soubor: *dode-with-cylinder-102.stl*

<u>Parametry výpočetní oblasti</u>:

- x = 0.0204m

- y = 0.0204m

- z = 0.075m

- kinematická viskozita ν = 1.5e-5 m^2/s (vzduch)

    - při hustotě ρ = 1.225 kg/m^3 je dynamická viskozita μ = 1.8375 kg/(m⋅s)

- dvě hodnoty toku *Q*:

    - 1 l/min => rychlost proudění na vstupu v = 0.05099 m/s

    - 5 l/min => rychlost proudění na vstupu v = 0.25495 m/s

- finální čas simulace T<sub>fin</sub> = 5s

    - zvolen arbitrárně (dostatečně) veliký, aby proudění mělo čas se ustálit; pokud je implementováno adaptivní ukončení po zkonvergování, tak dříve

- pevné stěny podél oblasti, na vstupu fixní rychlost (konstantní profil), na výstupu nulový tlak

### Formulace úlohy 1: 105% válec

Odpovídající STL soubor: *dode-with-cylinder-105.stl*

<u>Parametry výpočetní oblasti</u>:

- x = 0.021m

- y = 0.021m

- z = 0.075m

- kinematická viskozita ν = 1.5e-5 m^2/s (vzduch)

    - při hustotě ρ = 1.225 kg/m^3 je dynamická viskozita μ = 1.8375 kg/(m⋅s)

- dvě hodnoty toku *Q*:

    - 1 l/min => rychlost proudění na vstupu v = 0.04812 m/s

    - 5 l/min => rychlost proudění na vstupu v = 0.2406 m/s

- finální čas simulace T<sub>fin</sub> = 5s

    - zvolen arbitrárně (dostatečně) veliký, aby proudění mělo čas se ustálit; pokud je implementováno adaptivní ukončení po zkonvergování, tak dříve

- pevné stěny podél oblasti, na vstupu fixní rychlost (konstantní profil), na výstupu nulový tlak

### Měření dat a výstup

Tlak se měří v příčném plošném řezu ve vzdálenosti 0.0125mm od levého kraje - vstupu (tedy před překážkami, v polovině volné vzdálenosti), v tomto řezu se spočítá prostorový průměr a vypíše se spolu s timestampem a iterací do souboru (např. CSV) pro postprocessing (časový vývoj, průměry, ...). Frekvenci vzorkování je vhodné volit tak, aby tvořily reprezentativní vzorek z celé simulace.

Jelikož je na odtoku nulový tlak, je tato hodnota (po zkonvergování) přímo velikost tlakové ztráty.
